"""
The plotter back-end holds the data set we are visualizing,
caches computed columns, and feeds the front end responses to its
queries (both for actual data, and for metadata controlling how the
data should be interpreted).
"""

import json

from pathlib import Path
from typing import Awaitable, Iterable

from aiohttp import web
from pyarrow import ArrowException, Table, parquet, types as arrow_types

from multidex.data.cleanup import cleanup
from multidex.data.field_specs import METADATA_PROPERTIES, FieldType
from multidex.data.instruments import Instrument, identify_instrument
from multidex.data.sexps import parse_sexps
from multidex.utilz.http import respond_with_file, bool_query_arg


# TODO: possibly this belongs in m.d.field_specs
METADATA_FIELD_PROPS = {
    props.field: props
    for props in METADATA_PROPERTIES
}

class Backend:
    dataset: Table
    instrument: Instrument
    browse_images: Path | None
    cached_spectra: dict[tuple[int, bool, bool, bool | tuple[str, str]],
                         dict[str, object]]

    def __init__(self, dataset: Path, browse_images: Path | None):
        self.dataset = cleanup(
            parquet.read_table(dataset),
            have_browse_images = browse_images is not None,
        )
        self.instrument = identify_instrument(self.dataset)
        self.browse_images = browse_images
        self.cached_spectra = {}

    def routes(self) -> Iterable[web.RouteDef]:
        yield web.get("/data", self.send_data)
        yield web.get("/data/columns", self.send_columns)
        yield web.get("/data/spectrum", self.send_spectrum)
        yield web.get(
            r"/browse/{img:[a-zA-Z0-9_-]+\.(?:gif|jpe?g|png|svg|tiff)}",
            self.send_browse_image
        )

    async def send_columns(self, req: web.Request) -> web.StreamResponse:
        columns = {}
        for col in self.dataset.column_names:
            if (props := METADATA_FIELD_PROPS.get(col)) is None:
                ctype = self.dataset.schema.field(col).type
                columns[col] = {
                    "meta": False,
                    "type": (
                        "quant" if arrow_types.is_floating(ctype) else "qual"
                    )
                }
            else:
                assert props.type == FieldType.ATTRIBUTE
                columns[col] = {
                    "meta": True,
                    "type": props.value_type.value
                }

        return web.Response(
            body = json.dumps(columns),
            content_type="application/json"
        )

    async def send_data(self, req: web.Request) -> web.StreamResponse:
        dataset = self.dataset

        if (raw_filter_expr := req.query.get("filter")) is not None:
            try:
                filter_exprs = list(parse_sexps(raw_filter_expr))
            except ValueError as e:
                raise web.HTTPBadRequest(
                    text = f"bad filter argument: {e}"
                ) from e

            if len(filter_exprs) != 1:
                raise web.HTTPBadRequest(
                    text = ("bad filter argument: expected 1 expression, got "
                            + str(len(filter_exprs)))
                )

            try:
                dataset = dataset.filter(filter_exprs[0].to_arrow())
            except ArrowException as e:
                raise web.HTTPBadRequest(
                    text = f"bad filter argument: {e}"
                ) from e

        if (columns := req.query.get("columns")) is not None:
            req_cols = set(columns.split(","))
            # we always send the "id" column as well
            req_cols.add("id")
            dataset = dataset.select(req_cols)

        return web.Response(
            body = json.dumps(dataset.to_pydict(), separators=(',', ':')),
            content_type="application/json"
        )

    async def send_spectrum(self, req: web.Request) -> web.StreamResponse:
        rowid = req.query.get("id")
        if rowid is None:
            raise web.HTTPBadRequest(text = (
                "400 Bad Request\n\n"
                "/data/spectrum: ?id=<n> argument required\n"
            ))
        try:
            rowid = int(rowid)
            # Table row IDs start at 1, not 0, because SQL.
            if rowid <= 0:
                raise ValueError("out of domain")
        except ValueError:
            raise web.HTTPBadRequest(text = (
                f"400 Bad Request\n\n"
                f"/data/spectrum: invalid argument {rowid!r} to 'id':"
                f" must be a positive integer\n"
            )) from None

        try:
            avg = bool_query_arg(req.query.get("avg", "true"))
        except ValueError as e:
            raise web.HTTPBadRequest(text = (
                f"400 Bad Request\n\n"
                f"/data/spectrum: invalid argument to 'avg': {e}\n"
            )) from None

        try:
            bayer = bool_query_arg(req.query.get("bayer", "true"))
        except ValueError as e:
            raise web.HTTPBadRequest(text = (
                f"400 Bad Request\n\n"
                f"/data/spectrum: invalid argument to 'bayer': {e}\n"
            )) from None

        raw_scale = req.query.get("scale", "true")
        try:
            scale = bool_query_arg(raw_scale)
        except ValueError:
            try:
                left, right = raw_scale.split(",")
                scale = (left, right)
            except ValueError:
                raise web.HTTPBadRequest(text = (
                    f"400 Bad Request\n\n"
                    f"/data/spectrum: invalid argument to 'scale':"
                    f" cannot parse {raw_scale!r} as boolean or filter pair\n"
                )) from None

        # can't use functools.lru_cache because it would try to put the
        # dataset into the cache key
        cache_key = (rowid, avg, bayer, scale)
        if (spectrum := self.cached_spectra.get(cache_key)) is None:
            try:
                spectrum = self.instrument.compute_spectrum(
                    rowid, self.dataset,
                    scale = scale,
                    average_filters = avg,
                    show_bayers = bayer,
                )
            except (TypeError, ValueError) as e:
                raise web.HTTPBadRequest(text = (
                    f"400 Bad Request\n\n"
                    f"/data/spectrum: invalid arguments: {e}\n"
                )) from None
            except Exception as e:
                raise web.HTTPInternalServerError(text = (
                    f"500 Internal Server Error\n\n"
                    f"/data/spectrum: calculation failed: {e}\n"
                )) from None

            self.cached_spectra[cache_key] = spectrum

        return web.Response(
            body = json.dumps(spectrum, separators=(',', ':')),
            content_type="application/json"
        )

    def send_browse_image(
        self, req: web.Request
    ) -> Awaitable[web.StreamResponse]:
        if self.browse_images is None:
            raise web.HTTPNotFound(
              text=f"{req.rel_url}: 404 Not Found: browse images not available"
            )
        return respond_with_file(
            req,
             filename = req.match_info["img"],
             parent_dir = self.browse_images
        )
