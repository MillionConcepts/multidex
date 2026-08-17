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

from multidex.data.field_specs import METADATA_PROPERTIES, FieldType
from multidex.data.sexps import parse_sexps
from multidex.data.cleanup import cleanup
from multidex.utilz.http import respond_with_file


# TODO: possibly this belongs in m.d.field_specs
METADATA_FIELD_PROPS = {
    props.field: props
    for props in METADATA_PROPERTIES
}

class Backend:
    dataset: Table
    browse_images: Path | None

    def __init__(self, dataset: Path, browse_images: Path | None):
        self.dataset = cleanup(parquet.read_table(dataset))
        self.browse_images = browse_images

    def routes(self) -> Iterable[web.RouteDef]:
        yield web.get("/data", self.send_data)
        yield web.get("/data/columns", self.send_columns)
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
                    "type": "quant" if arrow_types.is_floating(ctype) else "qual"
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
            dataset = dataset.select(columns.split(","))

        return web.Response(
            body = json.dumps(
                dataset.to_pydict(),
                separators=(',', ':')
            ),
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
