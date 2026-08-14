"""The MultiDEx GUI.

When you run this command it will print a message like this:

    ======== Running on http://127.0.0.1:8080 ========
    (Press CTRL+C to quit)

Leave that window alone, open up your web browser, and point it at
the URL in the "Running on" line.
"""

import os
import json
import logging
import argparse

from collections.abc import Awaitable
from functools import partial
from pathlib import Path
from os.path import splitext

from aiohttp import web
from pyarrow import ArrowException, Table, parquet

from multidex.assets import load_asset
from multidex.data.sexps import parse_sexps
from multidex.data.cleanup import cleanup
from multidex.utilz.exceptions import format_oserror


# aiohttp doesn't do this for us
CONTENT_TYPE_FOR_EXT = {
    "css":   "text/css",
    "html":  "text/html",
    "js":    "text/javascript",
    "md":    "text/plain",
    "txt":   "text/plain",

    "gif":   "image/gif",
    "jpeg":  "image/jpeg",
    "jpg":   "image/jpeg",
    "png":   "image/png",
    "svg":   "image/svg+xml",
    "tiff":  "image/tiff",

    "otf":   "font/otf",
    "ttc":   "font/collection",
    "ttf":   "font/ttf",
    "woff":  "font/woff",
    "woff2": "font/woff2",
}


async def respond_with_asset(
    req: web.Request,
    *,
    asset_path: str,
) -> web.StreamResponse:
    suffix = splitext(asset_path)[1].lower().removeprefix(".")
    content_type = CONTENT_TYPE_FOR_EXT.get(suffix, "application/octet-stream")

    try:
        body = load_asset(asset_path)
    except OSError as e:
        raise web.HTTPNotFound(
            text=f"{req.rel_url}: 404 Not Found: {format_oserror(e)}"
        ) from e
    except Exception as e:
        raise web.HTTPNotFound(
            text=f"{req.rel_url}: 404 Not Found: {e}"
        ) from e
    return web.Response(
        body = body,
        content_type = content_type,
    )


async def respond_with_file(
    req: web.Request,
    *,
    filename: str,
    parent_dir: Path,
) -> web.StreamResponse:
    suffix = splitext(filename)[1].lower().removeprefix(".")
    content_type = CONTENT_TYPE_FOR_EXT.get(suffix, "application/octet-stream")

    try:
        with open(parent_dir / filename, "rb") as fp:
            body = fp.read()
    except OSError as e:
        raise web.HTTPNotFound(
            text=f"{req.rel_url}: 404 Not Found: {format_oserror(e)}"
        ) from e
    except Exception as e:
        raise web.HTTPNotFound(
            text=f"{req.rel_url}: 404 Not Found: {e}"
        ) from e
    return web.Response(
        body = body,
        content_type = content_type,
    )


def get_asset(route: str, asset: str) -> web.RouteDef:
    return web.get(route, partial(respond_with_asset, asset_path=asset))


def asset_dir(req: web.Request) -> Awaitable[web.StreamResponse]:
    return respond_with_asset(
        req,
        asset_path = req.match_info["asset"]
    )


def browse_image(
    req: web.Request, *, parent_dir: Path
) -> Awaitable[web.StreamResponse]:
    return respond_with_file(
        req,
        filename = req.match_info["img"],
        parent_dir = parent_dir,
    )


async def data(req: web.Request, *, dataset: Table) -> web.StreamResponse:
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


async def cols(req: web.Request, *, dataset: Table) -> web.StreamResponse:
    return web.Response(
        body = json.dumps(dataset.column_names),
        content_type="application/json"
    )


def main() -> None:
    """MultiDEx command line entry point."""

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    ap.add_argument("dataset", type=Path,
                    help="Parquet file containing the dataset to be viewed.")
    ap.add_argument("browse_images", type=Path, nargs="?", default=None,
                    help="Directory containing browse images"
                    " associated with the dataset.  If omitted, browse"
                    " images will not be displayed.")

    ap.add_argument("--debug", action="store_true",
                    help="Print lots of debugging messages on the console.")
    ap.add_argument("-p", "--port", default="8080",
                    help="TCP port where the back end should listen for"
                    " requests (default: 8080).")
    args = ap.parse_args()

    try:
        port = int(args.port, 10)
    except ValueError:
        ap.error(f"invalid PORT {args.port!r}: not an integer")
    if not (1 <= port <= 65535):
        ap.error(f"invalid PORT {args.port!r}:"
                 f" must be >= 1 and <= 65535")

    dataset = cleanup(parquet.read_table(args.dataset))

    routes = [
        get_asset("/", "plotter.html"),
        get_asset("/favicon.ico", "logo-small.png"),
        get_asset("/robots.txt", "robots.txt"),
        web.get("/data", partial(data, dataset=dataset)),
        web.get("/data/columns", partial(cols, dataset=dataset)),
        web.get("/s/{asset:[a-zA-Z0-9./_-]+}", asset_dir),
    ]
    if args.browse_images is not None:
        routes.append(
            web.get(
                r"/browse/{img:[a-zA-Z0-9_-]+\.(?:gif|jpe?g|png|svg|tiff)}",
                partial(browse_image, parent_dir=args.browse_images)
            )
        )

    app = web.Application()
    app.add_routes(routes)
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
        # yes this is how you do this (if you don't control the call
        # to asyncio.run, which I *could*, but...)
        os.environ["PYTHONASYNCIODEBUG"] = "1"
        web.run_app(
            app,
            host = "127.0.0.1",
            port = port,
        )

    else:
        logging.basicConfig(level=logging.INFO)
        web.run_app(
            app,
            host = "127.0.0.1",
            port = port,
            access_log = None,
        )


if __name__ == "__main__":
    main()
