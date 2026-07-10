"""The MultiDEx GUI.

When you run this command it will print a message like this:

    ======== Running on http://127.0.0.1:8080 ========
    (Press CTRL+C to quit)

Leave that window alone, open up your web browser, and point it at
the URL in the "Running on" line.
"""

import os
import logging
import argparse

from collections.abc import Awaitable
from functools import partial
from os.path import splitext

from aiohttp import web

from multidex.assets import load_asset


async def respond_with_asset(
    req: web.Request,
    *,
    asset_path: str,
) -> web.StreamResponse:
    suffix = splitext(asset_path)[1].lower().removeprefix(".")
    content_type = ({
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
    }).get(suffix, "application/octet-stream")

    try:
        body = load_asset(asset_path)
    except Exception as e:
        raise web.HTTPNotFound(text=f"{req.rel_url}: {e}") from e
    return web.Response(
        body = body,
        content_type = content_type,
    )


def get_asset(route: str, asset: str) -> web.RouteDef:
    return web.get(route, partial(respond_with_asset, asset_path=asset))


def asset_dir(req: web.Request) -> Awaitable[web.StreamResponse]:
    asset_url = req.match_info["asset"]
    return respond_with_asset(req, asset_path=asset_url)


def main() -> None:
    """MultiDEx command line entry point."""

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
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

    app = web.Application()
    app.add_routes([
        get_asset("/", "plotter.html"),
        get_asset("/favicon.ico", "logo-small.png"),
        get_asset("/robots.txt", "robots.txt"),
        web.get("/s/{asset:[a-zA-Z0-9./_-]+}", asset_dir),
    ])

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
