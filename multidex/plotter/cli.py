"""The MultiDEx GUI.

When you run this command it will print a message like this:

    ======== Running on http://127.0.0.1:8080 ========
    (Press CTRL+C to quit)

Leave that window alone, open up your web browser, and point it at
the URL in the "Running on" line.
"""

import logging
import argparse

from pathlib import Path

from aiohttp import web

from multidex.plotter.backend import Backend
from multidex.utilz.http import get_asset, get_any_asset


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

    backend = Backend(args.dataset, args.browse_images)

    routes = [
        get_asset("/", "plotter.html"),
        get_asset("/favicon.ico", "logo-small.png"),
        get_asset("/robots.txt", "robots.txt"),
        get_any_asset("/s"),
    ]
    routes.extend(backend.routes())

    app = web.Application()
    app.add_routes(routes)
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
        # yes this is how you do this (if you don't control the call
        # to asyncio.run, which I *could*, but...)
        from os import environ
        environ["PYTHONASYNCIODEBUG"] = "1"
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
