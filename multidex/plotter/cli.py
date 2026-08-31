"""The MultiDEx GUI."""

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiohttp import web

async def after_startup_hook(
    app: "web.Application",
    *,
    port: int,
    open_browser: bool,
) -> None:
    app_url = f"http://127.0.0.1:{port}/"
    if not open_browser:
        message = (
            f"=== MultiDEx back end now running. ===\n"
            f"\n"
            f"Open the GUI by visiting this URL in your web browser:\n"
            f"    <{app_url}>.\n"
        )
    else:
        import webbrowser
        if webbrowser.open_new(app_url):
            message = (
                f"=== MultiDEx GUI now open in your web browser. ===\n"
                f"\n"
                f"If you accidentally close it, you can get it back by\n"
                f"visiting this URL: <{app_url}>.\n"
            )
        else:
            message = (
                f"!!! Couldn't open the MultiDEx GUI in your web browser !!!\n"
                f"\n"
                f"The back end is running. You will have to open the GUI\n"
                f"yourself: visit this URL: <{app_url}>.\n"
            )

    message += (
        "\n"
        "To quit MultiDEx, close any browser tab(s) displaying the GUI,\n"
        "and then type control-C into this window.\n"
    )

    import sys
    sys.stdout.write(message)


def run_app(
    *,
    dataset: Path,
    browse_images: Path | None = None,
    port: int = 8080,
    debug: bool = False,
    open_browser: bool = True,
):
    import logging
    from functools import partial
    from aiohttp import web
    from multidex.plotter.backend import Backend
    from multidex.utilz.http import get_asset, get_any_asset

    backend = Backend(dataset, browse_images)
    routes = [
        get_asset("/", "plotter.html"),
        get_asset("/favicon.ico", "logo-small.png"),
        get_asset("/robots.txt", "robots.txt"),
        get_any_asset("/s"),
    ]
    routes.extend(backend.routes())

    app = web.Application()
    app.add_routes(routes)
    app.on_startup.append(partial(
        after_startup_hook,
        port = port,
        open_browser = open_browser,
    ))

    if debug:
        logging.basicConfig(level=logging.DEBUG)
        # yes this is how you do this (if you don't control the call
        # to asyncio.run, which I *could*, but...)
        from os import environ
        environ["PYTHONASYNCIODEBUG"] = "1"
        web.run_app(
            app,
            host = "127.0.0.1",
            port = port,
            print = None,
        )

    else:
        logging.basicConfig(level=logging.INFO)
        web.run_app(
            app,
            host = "127.0.0.1",
            port = port,
            access_log = None,
            print = None,
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

    ap.add_argument("--no-open-browser",
                    action="store_false", dest="open_browser",
                    help="Don't automatically open a browser window"
                    " to display the MultiDEx GUI.")
    ap.add_argument("--open-browser",
                    action="store_const", const=True, dest="open_browser",
                    help=argparse.SUPPRESS)

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

    run_app(
        dataset=args.dataset,
        browse_images=args.browse_images,
        port=port,
        debug=args.debug,
        open_browser=args.open_browser,
    )


if __name__ == "__main__":
    main()
