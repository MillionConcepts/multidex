"""
Utility functions and tables for serving HTTP.
"""

from collections.abc import Awaitable
from functools import partial
from os.path import splitext
from pathlib import Path

from aiohttp import web

from multidex.assets import load_asset
from multidex.utilz.exceptions import format_oserror


# aiohttp doesn't provide this information
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


def content_type_for_path(path: str) -> str:
    """
    Given a pathname PATH, return an appropriate content type for
    whatever's at that path, based on its file name suffix.
    """
    return CONTENT_TYPE_FOR_EXT.get(
        splitext(path)[1].lower().removeprefix("."),
        "application/octet-stream"
    )


async def respond_with_asset(
    req: web.Request,
    *,
    asset_path: str,
) -> web.StreamResponse:
    content_type = content_type_for_path(asset_path)

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
    """
    Respond to client request REQ by producing the contents of file
    {filename}, which should be located in {parent_dir}.
    """
    content_type = content_type_for_path(filename)

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


def get_asset(path: str, asset: str) -> web.RouteDef:
    """
    Route definition: 'GET /{path}' should produce the asset whose
    name is {asset}.
    """
    return web.get(path, partial(respond_with_asset, asset_path=asset))


def get_any_asset(prefix: str) -> web.RouteDef:
    """
    Route definition: 'GET /{prefix}/<asset_name>' should produce
    the asset whose name is <asset_name>, if it exists.
    """
    def respond_with_asset_from_dir(
        req: web.Request
    ) -> Awaitable[web.StreamResponse]:
        return respond_with_asset(
            req,
            asset_path = req.match_info["asset"]
        )

    return web.get(prefix + "/{asset:[a-zA-Z0-9./_-]+}",
                   respond_with_asset_from_dir)


def bool_query_arg(arg: str) -> bool:
    """
    Parse ARG as a boolean query argument.  Accepts all the things
    YAML 1.1 thinks are booleans, because we're casual like that.
    """
    larg = arg.lower()
    if larg in ("1", "t", "true", "y", "yes", "on"):
        return True
    if larg in ("0", "f", "false", "n", "no", "off"):
        return False
    raise ValueError(f"{arg} not recognized as a boolean value")
