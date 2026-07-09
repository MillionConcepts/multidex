"""
This subpackage contains resource files to be served verbatim.
Access them using assets.load_asset().
"""

import importlib.resources

from pathlib import PurePosixPath

def load_asset(rel_url: str) -> bytes:
    """
    Read the asset named by 'rel_url'.
    """
    # rel_url is a relative *URL* and we specifically want to convert
    # x/y/.././z to x/z using the rules for doing that to URLs, *not*
    # the rules for doing that to a filesystem path.  PurePosixPath
    # will do only part of this job.
    segments: list[str] = []
    for seg in PurePosixPath(rel_url).parts:
        if seg in ('/', '//'):
            continue # this can only happen at the beginning
        if seg == "..":
            try:
                segments.pop()
            except IndexError:
                pass
        else:
            segments.append(seg)

    asset = importlib.resources.files(__package__)
    for seg in segments:
        asset = asset / seg
    return asset.read_bytes()
