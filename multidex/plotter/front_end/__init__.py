"""
This subpackage contains resource files to be served verbatim.
Access them using front_end.load_resource().
"""

import importlib.resources
from pathlib import PurePosixPath

from multidex.utilz.exceptions import raise_PermissionError


__all__ = [
    "load_resource",
    "FORBIDDEN_RESOURCE_SUFFIXES",
]


#: load_resource refuses to load any file whose name ends with one of
#: these suffixes.
FORBIDDEN_RESOURCE_SUFFIXES = [
    # Python code; the presence of .pyc is a belt-and-suspenders
    # protective measure since we also refuse to look inside __pycache__.
    # For the same reason, we refuse to load .pyo files, even though
    # as of Python 3.5 they are no longer generated (see PEP 488).
    ".py", ".pyc", ".pyo",
    # Editor backup files, merge conflicts, etc.
    "~", "#", ".bak", ".old", ".orig", ".rej",
]


def load_resource(rel_url: str) -> bytes:
    """
    Read the resource file named by 'rel_url', which is understood as a
    pathname relative to this subpackage, and processed according to
    the rules for URL-paths; in particular, a/b/../c is equivalent to
    a/c, regardless of whether b exists.

    It is an error if `rel_url` names a directory, if it refers to any
    file inside a __pycache__/ directory, if it refers to a file whose
    name begins with a dot (i.e. Unix hidden files), or if it refers
    to a file whose name ends with one of the suffixes listed in
    FORBIDDEN_RESOURCE_SUFFIXES.
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
            continue
        segments.append(seg)

    resource = importlib.resources.files(__package__)
    for seg in segments:
        if seg == "__pycache__":
            raise_PermissionError(f"{str(resource)}/{seg}")
        resource = resource / seg

    fname = resource.name
    if (
        not fname
        or fname[0] == "."
        or any(fname.endswith(suffix) for suffix in FORBIDDEN_RESOURCE_SUFFIXES)
    ):
        raise_PermissionError(str(resource))

    return resource.read_bytes()
