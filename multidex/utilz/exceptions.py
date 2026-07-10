"""Helper functions for working with exceptions."""

from typing import NoReturn


def raise_PermissionError(path: str) -> NoReturn:
    """
    Raise a PermissionError exception, matching as closely
    as possible what you would get if the C-API function
    PyErr_SetFromErrnoWithFilename had been called when
    errno == EACCES.  (This _should_ be as simple as
    "raise PermissionError(path)", but if you do that
    you get an exception object with the path in the
    _errno_ slot and it doesn't print right.)
    """
    from errno import EACCES
    from os import strerror
    raise OSError(EACCES, strerror(EACCES), path)
