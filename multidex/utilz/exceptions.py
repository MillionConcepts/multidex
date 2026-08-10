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


def format_oserror(e: OSError) -> str:
    """
    Stringify an OSError the way I think it should be stringified;
    in particular, does *not* print the [Errno nnn] annotation that
    OSError.__str__ prints.  (I've been burned a few too many times
    by people telling me _only_ the number, which, fun fact, is OS-
    and CPU-specific!)
    """
    if e.strerror is not None:
        msg = e.strerror
    elif e.errno is not None:
        from os import strerror
        msg = os.strerror(e.errno)
    else:
        msg = type(e).__name__

    if e.filename is not None and e.filename2 is not None:
        msg = f"{msg}: {e.filename!r} -> {e.filename2!r}"
    elif e.filename is not None:
        msg = f"{msg}: {e.filename!r}"
    elif e.filename2 is not None:
        msg = f"{msg}: ?? -> {e.filename2!r}"
    else:
        pass

    return msg
