"""Grab bag of utility functions."""

# written this way so mypy understands it's OK for other modules to
# import ModuleType from here
from types import ModuleType as ModuleType


def __getattr__impl(name: str, parent: str) -> ModuleType:
    """Hook for attribute lookup on modules: attempt to load any
    undefined name as a submodule.  This makes 'import foo; foo.bar'
    work whenever 'import foo.bar' would have.

    Usage: in each __init__.py write

    from multidex.utilz import ModuleType, __getattr__impl
    def __getattr__(name: str) -> ModuleType:
        return __getattr__impl(name, __name__)

    You have to define __getattr__ separately in each __init__.py
    so that it passes the correct value of __name__ to __getattr__impl.
    """
    if '.' in name:
        raise AttributeError(f"module '{parent}' has no attribute '{name}'")

    import importlib
    import sys

    try:
        mod = importlib.import_module("." + name, parent)
        setattr(sys.modules[parent], name, mod)
        return mod
    except ImportError as e:
        raise AttributeError(
            f"module '{parent}' has no attribute '{name}'"
        ) from e


def __getattr__(name: str) -> ModuleType:
    return __getattr__impl(name, __name__)
