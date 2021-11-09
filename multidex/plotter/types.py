"""custom type aliases"""

from typing import Any
from typing_extensions import TypeAlias

# PyTypeChecker can't introspect most things in Django for reasons I do not
# care to understand. these are silly aliases to visually hint functions that
# accept / return Model classes and class instances.

# e.g., ZSpec, MSpec, ...

SpectrumModel: TypeAlias = Any
SpectrumModelInstance: TypeAlias = Any
