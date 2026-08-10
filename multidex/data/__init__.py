"""
Modules having to do with back-end data processing
"""

from multidex.utilz import ModuleType, __getattr__impl

def __getattr__(name: str) -> ModuleType:
    return __getattr__impl(name, __name__)
