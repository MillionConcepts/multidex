"""
bad dev tools

don't go one floor up...evaluates default arguments on import, not at
runtime. inconvenient to enter all this, defeats purpose.
bad, don't use
"""
from inspect import getframeinfo, currentframe


def dprint(*args, local_variables=locals(),
           frame=getframeinfo(currentframe())):
    # print name of current function, line number, string, and value of
    # variable with name
    # corresponding to string for all strings in args
    print(local_variables)
    info = (
        getattr(frame, 'function'),
        getattr(frame, 'lineno'),
        *((arg, local_variables[arg]) for arg in args)
    )
    print(info)
    return info


def dprinter():
    # print name of current function, line number, string, and value of
    # variable with name
    # corresponding to string for all strings in args
    def dprint2(*args, local_variables=locals(),
                frame=getframeinfo(currentframe())):
        print(local_variables)
        info = getattr(frame, 'function'), getattr(frame, 'lineno'), *(
            (arg, local_variables[arg]) for arg in args
        )
        print(info)
        return info

    return dprint2
