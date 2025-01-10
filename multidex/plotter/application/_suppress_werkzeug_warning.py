"""
Import this module to suppress werkzeug's warnings about using a
development server. This is useful in contexts like this that are intended to
run only locally.
"""
import functools

import werkzeug.serving

def _werkzeug_warning_filter(stylefunc):
    @functools.wraps(stylefunc)
    def filtered_style(*args, **kwargs):
        if (
            len(args) > 0
            and isinstance(args[0], str)
            and args[0].startswith('WARNING: This is a development')
        ):
            return ''
        return stylefunc(*args, **kwargs)

    return filtered_style


setattr(
    werkzeug.serving,
    "_ansi_style",
    _werkzeug_warning_filter(werkzeug.serving._ansi_style)
)
