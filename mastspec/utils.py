"""assorted utility functions for project"""

def rows(dataframe):
    """splits row-wise into a list of numpy arrays"""
    return [dataframe.loc[row] for row in dataframe.index]


def columns(dataframe):
    """splits column-wise into a list of numpy arrays"""
    return [dataframe.loc[:, column] for column in dataframe.columns]


def eta_methods(target, *input_args, kwarg_list=None, method=None):
    """
    object, optional list, optional string or function
    -> function(string or function, *args)
    create eta abstractions with arbitrary argument reordering on methods
    of target. returns a function whose first positional argument is a
    method of target or a string corresponding to a method of target, and
    whose subsequent positional arguments are mapped to the keyword
    arguments of that method in the ordering defined in kwarg_list. may
    optionally also produce a partially evaluated version with a defined
    method. permissive: will allow you to produce abstractions of individual
    functions, basically as a wordy alias for eta.
    """

    def do_method(requested_method, *args, **kwargs):
        if isinstance(requested_method, str):
            requested_method = getattr(target, requested_method)
        return passargs(*args, *kwargs)(
            eta(requested_method, *input_args, kwarg_list)
        )

    if method is None:
        return do_method
    return partial(do_method, requested_method=method)


def keygrab(dict_list, key, value):
    """returns first element of dict_list such that element[key]==value"""
    return next(filter(lambda x: x[key] == value, dict_list))