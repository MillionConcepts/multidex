"""assorted utility functions for project"""

from functools import partial, reduce, wraps
from operator import and_, contains


def rows(dataframe):
    """splits row-wise into a list of numpy arrays"""
    return [dataframe.loc[row] for row in dataframe.index]


def columns(dataframe):
    """splits column-wise into a list of numpy arrays"""
    return [dataframe.loc[:, column] for column in dataframe.columns]


def pickitems(dictionary, some_list):
    """items of dict where key is in some_list """
    keyfilter(in_me(some_list),(dictionary)).values()

def pickcomps(comp_dictionary, id_list):
    """items of dictionary of dash components where id is in id_list"""
    return pickitems(comp_dictionary, [comp.component_id for comp in comp_dictionary])


def passargs(*args, **kwargs):
    """
    returns a function f with f(g) = g(args).
    e.g. f = passargs(1, 10)
    f(range) = range(1, 10)
    f(operator.ge) = False
    formally, an eta abstraction for arbitrary g.
    """
    return lambda f: f(*args, **kwargs)


def eta(input_function, *args, kwarg_list=()):
    """
    create an eta abstraction g of input function with arbitrary argument
    ordering. positional arguments to g _after_ the arguments defined in
    kwarg_list are mapped to positional arguments of input_function; all
    keyword arguments to g are mapped to keyword arguments of input_function.
    positional arguments to g matching keywords in kwarg_list override keyword
    arguments to g.

    can be used to make short forms of functions. also useful along with partial
    to create partially applied versions of functions free from argument
    collision.

    passing eta a function with no further arguments simply produces an alias.
    """
    if not (args or kwarg_list):
        # with no arguments, just alias input_function. the remainder of the
        # function accomplishes basically this, but just passing the function
        # is cheaper
        return input_function
    kwarg_list = args + tuple(kwarg_list)

    @wraps(input_function)
    def do_it(*args, **kwargs):
        output_kwargs = {}
        positionals = []
        # are there more args than the eta-defined argument list? pass them to
        # input_function.
        if len(args) > len(kwarg_list):
            positionals = args[len(kwarg_list) :]
        # do we have an argument list? then zip it with args up to its
        # length.
        if kwarg_list:
            output_kwargs = dict(
                zip(kwarg_list[: len(kwarg_list)], args[: len(kwarg_list)])
            )
        # do we have kwargs? merge them with the keyword dictionary generated
        # from do_it's args.
        if kwargs:
            output_kwargs = merge(kwargs, output_kwargs)
        if not output_kwargs:
            return input_function(*positionals)
        return input_function(*positionals, **output_kwargs)

    return do_it



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
        if kwarg_list is None:
            etad = eta(requested_method, *input_args)
        else:
            etad = eta(requested_method, *input_args, kwarg_list)       
        return passargs(*args, **kwargs)(etad)
    if method is None:
        return do_method
    return partial(do_method, requested_method=method)


def keygrab(dict_list, key, value):
    """returns first element of dict_list such that element[key]==value"""
    return next(filter(lambda x: x[key] == value, dict_list))


def in_me(container):
    """returns function that checks if all its arguments are in container"""
    inclusion = partial(contains, container)

    def is_in(*args):
        return reduce(and_, map(inclusion, args))

    return is_in