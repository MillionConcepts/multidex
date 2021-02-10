"""assorted utility functions for project"""

import json
import re
from functools import partial, reduce
from inspect import currentframe, getframeinfo, signature
from operator import and_, or_, contains

import dash
import dash_html_components as html
from dash.dependencies import Input, Output
from django.db.models import Q
from toolz import keyfilter, merge, isiterable


# generic

def first(predicate, iterable):
    for item in iterable:
        if predicate(item):
            return item


def not_blank(obj):
    """a 'truthiness' test that avoids 0/1 boolean evaluations"""
    if reduce(and_, [
        obj != "",
        obj is not None,
        obj != [],
        obj != {}
    ]):
        return True
    return False


# django utility functions


def qlist(queryset, attribute):
    return list(queryset.values_list(attribute, flat=True))


def filter_null_attributes(queryset, attribute_list):
    for attribute in attribute_list:
        queryset = queryset.exclude(**{attribute + '__iexact': None})
    return queryset


def djget(model, value, field="name", method_name="filter",
          querytype="iexact"):
    """flexible interface to queryset methods"""
    # get the requested queryset-generating method of model.objects
    method = getattr(model.objects, method_name)
    # and then evaluate it on the requested parameters
    return method(**{field + "__" + querytype: value})


def modeldict(django_model_object):
    """tries to construct a dictionary from arbitrary django model instance"""
    return {
        field.name: getattr(django_model_object, field.name)
        for field in django_model_object._meta.get_fields()
    }


# pandas utility functions


def rows(dataframe):
    """splits row-wise into a list of numpy arrays"""
    return [dataframe.loc[row] for row in dataframe.index]


def columns(dataframe):
    """splits column-wise into a list of numpy arrays"""
    return [dataframe.loc[:, column] for column in dataframe.columns]


# dash-y dictionary utilities


def dict_to_paragraphs(dictionary, style):
    """
    parses dictionary to list of dash <p> components
    """
    return [
        html.P(
            str(key) + " " + str(value), style={"margin": 0, "fontSize": 14}
        )
        for key, value in dictionary.items()
    ]


def pickitems(dictionary, some_list):
    """items of dict where key is in some_list """
    return keyfilter(in_me(some_list), dictionary)


# TODO: What am I actually doing here? not what I say I am, for sure.
def pickcomps(comp_dictionary, id_list):
    """items of dictionary of dash components where id is in id_list"""
    return pickitems(comp_dictionary,
                     [comp.component_id for comp in comp_dictionary])


def comps_to_strings(component_list):
    """convert list of dash components with properties to list of strings"""
    return [
        comp.component_id + "." + comp.component_property for comp in
        component_list
    ]


def pickctx(context, component_list):
    """states and inputs of dash callback context if component is in
    component_list"""
    comp_strings = comps_to_strings(component_list)
    cats = []
    if context.states:
        cats.append(context.states)
    if context.inputs:
        cats.append(context.inputs)
    picked = [pickitems(cat, comp_strings) for cat in cats]
    if picked:
        return merge(picked)


def keygrab(dict_list, key, value):
    """returns first element of dict_list such that element[key]==value"""
    return next(filter(lambda x: x[key] == value, dict_list))


def ctxdict(ctx):
    return {
        'triggered': ctx.triggered,
        'inputs': ctx.inputs,
        'states': ctx.states,
        'response': ctx.response
    }


def not_triggered():
    """
    detect likely spurious triggers.
    will only function if called inside a callback.
    """
    if not dash.callback_context.triggered[0]['value']:
        return True
    return False


def triggered_by(component_id):
    """
    did a component matching this id string trigger the callback this function 
    is called in the context of? will only function if called inside a
    callback.
    """
    if component_id in dash.callback_context.triggered[0]['prop_id']:
        return True
    return False


def trigger_index(ctx):
    """
    dash.callbackcontext -> int, where int is the index of the triggering
    component
    assumes there is exactly one triggering component and it has an index!
    """
    trigger_id = ctx.triggered[0]['prop_id']
    index_index = re.search("index", trigger_id).span()[1] + 2
    return int(trigger_id[index_index])


## generic dev tools

# doesn't go one floor up...evaluates default arguments on import, not at
# runtime. inconvenient to enter all this, defeats purpose. hm.

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
    def dprint(*args, local_variables=locals(),
               frame=getframeinfo(currentframe())):
        print(local_variables)
        info = getattr(frame, 'function'), getattr(frame, 'lineno'), *(
            (arg, local_variables[arg]) for arg in args
        )
        print(info)
        return info

    return dprint


# ## dash dev tools


def dump_it(data, loud=True):
    """dump data as json"""
    dump = json.dumps(data, indent=2)
    if loud:
        print(dump)
    return dump


def make_printer(element, prop, app, print_target="print",
                 process_function=dump_it):
    """
    utility callback factory. impure! inserts the callback into the tree
    when called.
    when called, creates a callback to print property of element in app to
    print_target
    """

    def print_callback():
        app.callback(Output(print_target, "children"), [Input(element, prop)])(
            process_function)

    print_callback()


### lambda replacements


def in_me(container):
    """returns function that checks if all its arguments are in container"""
    inclusion = partial(contains, container)

    def is_in(*args):
        return reduce(and_, map(inclusion, args))

    return is_in


# ## generic


def get_if(boolean, dictionary, key):
    """return dictionary[key] iff boolean; otherwise return None"""
    if boolean:
        return dictionary.get(key)
    return None


def get_parameters(function):
    return [
        param.name for param in signature(function).parameters.values()
    ]


def partially_evaluate_from_parameters(function, parameters):
    """
    function, dict -> function
    return a copy of the input function partially evaluated from any value
    of the dict
    with a key matching a named argument of the function. useful for things
    like inserting
    'settings' variables into large numbers of functions.
    for instance:
    def add(a, b):
        return a + b
    parameters = {'b':1, 'c':3}
    add_1 = partially_evaluate_from_parameters(add, parameters)
    add_1(2) == 3
    True
    """
    return partial(
        function, **pickitems(parameters, get_parameters(function))
    )


def listify(thing):
    """Always a list, for things that want lists"""
    if isiterable(thing):
        return list(thing)
    return [thing]


def none_to_empty(thing):
    if thing is None:
        return ""
    return thing


### search functions


# some of the following functions hit the database multiple times during
# evaluation.
# this is necessary to allow flexibly loose and strict phrasal searches.
# it potentially reduces efficiency a great deal and is terrain for
# optimization
# if and when required.


def flexible_query(queryset, field, value):
    """
    little search function that checks exact and loose phrases.
    have to hit the database to do this, so less efficient than using
    """
    # allow exact phrase searches
    query = field + "__iexact"
    if queryset.filter(**{query: value}):
        return queryset.filter(**{query: value})
    # otherwise treat multiple words as an 'or' search",
    query = field + "__icontains"
    filters = [queryset.filter(**{query: word}) for word in value.split(" ")]
    return reduce(or_, filters)


def inflexible_query(queryset, field, value):
    """little search function that checks only exact phrases"""
    query = field + "__iexact"
    return queryset.filter(**{query: value})


def term_search(queryset, field, value, inflexible=None):
    """
    model, string, string or number or whatever -> queryset
    search for strings or whatever within a field of a model.
    """
    # allow inexact phrase matching?
    search_function = flexible_query
    if inflexible:
        search_function = inflexible_query
    return search_function(queryset, field, value)


def value_fetch_search(
        queryset, field, value_list
):
    """
    queryset, field of underlying model, list of values -> queryset
    simply return queryset of objects with any of those values
    in that field. strict, for the moment.
    """
    # note that the case-insensitive 'iexact' match method of django
    # queryset objects
    # is in fact _more_ sensitive wrt type; it will not match a float
    # representation of an
    # int 
    queries = [Q(**{field + "__exact": item}) for item in value_list]
    return queryset.filter(reduce(or_, queries))


def interval_search(
        queryset, field, interval_begin=None, interval_end=None, strictly=None
):
    """
    queryset, field of underlying model, begin, end -> queryset
    
    interval_begin and interval_end must be of types for which 
    the elements of the set of the chosen attribute of the elements of queryset
    possess a complete ordering exposed to django queryset API (probably
    defined in terms of
    standard python comparison operators). in most cases, you 
    will probably want these to be the same type, and to share a type with
    the attributes of the objects in question. 
    
    if both interval_begin and interval_end are defined, returns 
    all entries > begin and < end. 
    if only interval_begin is defined: all entries > begin. 
    if only interval_end is defined: all entries < end. 
    if neither: trivially returns the queryset.
    
    the idea of this function is to attempt to perform searches with somewhat 
    convoluted Python but a single SQL query. it could be _further_
    generalized to
    tuples of attributes that bound a convex space, most simply interval
    beginnings
    and endings of their own (start and stop times, for instance)
    """
    if strictly:
        less_than_operator = "__lt"
        greater_than_operator = "__gt"
    else:
        less_than_operator = "__lte"
        greater_than_operator = "__gte"

    # these variables are queries defined by the ordering on this attribute.
    greater_than_begin = Q(**{field + greater_than_operator: interval_begin})
    less_than_end = Q(**{field + less_than_operator: interval_end})

    # select only entries with attribute greater than interval_begin (if
    # defined)
    # and less than interval_end (if defined)
    queries = [Q()]
    if interval_begin:
        queries.append(greater_than_begin)
    if interval_end:
        queries.append(less_than_end)
    return queryset.filter(reduce(and_, queries))


def multiple_field_search(queryset, parameters):
    """
    dispatcher that handles multiple search parameters and returns a queryset.
    accepts options in dictionaries to search 'numerical' intervals
    or stringlike terms.
    """
    results = []
    print('mfs', parameters)
    for parameter in parameters:
        # do a relations-on-orderings search if requested
        if parameter.get("value_type") == "quant":
            if "value_list" in parameter.keys():
                search_result = value_fetch_search(
                    queryset,
                    parameter["field"],
                    parameter["value_list"]
                )
            else:
                search_result = interval_search(
                    queryset,
                    parameter["field"],
                    # begin and end and strictly are optional
                    parameter.get("begin"),
                    parameter.get("end"),
                    parameter.get("strictly"),
                )
            results.append(search_result)
        # otherwise just look for term matches
        else:
            for term in parameter["term"]:
                search_result = term_search(
                    queryset, parameter["field"], term,
                    parameter.get("flexible")
                )
                results.append(search_result)
    return reduce(and_, results)
