"""assorted utility functions for project"""

from functools import partial, reduce, wraps
import json
from operator import and_, or_, contains

from dash.dependencies import Input, Output
from toolz import identity, keyfilter, merge, isiterable, get_in


# django utility functions


def qlist(queryset, attribute):
    return list(queryset.values_list(attribute, flat=True))


def djget(model, value, field="name", method_name="filter", querytype="iexact"):
    """flexible interface to queryset methods"""
    # get the requested queryset-generating method of model.objects
    method = getattr(model.objects, method_name)
    # and then evaluate it on the requested parameters
    return method(**{field + "__" + querytype: value})


def modeldict(django_model_object, exclude_fields = None):
    """tries to construct a dictionary from arbitrary django model instance"""
    if exclude_fields == None:
        exclude_fields = []
    return keyfilter(lambda x: x not in exclude_fields, 
        {
        field.name: getattr(django_model_object, field.name)
        for field in django_model_object._meta.get_fields()
        }
    )


# pandas utility functions


def rows(dataframe):
    """splits row-wise into a list of numpy arrays"""
    return [dataframe.loc[row] for row in dataframe.index]


def columns(dataframe):
    """splits column-wise into a list of numpy arrays"""
    return [dataframe.loc[:, column] for column in dataframe.columns]


# dash-y dictionary utilities


def pickitems(dictionary, some_list):
    """items of dict where key is in some_list """
    return keyfilter(in_me(some_list), (dictionary))


def pickcomps(comp_dictionary, id_list):
    """items of dictionary of dash components where id is in id_list"""
    return pickitems(comp_dictionary, [comp.component_id for comp in comp_dictionary])


def comps_to_strings(component_list):
    """convert list of dash components with properties to list of strings"""
    return [
        comp.component_id + "." + comp.component_property for comp in component_list
    ]


def pickctx(context, component_list):
    """states and inputs of dash callback context if component is in component_list"""
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


### dash dev tools


def dump_it(data):
    """dump data as json"""
    return json.dumps(data, indent=2)


def make_printer(element, prop, app, print_target="print", process_function=dump_it):
    """
    utility callback factory. impure! inserts the callback into the tree when called. 
    when called, creates a callback to print property of element in app to print_target
    """

    def print_callback():
        app.callback(Output(print_target, "children"), [Input(element, prop)])(process_function)

    print_callback()




### lambda replacements


def in_me(container):
    """returns function that checks if all its arguments are in container"""
    inclusion = partial(contains, container)

    def is_in(*args):
        return reduce(and_, map(inclusion, args))

    return is_in


### search functions


# some of the following functions hit the database multiple times during evaluation.
# this is necessary to allow flexibly loose and strict phrasal searches.
# it potentially reduces efficiency a great deal and is terrain for optimization
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


def interval_search(
    queryset, field, interval_begin=None, interval_end=None, strictly=None
):
    """
    queryset, field of underlying model, begin, end -> queryset
    
    interval_begin and interval_end must be of types for which 
    the elements of the set of the chosen attribute of the elements of queryset
    possess a complete ordering exposed to django queryset API (probably defined in terms of 
    standard python comparison operators). in most cases, you 
    will probably want these to be the same type, and to share a type with
    the attributes of the objects in question. 
    
    if both interval_begin and interval_end are defined, returns 
    all entries > begin and < end. 
    if only interval_begin is defined: all entries > begin. 
    if only interval_end is defined: all entries < end. 
    if neither: trivially returns the queryset.
    
    the idea of this function is to attempt to perform searches with somewhat 
    convoluted Python but a single SQL query. it could be _further_ generalized to 
    tuples of attributes that bound a convex space, most simply interval beginnings 
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

    # select only entries with attribute greater than interval_begin (if defined)
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

    for parameter in parameters:
        # do a relations-on-orderings search if requested
        if parameter.get("value_type") == "quant":
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
                    queryset, parameter["field"], term, parameter.get("flexible")
                )
                results.append(search_result)
    return reduce(and_, results)
