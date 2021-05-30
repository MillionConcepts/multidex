"""assorted utility functions for project"""

import datetime as dt
import json
import re
from functools import partial, reduce
from inspect import signature
from operator import and_, gt, ge, lt, le, or_, contains
from typing import (
    Callable,
    Iterable,
    Any,
    TYPE_CHECKING,
    Mapping,
    Union,
    Optional,
)

from cytoolz import keyfilter
import dash
import dash_html_components as html
import numpy as np
import pandas as pd
from dash.dependencies import Input, Output
from django.db.models import Q
from toolz import merge, isiterable

if TYPE_CHECKING:
    from dash.development.base_component import Component
    from django.db.models.query import QuerySet
    from django.db.models import Model


DEFAULT_CSS_PATH = "assets/css/main.css"

# generic


def re_get(mapping, pattern):
    for key in mapping.keys():
        if re.search(pattern, key):
            return mapping[key]
    return None


def seconds_since_beginning_of_day(time: dt.time) -> float:
    return (
        dt.datetime.combine(dt.date(1, 1, 1), time)
        - dt.datetime(1, 1, 1, 0, 0, 0)
    ).total_seconds()


def first(predicate: Callable, iterable: Iterable) -> Any:
    for item in iterable:
        if predicate(item):
            return item


def not_blank(obj: Any) -> bool:
    """
    a 'truthiness' test that avoids 0/1 boolean evaluations
    """
    if reduce(and_, [obj != "", obj is not None, obj != [], obj != {}]):
        return True
    return False


def none_to_quote_unquote_none(
    list_containing_none: Iterable[Any],
) -> list[Any]:
    de_noned_list = []
    for element in list_containing_none:
        if element is not None:
            de_noned_list.append(element)
        else:
            de_noned_list.append("None")
    return de_noned_list


def arbitrarily_hash_strings(strings: Iterable[str]) -> tuple[dict, list[int]]:
    unique_string_values = list(set(strings))
    arbitrary_hash = {
        string: ix for ix, string in enumerate(unique_string_values)
    }
    return arbitrary_hash, [arbitrary_hash[string] for string in strings]


# django utility functions


def qlist(queryset: "QuerySet", attribute: str) -> list:
    """note: does not preserve order!"""
    return list(queryset.values_list(attribute, flat=True))


def filter_null_attributes(
    queryset: "QuerySet", attribute_list: Iterable[str], check_related=None
) -> "QuerySet":
    if check_related is not None:
        attribute_list = [
            check_related + "__" + attribute for attribute in attribute_list
        ]
    for attribute in attribute_list:
        queryset = queryset.exclude(**{attribute + "__iexact": None})
    return queryset


def field_values(metadata_df, field=None):
    special_options = [
        {"label": "any", "value": "any"},
        # too annoying to detect all 'blank' things atm
        # {'label':'no assigned value','value':''}
    ]
    if (not field) or (field not in metadata_df.columns):
        return special_options
    unique_elements = metadata_df[field].unique()
    options = none_to_quote_unquote_none(list(unique_elements))
    options.sort()
    formatted_options = [
        {"label": option, "value": option}
        for option in options
        if option not in ["", "nan"]
    ]
    return special_options + formatted_options


def djget(
    model: "Model",
    value: Any,
    field: str = "name",
    method_name: str = "filter",
    querytype: str = "iexact",
) -> Union["QuerySet", "Model"]:
    """flexible interface to queryset methods"""
    # get the requested method of model.objects
    method = getattr(model.objects, method_name)
    # and then evaluate it on the requested parameters
    # most of these, like 'filter', return other querysets;
    # some, like 'get', return individual model instances
    return method(**{field + "__" + querytype: value})


def field_names(django_model_object: "Model") -> list[str]:
    """tries to construct a dictionary from arbitrary django model instance"""
    return [field.name for field in django_model_object._meta.get_fields()]


def modeldict(django_model_object: "Model") -> dict:
    """tries to construct a dictionary from arbitrary django model instance"""
    return {
        field.name: getattr(django_model_object, field.name)
        for field in django_model_object._meta.get_fields()
    }


# pandas utility functions

# TODO: now that it's available, DataFrame.iterrows() should be used instead
def rows(dataframe: pd.DataFrame) -> list[pd.Series]:
    """splits row-wise into a list of series"""
    return [dataframe.loc[row] for row in dataframe.index]


def columns(dataframe: pd.DataFrame) -> list[np.ndarray]:
    """splits column-wise into a list of numpy arrays"""
    return [dataframe.loc[:, column] for column in dataframe.columns]


# dash-y dictionary utilities


def dict_to_paragraphs(dictionary, style=None, ordering=None):
    """
    parses dictionary to list of dash <p> components
    """
    if style is None:
        style = {"margin": 0, "fontSize": 14}

    def make_paragraph(key, value):
        return html.P(str(key) + " " + str(value), style=style)

    if ordering is None:
        ordering = []
    ordered_grafs = [
        make_paragraph(key, dictionary.get(key))
        for key in ordering
        if key in dictionary.keys()
    ]
    unordered_grafs = [
        make_paragraph(key, value)
        for key, value in dictionary.items()
        if key not in ordering
    ]
    return ordered_grafs + unordered_grafs


def pickitems(dictionary: Mapping, some_list: Iterable) -> dict:
    """items of dict where key is in some_list"""
    return keyfilter(in_me(some_list), dictionary)


# TODO: What am I actually doing here? not what I say I am, for sure.
def pickcomps(comp_dictionary, id_list):
    """items of dictionary of dash components where id is in id_list"""
    return pickitems(
        comp_dictionary, [comp.component_id for comp in comp_dictionary]
    )


def comps_to_strings(component_list: Iterable["Component"]) -> list[str]:
    """convert list of dash components with properties to list of strings"""
    return [
        comp.component_id + "." + comp.component_property
        for comp in component_list
    ]


def pickctx(
    ctx: dash._callback_context.CallbackContext,
    component_list: Iterable["Component"],
) -> dict:
    """states and inputs of dash callback context if component is in
    component_list"""
    comp_strings = comps_to_strings(component_list)
    cats = []
    if ctx.states:
        cats.append(ctx.states)
    if ctx.inputs:
        cats.append(ctx.inputs)
    picked = [pickitems(cat, comp_strings) for cat in cats]
    if picked:
        return merge(picked)


def keygrab(dict_list: Iterable[Mapping], key: Any, value: Any) -> Mapping:
    """returns first element of dict_list such that element[key]==value"""
    return next(filter(lambda x: x[key] == value, dict_list))


def ctxdict(ctx: dash._callback_context.CallbackContext) -> dict:
    return {
        "triggered": ctx.triggered,
        "inputs": ctx.inputs,
        "states": ctx.states,
        "response": ctx.response,
    }


def triggered_by(component_id: str) -> bool:
    """
    did a component matching this id string trigger the callback this function
    is called in the context of? will only function if called inside a
    callback.
    """
    if component_id in dash.callback_context.triggered[0]["prop_id"]:
        return True
    return False


def trigger_index(ctx: dash._callback_context.CallbackContext) -> int:
    """
    dash.callbackcontext -> int, where int is the index of the triggering
    component
    assumes there is exactly one triggering component and it has an index!
    """
    trigger_id = ctx.triggered[0]["prop_id"]
    index_index = re.search("index", trigger_id).span()[1] + 2
    return int(trigger_id[index_index])


# ## dash dev tools


def dump_it(data, loud=True):
    """dump data as json"""
    dump = json.dumps(data, indent=2)
    if loud:
        print(dump)
    return dump


def make_printer(
    element, prop, app, print_target="print", process_function=dump_it
):
    """
    utility callback factory. impure! inserts the callback into the tree
    when called.
    when called, creates a callback to print property of element in app to
    print_target
    """

    def print_callback():
        app.callback(Output(print_target, "children"), [Input(element, prop)])(
            process_function
        )

    print_callback()


# ## lambda replacements


def in_me(container: Iterable) -> Callable:
    """returns function that checks if all its arguments are in container"""
    inclusion = partial(contains, container)

    def is_in(*args: Any) -> bool:
        return reduce(and_, map(inclusion, args))

    return is_in


# ## generic


def get_if(boolean: bool, dictionary: Mapping, key: Any) -> Any:
    """return dictionary[key] iff boolean; otherwise return None"""
    if boolean:
        return dictionary.get(key)
    return None


def get_parameters(func: Callable) -> list[str]:
    return [param.name for param in signature(func).parameters.values()]


def partially_evaluate_from_parameters(
    func: Callable, parameters: Mapping
) -> Callable:
    """
    return a copy of the input function partially evaluated from any value
    of the dict with a key matching a named argument of the function.
    useful for things like inserting 'settings' variables into large numbers
    of functions.

    for instance:

    def add(a, b):
        return a + b
    parameters = {'b':1, 'c':3}
    add_1 = partially_evaluate_from_parameters(add, parameters)
    assert add_1(2) == 3
    """
    return partial(func, **pickitems(parameters, get_parameters(func)))


def listify(thing: Any) -> list:
    """Always a list, for things that want lists"""
    if isiterable(thing):
        return list(thing)
    return [thing]


def none_to_empty(thing: Any) -> Any:
    if thing is None:
        return ""
    return thing


# ## search functions


# some of the following functions hit the database multiple times during
# evaluation.
# this is necessary to allow flexibly loose and strict phrasal searches.
# it potentially reduces efficiency a great deal and is terrain for
# optimization
# if and when required.


def flexible_query(
    queryset: "QuerySet", field: "str", value: Any
) -> "QuerySet":
    """
    little search function that checks exact and loose phrases.
    have to hit the database to do this, so less efficient than using
    others that don't
    """
    # allow exact phrase searches
    query = field + "__iexact"
    if queryset.filter(**{query: value}):
        return queryset.filter(**{query: value})
    # otherwise treat multiple words as an 'or' search",
    query = field + "__icontains"
    filters = [queryset.filter(**{query: word}) for word in value.split(" ")]
    return reduce(or_, filters)


def inflexible_query(
    queryset: "QuerySet", field: "str", value: Any
) -> "QuerySet":
    """little search function that checks only exact phrases"""
    query = field + "__iexact"
    return queryset.filter(**{query: value})


def term_search(
    queryset: "QuerySet", field: "str", value: Any, inflexible=False
) -> "QuerySet":
    """
    model, string, string or number or whatever -> queryset
    search for strings or whatever within a field of a model.
    """
    # toss out "any" searches
    if value.lower() == "any":
        return queryset
    # allow inexact phrase matching?
    search_function = flexible_query
    if inflexible:
        search_function = inflexible_query
    return search_function(queryset, field, value)


def value_fetch_search(
    queryset: "QuerySet", field: "str", value_list: Iterable
) -> "QuerySet":
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
    queryset: "QuerySet",
    field: str,
    interval_begin: Any = None,
    interval_end: Any = None,
    strictly: bool = False,
) -> "QuerySet":
    """
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


def multiple_field_search(
    queryset: "QuerySet", parameters: Iterable
) -> "QuerySet":
    """
    dispatcher that handles multiple search parameters and returns a queryset.
    accepts options in dictionaries to search 'numerical' intervals
    or stringlike terms.
    """
    results = []
    for parameter in parameters:
        # do a relations-on-orderings search if requested
        if parameter.get("value_type") == "quant":
            if "value_list" in parameter.keys():
                search_result = value_fetch_search(
                    queryset, parameter["field"], parameter["value_list"]
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
        # otherwise just look for term matches;
        # "or" them within a category
        else:
            param_results = []
            for term in parameter["term"]:
                param_result = term_search(
                    queryset,
                    parameter["field"],
                    term,
                    parameter.get("flexible"),
                )
                param_results.append(param_result)
            results.append(reduce(or_, param_results))
    return reduce(and_, results)


def df_flexible_query(
    metadata_df: pd.DataFrame, field: "str", value: Any
) -> pd.DataFrame.index:
    """
    little search function that checks exact and loose phrases.
    """
    # allow exact phrase searches
    exact = metadata_df[field] == value
    # noinspection PyTypeChecker
    if any(exact):
        return metadata_df[field].loc[exact].index
    # otherwise treat multiple words as an 'or' search",
    return metadata_df.loc[metadata_df[field].isin(value.split(" "))].index


def df_inflexible_query(
    metadata_df: pd.DataFrame, field: "str", value: Any
) -> pd.DataFrame.index:
    """little search function that checks only exact phrases"""
    return metadata_df.loc[metadata_df[field] == value].index


def df_term_search(
    metadata_df: pd.DataFrame, field: "str", value: Any, inflexible=False
) -> pd.DataFrame.index:
    """
    model, string, string or number or whatever -> queryset
    search for strings or whatever within a field of a model.
    """
    # toss out "any" searches
    if value.lower() == "any":
        return metadata_df.index
    # allow inexact phrase matching?
    search_function = df_flexible_query
    if inflexible:
        search_function = df_inflexible_query
    return search_function(metadata_df, field, value)


def df_interval_search(
    metadata_df: pd.DataFrame,
    field: str,
    interval_begin: Any = None,
    interval_end: Any = None,
    strictly: bool = False,
) -> pd.DataFrame.index:
    """
    interval_begin and interval_end must be of types for which
    the elements of the set of the chosen attribute of the elements of queryset
    possess a complete ordering exposed to pandas
    if both interval_begin and interval_end are defined, returns
    all entries > begin and < end.
    if only interval_begin is defined: all entries > begin.
    if only interval_end is defined: all entries < end.
    if neither: trivially returns the index.)
    """
    if strictly:
        less_than = lt
        greater_than = gt
    else:
        less_than = le
        greater_than = ge

    # these variables are queries defined by the ordering on this attribute.
    greater_than_begin = (
        metadata_df[field]
        .loc[greater_than(metadata_df[field], interval_begin)]
        .index
    )
    less_than_end = (
        metadata_df[field]
        .loc[less_than(metadata_df[field], interval_end)]
        .index
    )

    # select only entries with attribute greater than interval_begin (if
    # defined)
    # and less than interval_end (if defined)
    indices = [metadata_df.index]
    if interval_begin:
        indices.append(greater_than_begin)
    if interval_end:
        indices.append(less_than_end)
    return reduce(pd.Index.intersection, indices)


def df_value_fetch_search(
    metadata_df: pd.DataFrame, field: "str", value_list: Iterable
) -> pd.DataFrame.index:
    return metadata_df[field].isin(value_list).index


def df_multiple_field_search(
    metadata_df: pd.DataFrame, parameters: Iterable
) -> list:
    """
    dispatcher that handles multiple search parameters and returns a queryset.
    accepts options in dictionaries to search 'numerical' intervals
    or stringlike terms.
    """
    results = []
    for parameter in parameters:
        # do a relations-on-orderings search if requested
        if parameter.get("value_type") == "quant":
            if "value_list" in parameter.keys():
                search_result = df_value_fetch_search(
                    metadata_df, parameter["field"], parameter["value_list"]
                )
            else:
                search_result = df_interval_search(
                    metadata_df,
                    parameter["field"],
                    # begin and end and strictly are optional
                    parameter.get("begin"),
                    parameter.get("end"),
                    parameter.get("strictly"),
                )
            results.append(search_result)
        # otherwise just look for term matches;
        # "or" them within a category
        else:
            param_results = []
            for term in parameter["term"]:
                param_result = df_term_search(
                    metadata_df,
                    parameter["field"],
                    term,
                    parameter.get("flexible"),
                )
                param_results.append(param_result)
            results.append(reduce(pd.Index.union, param_results))
    return list(reduce(pd.Index.intersection, results))


def fetch_css_variables(css_file: str = DEFAULT_CSS_PATH) -> dict[str, str]:
    css_variable_dictionary = {}
    with open(css_file) as stylesheet:
        css_lines = stylesheet.readlines()
    for line in css_lines:
        if not re.match(r"\s+--", line):
            continue
        key, value = re.split(r":\s+", line)
        key = re.sub(r"(--)|[ :]", "", key)
        value = re.sub(r"[ \n;]", "", value)
        css_variable_dictionary[key] = value
    return css_variable_dictionary


# TODO: this can be made more efficient using idiomatic django cursor calls
def model_metadata_df(
    model: Any, relation_names: Optional[list[str]] = None
) -> pd.DataFrame:
    try:
        dict_function = getattr(model, "metadata_dict")
    except AttributeError:
        dict_function = modeldict
    if relation_names is None:
        relation_names = []
    value_list = []
    id_list = []
    for obj in model.objects.all().prefetch_related(*relation_names):
        value_list.append(dict_function(obj))
        id_list.append(obj.id)
    return pd.DataFrame(value_list, index=id_list)


# not currently used
def regex_keyfilter(regex: str, dictionary: Mapping) -> dict:
    return {
        key: value
        for key, value in dictionary.items()
        if re.search(regex, key)
    }


def dedict(dictionary: Mapping) -> Any:
    assert len(dictionary) == 1
    return dictionary[list(dictionary)[0]]
