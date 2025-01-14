"""assorted utility functions for project"""

import datetime as dt
import json
import re
import sys
from hashlib import md5
from collections import defaultdict
from functools import partial, reduce
from inspect import signature, getmembers
from operator import and_, gt, ge, lt, le, contains
from pathlib import Path
from string import whitespace, punctuation
from types import MappingProxyType as MPt
from typing import (
    Callable,
    Iterable,
    Any,
    TYPE_CHECKING,
    Mapping,
    Union,
    Optional,
    Sequence, MutableMapping,
)

from cytoolz import curry, keyfilter
import dash
from dash import html
from dash.dependencies import Input, Output
from dustgoggles.structures import dig_and_edit
import Levenshtein as lev
import numpy as np
import pandas as pd
from toolz import merge

if TYPE_CHECKING:
    from dash.development.base_component import Component
    from django.db.models.query import QuerySet
    from django.db.models import Model


DEFAULT_CSS_PATH = str(
    Path(Path(__file__).parent, "plotter/application/assets/css/main.css")
)

# generic

def re_get(mapping, pattern):
    for key in mapping.keys():
        if re.search(pattern, key):
            return mapping[key]
    return None


def integerize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocessing function for CSV output.

    Checks all floating-point columns of `df` to see if they are 'actually'
    integer columns converted to float by numpy/pandas due to the presence of
    nulls. If so, modifies `df` inplace to replace them with columns of object
    type that replaces those floats with stringified integer versions (i.e.,
    removing the superfluous ".0") and replaces all invalid values with "".
    """
    for colname, col in df.items():
        if not pd.api.types.is_float_dtype(col):
            continue
        notna = col[~col.isna()]
        if not (notna.round() == notna).all():
            continue
        strings = pd.Series("", index=col.index, dtype=str)
        strings[notna.index] = notna.astype(str).str.extract(r"(\d+)\.")[0]
        df[colname] = strings
    return df


# TODO: similar thing in plotter.application.run should be replaced with this
def seconds_since_beginning_of_day(time: dt.time) -> float:
    return (
        dt.datetime.combine(dt.date(1, 1, 1), time)
        - dt.datetime(1, 1, 1, 0, 0, 0)
    ).total_seconds()


def seconds_since_beginning_of_day_to_iso(
    seconds: Optional[int], round_to=0
) -> Optional[str]:
    if seconds is None:
        return None
    hour = int(seconds // 3600)
    remainder = seconds - hour * 3600
    minute = int(remainder // 60)
    seconds = round(remainder - minute * 60, round_to)
    return ":".join(map(lambda t: str(t).zfill(2), (hour, minute, seconds)))


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
            de_noned_list.append("none")
    return de_noned_list


def hash_strings(strings: Iterable[str], key) -> dict:
    unique_string_values = sorted(list(set(strings)), key=key, reverse=True)
    return {string: ix for ix, string in enumerate(unique_string_values)}


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
    if "field_names" in dir(django_model_object):
        return {
            field_name: getattr(django_model_object, field_name)
            for field_name in django_model_object.field_names
        }
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


def dict_to_paragraphs(
    dictionary, style=None, ordering=None, filterfalse=True
):
    """
    parses dictionary to list of dash <p> components
    """
    if style is None:
        style = {"margin": 0, "fontSize": 14}

    def make_paragraph(key, value):
        return html.P(str(key) + " " + str(value), style=style)

    if ordering is None:
        ordering = []
    from cytoolz import valfilter

    if filterfalse is True:
        dictionary = valfilter(lambda x: x not in (False, None), dictionary)
    ordered_grafs = [
        make_paragraph(key, dictionary.get(key))
        for key in ordering
        if key in dictionary.keys()
    ]
    unordered_grafs = [
        make_paragraph(key, value)
        for key, value in dictionary.items()
        if key not in ordering
        # TODO: very, very hacky
        and not re.search(r"_[hwad]+mag$", key)
    ]
    return ordered_grafs + unordered_grafs


def pickitems(dictionary: Mapping, some_list: Iterable) -> dict:
    """items of dict where key is in some_list"""
    return keyfilter(in_me(some_list), dictionary)


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


def none_to_empty(thing: Any) -> Any:
    if thing is None:
        return ""
    return thing


# ## search functions
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
    if str(value).lower() == "any":
        return metadata_df.index
    # allow inexact phrase matching?
    search_function = df_flexible_query
    if inflexible:
        search_function = df_inflexible_query
    return search_function(metadata_df, field, value)


def df_interval_search(
    metadata_df: pd.DataFrame,
    column: str,
    interval_begin: Any = None,
    interval_end: Any = None,
    strictly: bool = False,
) -> pd.DataFrame.index:
    """
    interval_begin and interval_end must be of types for which the elements of
    column from metadata_df possess a complete ordering exposed to pandas.
    if both interval_begin and interval_end are defined, returns
    all entries > begin and < end.
    if only interval_begin is defined: all entries > begin.
    if only interval_end is defined: all entries < end.
    if neither: trivially return the index.
    """
    if strictly:
        less_than, greater_than = (lt, gt)
    else:
        less_than, greater_than = (le, ge)
    indices = [metadata_df.index]
    for relation, bound in zip(
        (greater_than, less_than), (interval_begin, interval_end)
    ):
        if bound is None:
            continue
        indices.append(
            metadata_df[column].loc[relation(metadata_df[column], bound)].index
        )
    return reduce(pd.Index.intersection, indices)


def df_value_fetch_search(
    metadata_df: pd.DataFrame, field: "str", value_list: Iterable
) -> pd.DataFrame.index:
    return metadata_df.loc[metadata_df[field].isin(value_list)].index


def df_quant_field_search(search_df, parameter):
    if "value_list" in parameter.keys():
        return df_value_fetch_search(
            search_df, parameter["field"], parameter["value_list"]
        )
    else:
        return df_interval_search(
            search_df,
            parameter["field"],
            # begin and end and strictly are optional
            parameter.get("begin"),
            parameter.get("end"),
            parameter.get("strictly"),
        )


def df_qual_field_search(search_df, parameter):
    param_results = []
    for term in parameter["terms"]:
        param_result = df_term_search(search_df, parameter["field"], term)
        param_results.append(param_result)
    return reduce(pd.Index.union, param_results)


def df_multiple_field_search(
    search_df: pd.DataFrame,
    tokens: dict,
    parameters: Sequence[Mapping],
    logical_quantifier: str,
) -> list:
    """
    dispatcher that handles multiple search parameters and returns a queryset.
    accepts options in dictionaries to search 'numerical' intervals
    or stringlike terms.
    """
    results = []
    for parameter in parameters:
        valtype = parameter.get("value_type")
        # permit free text search
        if parameter.get('is_free') is True and valtype != "quant":
            result = loose_match(parameter['free'], tokens[parameter['field']])
            if result is None:
                result = search_df.index
        # do a relations-on-orderings search if requested
        elif parameter.get("value_type") == "quant":
            result = df_quant_field_search(search_df, parameter)
        # otherwise just look for term matches;
        # "or" them within a category
        elif parameter.get('terms') in ('', []):
            result = search_df.index
        else:
            # exact match
            result = df_qual_field_search(search_df, parameter)
        # allow all missing values if requested
        if parameter["null"] is True:
            result = pd.Index.union(
                result,
                search_df.loc[search_df[parameter["field"]].isna()].index,
            )
        # take complement if requested
        if parameter["invert"] is True:
            result = pd.Index(
                [ix for ix in search_df.index if ix not in result]
            )
        results.append(result)
    if logical_quantifier == "AND":
        index_logic_method = pd.Index.intersection
    else:
        index_logic_method = pd.Index.union
    return list(reduce(index_logic_method, results))


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
    model: Any,
    relation_names: Optional[list[str]] = None,
    dict_function: Optional[Callable] = None,
) -> pd.DataFrame:
    if dict_function is None:
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


# TODO: these kinds of printing rules probably need to go on individual
#  models for cross-instrument compatibility
def rearrange_band_depth_for_title(text: str) -> str:
    filts = re.split(r"([L|R]?\d[RGB]?)", text, maxsplit=0)
    return (
        f"{filts[0]}{filts[5]}, " f"shoulders at {filts[1]} and " f"{filts[3]}"
    )


def insert_wavelengths_into_text(text: str, spec_model: "Model") -> str:
    if "depth" in text:
        text = rearrange_band_depth_for_title(text)
    for filt, wavelength in (
        spec_model.filters | spec_model.virtual_filters
    ).items():
        text = re.sub(filt, filt + " (" + str(wavelength) + "nm)", text)
    text = re.sub(r"_", r" ", text)
    return text


def directory_of(path: Path) -> str:
    if path.is_dir():
        return str(path)
    return str(path.parent)


def get_verbose_name(field_name, model):
    return next(
        filter(lambda f: f.name == field_name, model._meta.fields)
    ).verbose_name


def patch_settings_from_module(settings, module_name):
    settings = {
        name: setting for name, setting in settings if "SETTING" in name
    }
    patches = {
        name: patch
        for name, patch in getmembers(
            sys.modules[module_name], lambda obj: isinstance(obj, Mapping)
        )
        if name in settings.keys()
    }
    for name, patch in patches.items():
        settings[name] |= patch


def tokenize(text):
    lowered = text.lower()
    return re.split(rf"[{punctuation + whitespace}]+", lowered)


def tokenize_series(series):
    return series.str.lower().str.replace(
        rf"[{punctuation + whitespace}]+", "_", regex=True
    ).str.split("_")


def make_tokens(metadata):
    fields = {}
    # TODO: don't tokenize quant fields, waste of time
    for colname, col in metadata.astype(str).items():
        coltoks = tokenize_series(col).tolist()
        lower = col.str.lower().tolist()
        records = [
            {'tokens': tokens, 'text': text, 'ix': ix}
            for tokens, text, ix in zip(coltoks, lower, col.index)
        ]
        fields[colname] = records
    tokenized = {}
    for rec_name, recs in fields.items():
        tokenized[rec_name] = defaultdict(list)
        for rec in recs:
            tokenized[rec_name][rec['text']].append(rec['ix'])
            for token in rec['tokens']:
                tokenized[rec_name][token].append(rec['ix'])
    return tokenized


def loose_match(term, tokens, cutoff_distance=2):
    if term is None:
        return None
    term = term.lower()
    matches, keys = [], list(tokens.keys())
    for word in set(filter(None, map(str.strip, term.split(";")))):
        if re.match(r"[\"'].*?[\"]", word) is not None:
            if (tokmatch := word.strip("\"'")) in keys:
                matches += tokens[tokmatch]
            continue
        # noinspection PyArgumentList
        for i, distance in enumerate(map(curry(lev.distance)(word), tokens)):
            if distance <= cutoff_distance:
                matches += tokens[keys[i]]
    return pd.Index(matches)


def freeze_nested_mapping(m: MutableMapping):
    return MPt(
        dig_and_edit(
            m,
            lambda _, v: isinstance(v, MutableMapping),
            lambda _, v: MPt(v),
            (MutableMapping,)
        )
    )


def md5sum(path):
    hasher = md5()
    with open(path, 'rb') as f:
        for c in iter(lambda: f.read(8192), b''):
            hasher.update(c)
    return hasher.hexdigest()