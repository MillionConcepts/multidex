from copy import deepcopy
from operator import or_, and_, contains

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
from toolz import keyfilter, merge, isiterable, get_in

from utils import (
    rows,
    columns,
    qlist,
    keygrab,
    in_me,
    flexible_query,
    multiple_field_search,
    pickitems,
    pickctx,
)

"""
functions used for selecting, manipulating, and drawing spectral data within plotly-dash objects.
this module is _separate from app structure definition_. 
these functions are partially defined and/or passed to callback decorators in order to
generate flow control within a dash app.
"""

#### cache functions

# many of the other functions in this module take outputs of these two functions as arguments.
# returning a function that calls a specific cache defined in-app
# allows us to share data between defined clusters of dash objects.
# the specific cache is in some sense a set of pointers that serves as a namespace.


def cache_set(cache):
    def cset(key, value):
        return cache.set(key, value)

    return cset


def cache_get(cache):
    def cget(key):
        return cache.get(key)

    return cget


def make_axis(settings, queryset, suffix):

    # what is requested function or property?
    axis_option = settings["axis-option-" + suffix]
    # what are the characteristics of that function or property?
    props = keygrab(
        queryset.model.axis_value_properties, "value", axis_option
    )
    if props["type"] == "method":
        # we assume here that 'methods' all take a spectrum's filter names
        # as arguments, and have arguments in an order corresponding to the inputs.
        filt_args = [
            settings["filter-" + str(ix) + "-" + suffix]
            for ix in range(1, props["arity"] + 1)
        ]
        # if some values are blank, don't try to call the function
        if all(filt_args):
            return [
                getattr(spectrum, props["value"])(*filt_args)
                for spectrum in queryset
            ]
        return None
    if props["type"] == "parent_property":
        return [
            getattr(spectrum.observation, props["value"])
            for spectrum in queryset
        ]


def main_graph():
    return dcc.Graph(
        id="main-graph", figure=go.Figure(), style={"height": "70vh"}
    )


def spec_graph():
    fig=go.Figure()
    fig.update_layout(margin={'l':10,'r':10,'t':10,'b':10})
    return dcc.Graph(id="spec-graph", figure=fig)


def main_graph_scatter(x_axis, y_axis, text, customdata):
    """partial placeholder scatter function for main graph"""
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=x_axis,
            y=y_axis,
            # change this to be hella popup text
            text=text,
            customdata=customdata,
            mode="markers",
            marker={"color": "blue"},
        )
    )
    return fig


def spec_graph_line(x_axis, y_axis):
    """partial placeholder line graph for individual spectra"""
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=x_axis, y=y_axis, mode="lines+markers"))
    fig.update_layout(margin={'l':10,'r':10,'t':10,'b':10})
    return fig


def axis_value_drop(spec_model, element_id):
    """
    dropdown for selecting calculation options for axes
    """
    options = [
        {"label": option["label"], "value": option["value"]}
        for option in spec_model.axis_value_properties
    ]
    return dcc.Dropdown(
        id=element_id, options=options, value=options[0]["value"]
    )


# this is somewhat nasty.
# is there a cleaner way to do this?
# flow control becomes really hard if we break the function up.
# it requires triggers spread across multiple divs or cached globals
# and is much uglier than even this
def recalculate_graph(*args, x_inputs, y_inputs, graph_function, cget, cset):
    ctx = dash.callback_context
    # do nothing on page load
    if not ctx.triggered:
        raise PreventUpdate
    queryset = cget("queryset")
    x_settings = pickctx(ctx, x_inputs)
    y_settings = pickctx(ctx, y_inputs)
    # this is for future functions that display or recall
    # the settings used to generate a graph
    cset("x_settings", x_settings)
    cset("y_settings", y_settings)
    # we can probably get away without any fancy flow control
    # b/c memoization...if it turns out that hitting the cache this way sucks,
    # we will add it.
    x_axis = make_axis(x_settings, queryset, suffix="x.value")
    y_axis = make_axis(y_settings, queryset, suffix="y.value")
    # these text and customdata choices are likely placeholders
    text = [spec.observation.mcam + " " + spec.roi_color for spec in queryset]
    customdata = [spec.id for spec in queryset]
    # this case is most likely shortly after page load
    # when not everything is filled out
    if not (x_axis and y_axis):
        raise PreventUpdate
    return graph_function(x_axis, y_axis, text, customdata)


def field_values(queryset, field):
    """
    generates dict if all unique values in model's field
    + any and blank, for passing to HTML select constructors
    as this is based on current queryset,
    it will by default display options as constrained by other search
    parameters. this has upsides and downsides.
    it will also lead to odd behavior if care is not given.
    maybe it's a bad idea.
    """
    if not field:
        return
    options_list = [
        {"label": item, "value": item}
        for item in set(qlist(queryset, field))
        if item not in ["", "nan"]
    ]
    special_options = [
        {"label": "any", "value": "any"},
        # too annoying to detect all 'blank' things atm
        # {'label':'no assigned value','value':''}
    ]
    return special_options + options_list


def filter_drop(model, element_id):
    """dropdown for filter selection"""
    return dcc.Dropdown(
        id=element_id,
        options=[{"label": filt, "value": filt} for filt in model.filters],
        style={"width": "10rem", "display": "inline-block"},
    )


def field_drop(fields, element_id, index):
    """dropdown for field selection -- no special logic atm"""
    return dcc.Dropdown(
        id={"type": element_id, "index": index},
        options=[
            {"label": field["label"], "value": field["label"]}
            for field in fields
        ],
    )


def model_range_entry(element_id, index):
    """
    pair of entry fields for selecting a range of values for a
    quantitatively-valued field.
    """
    return [
        dcc.Input(
            id={"type": element_id + "-begin", "index": index}, type="text"
        ),
        dcc.Input(
            id={"type": element_id + "-end", "index": index}, type="text"
        ),
    ]


def model_options_drop(field, element_id, index):
    """
    dropdown for selecting search values for a specific field
    could end up getting unmanageable as a UI element
    """
    return dcc.Dropdown(
        id={"type": element_id, "index": index},
        options={"label": "any", "value": "any"},
        multi=True,
    )


def model_range_display(element_id, index):
    """placeholder area for displaying range for number field searches"""
    return html.P(id={"type": element_id, "index": index})


def search_parameter_div(index, searchable_fields, cget):
    return html.Div(
        children=[
            field_drop(searchable_fields, "field-search", index),
            model_options_drop("group", "term-search", index),
            *model_range_entry("number-search", index),
            model_range_display("number-range-display", index),
        ]
    )


def toggle_search_input_visibility(field, spec_model):
    """
    toggle between showing and hiding term drop down / number range boxes.
    right now just returns simple dicts to the dash callback, as appropriate.
    """
    if not field:
        raise PreventUpdate

    if (
        keygrab(spec_model.searchable_fields, "label", field)["value_type"]
        == "quant"
    ):
        return [{"display": "none"}, {}, {}]
    return [{}, {"display": "none"}, {"display": "none"}]


def spectrum_values_range(queryset, field, field_type):
    """
    returns minimum and maximum values of property within queryset of spectra.
    for cueing or aiding searches.
    """
    if field_type == "parent_property":
        values_list = sorted(
            [
                getattr(getattr(item, "observation"), field)
                for item in queryset.prefetch_related("observation")
            ]
        )
    else:
        values_list = sorted([item[field] for item in queryset.objects.all()])
    return (values_list[0], values_list[-1])


def update_search_options(field, cget):
    """
    populate term values and parameter range as appropriate when different fields are selected in the search interface
    currently this cascades in narrowness in a not-totally predictable way as new terms are added.
    this may or may not end up being desirable.
    """
    if not field:
        raise PreventUpdate
    queryset = cget("queryset")

    props = keygrab(queryset.model.searchable_fields, "label", field)
    # if it's a field we do number interval searches on, reset term interface and show number ranges
    # in the range display
    if props["value_type"] == "quant":
        return [
            {"label": "any", "value": "any"},
            "minimum/maximum: "
            + str(spectrum_values_range(queryset, field, props["type"])),
        ]

    # otherwise, populate the term interface and reset the range display
    return [field_values(queryset, field), ""]


def change_calc_input_visibility(calc_type, spec_model):
    """
    turn visibility of filter dropdowns (and later other inputs)
    on and off in response to changes in arity / type of 
    requested calc
    """
    props = keygrab(spec_model.axis_value_properties, "value", calc_type)
    # 'methods' are specifically those methods of spec_model
    # that take its filters as arguments
    if props["type"] == "method":
        return [
            {"width": "10rem", "display": "inline-block"}
            if x < props["arity"]
            else {"width": "10rem", "display": "none"}
            for x in range(3)
        ]
    return [{"display": "none"} for x in range(3)]


def handle_graph_search(model, parameters):
    """
    dispatcher / manager for user-issued searches within the graph interface.
    fills fields from model definition and feeds resultant list to a general-
    purpose search function.
    """
    # add value_type and type information to dictionaries (based on
    # search properties defined in the model)
    for parameter in parameters:
        field = parameter.get("field")
        if field:
            props = keygrab(model.searchable_fields, "label", field)
            parameter["value_type"] = props["value_type"]
            # format references to parent observation appropriately for Q objects
            if props["type"] == "parent_property":
                parameter["field"] = "observation__" + field

    # toss out 'any' entries -- they do not restrict the search
    parameters = [
        parameter
        for parameter in parameters
        if not (
            (parameter.get("value_type") == "qual")
            and parameter.get("term") == "any"
        )
    ]

    # do we have any actual constraints? if not, return the entire data set
    if not parameters:
        return model.objects.all()
    # otherwise, actually perform a search
    return multiple_field_search(
        model.objects.all().prefetch_related("observation"), parameters
    )


def update_queryset(
    n_clicks,
    fields,
    terms,
    begin_numbers,
    end_numbers,
    cget,
    cset,
    spec_model,
):
    """
    updates the spectra displayed in the graph view.
    """
    # don't do anything on page load
    # or if a blank request is issued
    if not (fields and (terms or begin_numbers)):
        raise PreventUpdate
    # construct a list of search parameters from the filled inputs
    search_list = [
        {"field": field, "term": term, "begin": begin, "end": end}
        for field, term, begin, end in zip(
            fields, terms, begin_numbers, end_numbers
        )
        if (field is not None and (term is not None or begin is not None))
    ]
    # if every search parameter is blank, don't do anything
    if not search_list:
        raise PreventUpdate

    # if the search parameters have changed,
    # make a new queryset and trigger graph update
    # using copy.deepcopy here to avoid passing doubly-ingested input back after the check
    # although on the other hand it should be memoized -- but still

    # TODO: this isn't handling cases that return an empty queryset -- check this
    if handle_graph_search(spec_model, deepcopy(search_list)) != cget(
        "queryset"
    ):
        cset(
            "queryset",
            handle_graph_search(spec_model, search_list).prefetch_related(
                "observation"
            ),
        )
        return n_clicks


def add_dropdown(n_clicks, children, spec_model, cget):
    """
    adds another dropdown for search constraints.
    i guess we should also have a button to remove or reset constraints!
    """
    searchable_fields = spec_model.searchable_fields
    children.append(search_parameter_div(n_clicks, searchable_fields, cget))
    return children
