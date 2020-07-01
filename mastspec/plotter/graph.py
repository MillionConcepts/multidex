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
    eta_methods,
    qlist,
    keygrab,
    in_me,
    particular_fields_search,
    pickitems,
    pickctx,
)

"""constants / settings"""

AXIS_VALUE_PROPERTIES = [
    {"label":"band average","value":"band_avg","type":"method","arity":2},
    {"label":"band maximum","value":"band_max","type":"method","arity":2},
    {"label":"band minimum","value":"band_min","type":"method","arity":2},
    {"label":"ratio","value":"ref_ratio","type":"method","arity":2},
    {"label":"band depth at middle filter","value":"band_depth_custom","type":"method","arity":3},
    {"label":"band depth at band minimum", "value":"band_depth_min","type":"method","arity":2},
    {"label":"band value", "value":"ref","type":"method","arity":1},
    {"label":"sol", "value":"sol","type":"parent_property"}
    ]


"""functions used for selecting, manipulating, and drawing spectral data within plotly-dash objects"""


def cache_set(cache):
    def cset(key, value):
        return cache.set(key, value)

    return cset


def cache_get(cache):
    def cget(key):
        return cache.get(key)

    return cget


def make_axis(settings, queryset, suffix):

    # grab text for everything in queryset

    # what is requested function or property?
    axis_option = settings["axis-option-" + suffix]
    # what are the characteristics of that function or property?
    props = keygrab(AXIS_VALUE_PROPERTIES, "value", axis_option)
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
    # do some other stuff to grab parent properties


def main_graph():
    return dcc.Graph(
        id="main-graph", figure=go.Figure(), style={"height": "100vh"}
    )


def scatter(x_axis, y_axis):
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=x_axis,
            y=y_axis,
            # change this to be hella popup text
            text=None,
            mode="markers",
            marker={"color": "blue"},
        )
    )
    return fig


def axis_value_drop(element_id):
    """
    dropdown for selecting calculation options for axes
    """
    options = [
        {"label": option["label"], "value": option["value"]}
        for option in AXIS_VALUE_PROPERTIES
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
    # this case is most likely shortly after page load
    # when not everything is filled out
    if not (x_axis and y_axis):
        raise PreventUpdate
    return graph_function(x_axis, y_axis)


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


def field_drop(fields, element_id):
    """dropdown for field selection -- no special logic atm"""
    return dcc.Dropdown(
        id=element_id,
        options=[{"label": field, "value": field} for field in fields],
    )


def model_options_drop(queryset, field, element_id):
    """
    dropdown for selecting search values for a specific field
    could end up getting unmanageable as a UI element
    """
    return dcc.Dropdown(
        id=element_id, options=field_values(queryset, field), multi=True
    )


def handle_search(model, search_dict, searchable_fields):
    """
    dispatcher. right now just handles 'no assigned'
    and 'any' cases
    """
    for field, value in search_dict.items():
        if "any" in value:
            return model.objects.all()
    return particular_fields_search(model, search_dict, searchable_fields)


def change_input_visibility(calc_type):
    """
    turn visibility of filter dropdowns (and later other inputs)
    on and off in response to changes in arity / type of 
    requested calc
    """
    props = keygrab(AXIS_VALUE_PROPERTIES, "value", calc_type)
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


def update_model_field(field, cget):
    """populate set of values when field in search box changes"""
    queryset = cget("queryset")
    return field_values(queryset, field)


def update_queryset(n_clicks, field, value, cget, cset):
    """
    updates the spectra displayed in the graph view.
    
    we'd actually like to extend this to include fields from both spec
    and obs
    
    and multiple fields
    """
    # don't do anything on page load
    # or if a partially blank request is issued
    if not (field and value):
        raise PreventUpdate

    # if the search parameters have changed,
    # make a new queryset and trigger graph update
    if handle_search(spec_model, {field: value}, [field]) != cget("queryset"):
        cset(
            "queryset",
            handle_search(
                spec_model, {field: value}, [field]
            ).prefetch_related("observation"),
        )
        return n_clicks
