"""
functions used for selecting, manipulating, and drawing spectral data
within plotly-dash objects. this module is _separate_ from app structure
definition_ and, to the extent possible, components. considering where
exactly...! these functions are partially defined and/or passed to callback
decorators in order to generate flow control within a dash app.
"""

import datetime as dt
import re
from ast import literal_eval
from collections import Iterable
from copy import deepcopy
from functools import reduce
from itertools import chain, cycle
from operator import or_
from typing import TYPE_CHECKING, Any, Callable, Optional

import dash
import dash_html_components as html
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate

from plotter import spectrum_ops
from plotter.components import (
    parse_model_quant_entry,
    search_parameter_div,
    viewer_tab,
    search_tab,
)
from plotter.spectrum_ops import filter_df_from_queryset
from plotter_utils import (
    djget,
    dict_to_paragraphs,
    rows,
    keygrab,
    pickctx,
    not_blank,
    not_triggered,
    trigger_index,
    triggered_by,
    seconds_since_beginning_of_day,
    arbitrarily_hash_strings,
    none_to_quote_unquote_none,
    field_values,
    fetch_css_variables,
    df_multiple_field_search,
    re_get,
)

from plotter.models import MSpec, ZSpec

if TYPE_CHECKING:
    from django.db.models import Model
    import flask_caching

css_variables = fetch_css_variables()
COLORBAR_SETTINGS = {
    "tickfont": {
        "family": "Fira Mono",
        "color": css_variables["midnight-ochre"],
    },
    "titlefont": {"family": "Fira Mono"}
}


# ### cache functions ###

# many of the other functions in this module take outputs of these two
# functions as arguments. returning a function that calls a specific cache
# defined in-app allows us to share data between defined clusters of dash
# objects. the specific cache is in some sense a set of pointers that serves
# as a namespace. this, rather than a global variable or variables, is used
# because Flask does not guarantee thread safety of globals.


def cache_set(cache: "flask_caching.Cache") -> Callable[[str, Any], bool]:
    def cset(key: str, value: Any) -> bool:
        return cache.set(key, value)

    return cset


def cache_get(cache: "flask_caching.Cache") -> Callable[[str], Any]:
    def cget(key: str) -> Any:
        return cache.get(key)

    return cget


def truncate_id_list_for_missing_properties(
    settings: dict,
    id_list: Iterable,
    input_suffixes: list,
    filter_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    spec_model,
):
    metadata_args = []
    filt_args = []
    indices = []
    for suffix in input_suffixes:
        axis_option = re_get(settings, "-graph-option-" + suffix)
        model_property = keygrab(
            spec_model.graphable_properties, "value", axis_option
        )
        if model_property["type"] == "method":
            # we assume here that 'methods' all take a spectrum's filter names
            # as arguments, and have arguments in an order corresponding to the
            # inputs.
            filt_args.append(
                [
                    re_get(settings, "-filter-" + str(ix) + "-" + suffix)
                    for ix in range(1, model_property["arity"] + 1)
                ]
            )
        else:
            metadata_args.append(axis_option)
    if filt_args:
        indices.append(
            filter_df.loc[id_list][list(set(chain.from_iterable(filt_args)))]
            .dropna()
            .index
        )
    if metadata_args:
        indices.append(
            metadata_df.loc[id_list][list(set(metadata_args))].dropna().index
        )
    return list(reduce(pd.Index.intersection, indices))


def perform_spectrum_op(
    id_list,
    spec_model,
    filter_df,
    settings,
    props,
    get_errors=False,
):
    # we assume here that 'methods' all take a spectrum's filter names
    # as arguments, and have arguments in an order corresponding to
    # the inputs.
    queryset_df = filter_df.loc[id_list]
    filt_args = [
        re_get(settings, "-filter-" + str(ix))
        for ix in range(1, props["arity"] + 1)
    ]
    spectrum_op = getattr(spectrum_ops, props["value"])
    title = props["value"] + " " + str(" ".join(filt_args))
    try:
        vals, errors = spectrum_op(
            queryset_df, spec_model, *filt_args, get_errors
        )
        if get_errors:
            return (
                list(np.array(vals)),
                list(np.array(errors)),
                title,
            )
        return list(vals.values), None, title
    except ValueError:  # usually representing intermediate input states
        raise PreventUpdate


def make_axis(
    settings: dict,
    id_list: Iterable,
    spec_model: Any,
    filter_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    get_errors: bool,
    _highlight,
) -> tuple[list[float], Optional[list[float]], str]:
    """
    make an axis for one of our graphs by looking at the appropriate rows from
    our big precalculated metadata / data dataframes; data has already been
    scaled and averaged as desired. expects a list that has been
    processed by truncate_id_list_for_missing_properties
    """
    # what is requested function or property?
    axis_option = re_get(settings, "-graph-option-")
    # what are the characteristics of that function or property?
    props = keygrab(spec_model.graphable_properties, "value", axis_option)
    if props["type"] == "method":
        return perform_spectrum_op(
            id_list,
            spec_model,
            filter_df,
            settings,
            props,
            get_errors,
        )
    value_series = metadata_df.loc[id_list][props["value"]]
    if props["value"] == "ltst":
        value_series = [
            instant.hour * 3600 + instant.minute * 60 + instant.second
            for instant in value_series
        ]
    return value_series, None, axis_option


def make_marker_properties(
    settings,
    id_list,
    spec_model,
    filter_df,
    metadata_df,
    _get_errors,
    highlight_id_list,
):
    """
    this expects an id list that has already
    been processed by truncate_id_list_for_missing_properties
    """
    marker_option = re_get(settings, "-graph-option-")
    props = keygrab(spec_model.graphable_properties, "value", marker_option)
    # it would really be better to do this in components
    # but is difficult because you have to instantiate the colorbar somewhere
    # it would also be better to style with CSS but it seems like plotly
    # really wants to put element-level style declarations on graph ticks!
    colorbar_dict = COLORBAR_SETTINGS.copy()
    if props["type"] == "method":
        property_list, _, title = perform_spectrum_op(
            id_list, spec_model, filter_df, settings, props
        )
    else:
        property_list, title = (
            metadata_df.loc[id_list][props["value"]].values,
            props["value"],
        )
    colorbar_dict |= {"title_text": title}
    if props["value_type"] == "qual":
        string_hash, color_indices = arbitrarily_hash_strings(
            none_to_quote_unquote_none(property_list)
        )
        colorbar_dict |= {
            "tickvals": list(string_hash.values()),
            "ticktext": list(string_hash.keys()),
        }
    else:
        if len(property_list) > 0:
            if isinstance(property_list[0], dt.time):
                property_list = list(
                    map(seconds_since_beginning_of_day, property_list)
                )
        color_indices = property_list
    colormap = re_get(settings, "-color.value")

    # define marker size settings
    # note that you have to define individual marker sizes
    # in order to be able to set marker outlines -- it causes them to be
    # drawn in some different (and more expensive) way
    opacity = 1
    if re_get(settings, "-highlight-toggle.value") == "on":
        marker_size = [
            32 if spectrum in highlight_id_list else 9 for spectrum in id_list
        ]
        opacity = 0.5
    elif re_get(settings, "-outline-radio.value") != "off":
        marker_size = [9 for _ in id_list]
    else:
        marker_size = 9

    # define marker outline
    if re_get(settings, "-outline-radio.value") != "off":
        marker_line = {
            "color": re_get(settings, "-outline-radio.value"),
            "width": 5,
        }
    else:
        marker_line = {}

    return {
        "marker": {
            "color": color_indices,
            "colorscale": colormap,
            "colorbar": go.scatter.marker.ColorBar(**colorbar_dict),
            "size": marker_size,
            "opacity": opacity,
        },
        "line": marker_line,
    }


def make_graph_display_settings(settings):
    settings_dict = {}
    axis_settings_dict = {}
    if re_get(settings, "-graph-bg"):
        settings_dict["plot_bgcolor"] = re_get(settings, "-graph-bg")
    if re_get(settings, "-gridlines") == "on":
        axis_settings_dict["showgrid"] = True
    else:
        axis_settings_dict["showgrid"] = False

    return settings_dict, axis_settings_dict


# this is somewhat nasty. is there a cleaner way to do this?
# flow control becomes really hard if we break the function up.
# it requires triggers spread across multiple divs or cached globals
# and is much uglier than even this. dash's restriction on callbacks to a
# single output makes it even worse. probably the best thing to do
# in the long run is to treat this basically as a dispatch function.
def recalculate_main_graph(
    *args,
    x_inputs,
    y_inputs,
    marker_inputs,
    graph_display_inputs,
    graph_function,
    cget,
    cset,
    spec_model,
    record_settings=True,
):
    ctx = dash.callback_context
    if ctx.triggered[0]["prop_id"] == "main-graph-bounds.value":
        zoom_string = ctx.triggered[0]["value"].split()
        if len(zoom_string) != 4:
            raise PreventUpdate
        graph = go.Figure(ctx.states["main-graph.figure"])
        graph.update_layout(
            {
                "xaxis": {
                    "range": [zoom_string[0], zoom_string[1]],
                    "autorange": False,
                },
                "yaxis": {
                    "range": [zoom_string[2], zoom_string[3]],
                    "autorange": False,
                },
            }
        )
        return graph

    x_settings = pickctx(ctx, x_inputs)
    y_settings = pickctx(ctx, y_inputs)
    marker_settings = pickctx(ctx, marker_inputs)
    graph_display_settings = pickctx(ctx, graph_display_inputs)
    if isinstance(ctx.triggered[0]["prop_id"], dict):
        if ctx.triggered[0]["prop_id"]["type"] == "highlight-trigger":
            if marker_settings["main-highlight-toggle.value"] == "off":
                raise PreventUpdate
    search_ids = cget("main_search_ids")
    highlight_ids = cget("main_highlight_ids")
    filter_df = cget("main_graph_filter_df")
    metadata_df = cget("metadata_df")
    if "error" in ctx.inputs["main-graph-error.value"]:
        get_errors = True
    else:
        get_errors = False
    truncated_ids = truncate_id_list_for_missing_properties(
        x_settings | y_settings | marker_settings,
        search_ids,
        ["x.value", "y.value", "marker.value"],
        filter_df,
        metadata_df,
        spec_model,
    )
    graph_content = [
        truncated_ids,
        spec_model,
        filter_df,
        metadata_df,
        get_errors,
        highlight_ids,
    ]
    x_axis, x_errors, x_title = make_axis(x_settings, *graph_content)
    y_axis, y_errors, y_title = make_axis(y_settings, *graph_content)
    marker_properties = make_marker_properties(marker_settings, *graph_content)
    truncated_metadata = metadata_df.loc[truncated_ids]
    feature_color = truncated_metadata["feature"].copy()
    no_feature_ix = feature_color.loc[feature_color.isna()].index
    feature_color.loc[no_feature_ix] = truncated_metadata["color"].loc[
        no_feature_ix
    ]
    text = (
        "sol"
        + truncated_metadata["sol"].astype(str)
        + " "
        + truncated_metadata["name"]
        + " "
        + feature_color
    )
    customdata = truncated_ids
    graph_display_dict, axis_display_dict = make_graph_display_settings(
        graph_display_settings
    )
    # for functions that (perhaps asynchronously) fetch the state of the graph.
    # this is another perhaps ugly flow control thing!
    # TODO (maybe): add graph display settings to recorded settings
    if record_settings:
        for parameter in (
            "x_settings",
            "y_settings",
            "marker_settings",
            "x_axis",
            "y_axis",
            "marker_properties",
            "text",
            "customdata",
            "graph_function",
        ):
            cset(parameter, locals()[parameter])
    graph_layout = ctx.states["main-graph.figure"]["layout"]

    # automatically reset graph zoom only if we're loading the page or
    # changing options and therefore scales
    if (
        not_triggered()
        or (
            ctx.triggered[0]["prop_id"]
            in [
                "main-graph-option-y.value",
                "main-graph-option-x.value",
            ]
        )
        or (len(ctx.triggered) > 1)
    ):
        zoom = None
    else:
        zoom = (graph_layout["xaxis"]["range"], graph_layout["yaxis"]["range"])
    # TODO: refactor (too many arguments)
    return graph_function(
        x_axis,
        y_axis,
        marker_properties,
        graph_display_dict,
        axis_display_dict,
        text,
        customdata,
        zoom,
        x_errors,
        y_errors,
        x_title,
        y_title,
    )


def toggle_search_input_visibility(field, *, spec_model):
    """
    toggle between showing and hiding term drop down / number range boxes.
    right now just returns simple dicts to the dash callback based on the
    field's value type
    (quant or qual) as appropriate.
    """
    if not field:
        raise PreventUpdate

    if (
        keygrab(spec_model.searchable_fields, "label", field)["value_type"]
        == "quant"
    ):
        return [{"display": "none"}, {}]
    return [{}, {"display": "none"}]


def style_toggle(style, style_property="display", states=("none", "revert")):
    """
    generic style-toggling function that just cycles
    style property of component between states
    by default it toggles visibility
    """
    if style is None:
        style = {}
    if style.get(style_property) not in states:
        style[style_property] = states[0]
        return style
    style_cycle = cycle(states)
    while next(style_cycle) != style.get(style_property):
        continue
    style[style_property] = next(style_cycle)
    return style


def toggle_panel_visibility(_click, panel_style, arrow_style, text_style):
    """
    switches collapsible panel between visible and invisible,
    and rotates and sets text on its associated arrow.
    """
    if not_triggered():
        raise PreventUpdate
    panel_style = style_toggle(panel_style, states=("none", "revert"))
    arrow_style = style_toggle(
        arrow_style, "WebkitTransform", ("rotate(45deg)", "rotate(-45deg)")
    )
    arrow_style = style_toggle(
        arrow_style, "transform", ("rotate(45deg)", "rotate(-45deg)")
    )
    text_style = style_toggle(text_style, states=("inline-block", "none"))
    return panel_style, arrow_style, text_style


def toggle_averaged_filters(
    do_average,
    # n_intervals,
    *,
    spec_model,
):
    if dash.callback_context.triggered[0]["prop_id"] == ".":
        raise PreventUpdate
    if "average" in do_average:
        options = [
            {"label": filt, "value": filt}
            for filt in spec_model.canonical_averaged_filters
        ]
    else:
        options = [
            {"label": filt, "value": filt} for filt in spec_model.filters
        ]
    ctx = dash.callback_context
    number_of_outputs = int(len(ctx.outputs_list) / 2)
    # from random import choice
    # TODO: nonsense debug option
    # return [
    #            options for _ in range(number_of_outputs)
    #        ] + [options[choice(range(len(options)))]['value']
    #        for _ in range(number_of_outputs)]
    cycler = cycle([0, 1, 2, 3])
    return [options for _ in range(number_of_outputs)] + [
        options[next(cycler)]["value"] for _ in range(number_of_outputs)
    ]


def spectrum_values_range(metadata_df, field):
    """
    returns minimum and maximum values of property within id list of spectra.
    for cueing or aiding searches.
    """
    values = metadata_df[field]
    if field == "ltst":
        values = [0 for _ in values]
    return values.min(), values.max()


def update_search_options(
    field, _load_trigger_index, current_quant_search, *, cget, spec_model
):
    """
    populate term values and parameter range as appropriate when different
    fields are selected in the search interface
    currently this does _not_ cascade according to selected terms. this may
    or may not be desirable
    """
    if not_triggered():
        raise PreventUpdate
    if not field:
        raise PreventUpdate
    is_loading = (
        "search-load-trigger" in dash.callback_context.triggered[0]["prop_id"]
    )
    props = keygrab(spec_model.searchable_fields, "label", field)
    metadata_df = cget("metadata_df")
    # if it's a field we do number interval searches on, reset term
    # interface and show number ranges in the range display. but don't reset
    # the number entries if we're in the middle of a load!
    if props["value_type"] == "quant":
        if is_loading:
            search_text = current_quant_search
        else:
            search_text = ""
        return [
            [{"label": "any", "value": "any"}],
            "minimum/maximum: "
            + str(spectrum_values_range(metadata_df, field))
            # TODO: this should be a hover-over tooltip eventually
            + """ e.g., '100--200' or '100, 105, 110'""",
            search_text,
        ]

    # otherwise, populate the term interface and reset the range display and
    # searches
    return [field_values(metadata_df, field), "", ""]


def trigger_search_update(_load_trigger, search_triggers):
    return ["bang" for _ in search_triggers]


def change_calc_input_visibility(calc_type, *, spec_model):
    """
    turn visibility of filter dropdowns (and later other inputs)
    on and off in response to changes in arity / type of
    requested calc
    """
    props = keygrab(spec_model.graphable_properties, "value", calc_type)
    # 'methods' are specifically those methods of spec_model
    # that take its filters as arguments
    if props["type"] == "method":
        return [
            {"display": "flex", "flexDirection": "column"}
            if x < props["arity"]
            else {"display": "none"}
            for x in range(3)
        ]
    return [{"display": "none"} for _ in range(3)]


def non_blank_search_parameters(parameters):
    entry_keys = ["term", "begin", "end", "value_list"]
    return [
        parameter
        for parameter in parameters
        if reduce(or_, [not_blank(parameter.get(key)) for key in entry_keys])
    ]


def handle_graph_search(metadata_df, parameters, spec_model):
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
            props = keygrab(spec_model.searchable_fields, "label", field)
            parameter["value_type"] = props["value_type"]
    # toss out blank strings, etc. -- they do not restrict the search
    parameters = non_blank_search_parameters(parameters)
    # do we have any actual constraints? if not, return the entire data set
    if not parameters:
        return list(metadata_df.index)
    # otherwise, actually perform a search
    return df_multiple_field_search(metadata_df, parameters)


def update_filter_df(
    _load_trigger,
    scale_to,
    average_filters,
    r_star,
    scale_trigger_count,
    *,
    cset,
    spec_model,
):
    if dash.callback_context.triggered[0]["prop_id"] == ".":
        raise PreventUpdate
    if "average" in average_filters:
        average_filters = True
    else:
        average_filters = False
    if scale_to != "None":
        scale_to = spec_model.virtual_filter_mapping[scale_to]
    if "r-star" in r_star:
        r_star = True
    else:
        r_star = False
    cset(
        "main_graph_filter_df",
        filter_df_from_queryset(
            spec_model.objects.all(),
            average_filters=average_filters,
            scale_to=scale_to,
            r_star=r_star,
        ),
    )
    if scale_to != "None":
        scale_to_string = "_".join(scale_to)
    else:
        scale_to_string = scale_to
    cset("scale_to", scale_to_string)
    cset("average_filters", average_filters)
    if not scale_trigger_count:
        return 1
    return scale_trigger_count + 1


def make_sibling_set(observations):
    """
    returns a set of tuples, each containing the child spectra of one
    observation.
    for pairing spectra with their siblings without hitting the database
    repeatedly.
    """
    return set(
        [tuple([spec.id for spec in obs.spectra()]) for obs in observations]
    )


def spectrum_queryset_siblings(queryset):
    return make_sibling_set(set([spec.observation for spec in queryset]))


def update_search_ids(
    _search_n_clicks,
    _load_trigger_index,
    fields,
    terms,
    quant_search_entries,
    search_trigger_dummy_value,
    *,
    cget,
    cset,
    spec_model,
):
    """
    updates the spectra displayed in the graph view.
    """
    # don't do anything on page load
    if not_triggered():
        raise PreventUpdate
    # or if a blank request is issued
    if not (fields and (terms or quant_search_entries)):
        raise PreventUpdate
    # construct a list of search parameters from the filled inputs
    # (ideally totally non-filled inputs would also be rejected by
    # handle_graph_search)
    entries = [
        parse_model_quant_entry(entry) for entry in quant_search_entries
    ]
    search_list = [
        {"field": field, "term": term, **entry}
        for field, term, entry in zip(fields, terms, entries)
        if not_blank(field) and (not_blank(term) or not_blank(entry))
    ]

    # if the search parameters have changed or if it's a new load, make a
    # new id list and trigger graph update using copy.deepcopy here to
    # avoid passing doubly-ingested input back after the check although on
    # the other hand it should be memoized -- but still but yes seriously it
    # should be memoized
    metadata_df = cget("metadata_df")
    search = handle_graph_search(
        metadata_df, deepcopy(search_list), spec_model
    )
    ctx = dash.callback_context
    if (
        set(search) != set(cget("main_search_ids"))
        or "load-trigger" in ctx.triggered[0]["prop_id"]
    ):
        cset("main_search_ids", search)
        # save search parameters for graph description
        cset("search_parameters", search_list)
        return search_trigger_dummy_value + 1
    raise PreventUpdate


def add_dropdown(children, spec_model, cget, cset):
    """
    adds another dropdown for search constraints.
    """
    # check with the cache in order to pick an index
    # this is because resetting the layout for tabs destroys n_clicks
    # could parse the page instead but i think this is better
    index = cget("search_parameter_index")
    if not index:
        index = 1
    else:
        index = index + 1

    searchable_fields = spec_model.searchable_fields
    children.append(search_parameter_div(index, searchable_fields, None))
    cset("search_parameter_index", index)
    return children


def remove_dropdown(index, children, _cget, _cset):
    """
    remove a search constraint dropdown, and its attendant constraints on
    searches.
    """
    param_to_remove = None
    for param in children:
        if param["props"]["id"]["index"] == index:
            param_to_remove = param
    if not param_to_remove:
        raise ValueError("Got the wrong param-to-remove index from somewhere.")

    new_children = deepcopy(children)
    new_children.remove(param_to_remove)
    return new_children


def clear_search(cset, spec_model):
    cset("search_parameter_index", 0)
    searchable_fields = spec_model.searchable_fields
    return [search_parameter_div(0, searchable_fields, None)]


def control_search_dropdowns(
    _add_clicks,
    _clear_clicks,
    _remove_clicks,
    children,
    search_trigger_clicks,
    *,
    spec_model,
    cget,
    cset,
):
    """
    dispatch function for building a new list of search dropdown components
    based on requests to add or remove components.
    returns the new list and an incremented value to trigger update_queryset
    """
    if not_triggered():
        raise PreventUpdate
    if search_trigger_clicks is None:
        search_trigger_clicks = 0
    if triggered_by("add-param"):
        drops = add_dropdown(children, spec_model, cget, cset)
    elif triggered_by("remove-param"):
        ctx = dash.callback_context
        index = trigger_index(ctx)
        drops = remove_dropdown(index, children, cget, cset)
    elif triggered_by("clear-search"):
        drops = clear_search(cset, spec_model)
    else:
        raise PreventUpdate
    return drops, search_trigger_clicks + 1


# ## individual-spectrum display functions


def spectrum_from_graph_event(
    event_data: dict, spec_model: "Model"
) -> "Model":
    """
    dcc.Graph event data (e.g. hoverData), plotter.Spectrum class ->
    plotter.Spectrum instance
    this function assumes it's getting data from a browser event that
    highlights
    a single graphed point, like clicking it or hovering on it, and returns
    the associated Spectrum object.
    """
    # the graph's customdata property should contain numbers corresponding
    # to database pks of spectra of associated points.
    return djget(
        spec_model, event_data["points"][0]["customdata"], "id", "get"
    )


# TODO: inefficient -- but this may be irrelevant?
def graph_point_to_metadata(event_data, *, spec_model, style=None):
    if style is None:
        style = {"margin": 0, "fontSize": 14}
    # parses hoverdata, clickdata, etc from main graph
    # into html <p> elements containing metadata of associated Spectrum
    if not event_data:
        raise PreventUpdate
    meta_print = dict_to_paragraphs(
        spectrum_from_graph_event(event_data, spec_model).metadata_dict(),
        style=style,
        ordering=["name", "sol", "seq_id", "feature"],
    )
    return meta_print


# TODO: probably inefficient
def update_spectrum_graph(
    event_data,
    scale_to,
    r_star,
    average_input_value,
    error_bar_value,
    *,
    spec_model,
    spec_graph_function,
):
    if scale_to != "None":
        scale_to = spec_model.virtual_filter_mapping[scale_to]
    average_filters = True if average_input_value == ["average"] else False
    show_error = True if error_bar_value == ["error"] else False
    if not event_data:
        raise PreventUpdate
    spectrum = spectrum_from_graph_event(event_data, spec_model)
    return spec_graph_function(
        spectrum,
        scale_to=scale_to,
        average_filters=average_filters,
        r_star=r_star,
        show_error=show_error,
    )


def make_mspec_browse_image_components(
    mspec, image_directory, static_image_url
):
    """
    MSpec object, size factor (viewport units), image directory ->
    pair of dash html.Img components containing the spectrum-reduced
    images associated with that object, pathed to the static image
    route defined in the live app instance
    """
    file_info = mspec.overlay_browse_file_info(image_directory)
    image_div_children = []
    for eye in ["left", "right"]:
        try:
            # size = file_info[eye + "_size"]
            filename = static_image_url + file_info[eye + "_file"]
        except KeyError:
            # size = (480, 480)
            filename = static_image_url + "missing.jpg"
        # aspect_ratio = size[0] / size[1]
        # width = base_size * aspect_ratio
        # height = base_size / aspect_ratio
        image_div_children.append(
            html.Img(
                src=filename,
                style={"width": "50%", "height": "50%"},
                id="spec-image-" + eye,
            )
        )
        component = html.Div(
            children=image_div_children,
        )
    return component


# TODO: assess whether this hack remains in
def make_zspec_browse_image_components(
    zspec, image_directory, static_image_url
):
    """
    ZSpec object, size factor (viewport units), image directory ->
    pair of dash html.Img components containing the rgb and enhanced
    images associated with that object, pathed to the static image
    route defined in the live app instance -- silly hack rn
    """
    file_info = zspec.overlay_browse_file_info(image_directory)
    image_div_children = []
    for image_type in ["rgb_image", "enhanced_image"]:
        try:
            # size = file_info[eye + "_size"]
            filename = static_image_url + file_info[image_type + "_file"]
        except KeyError:
            # size = (480, 480)
            filename = static_image_url + "missing.jpg"
        # aspect_ratio = size[0] / size[1]
        # width = base_size * aspect_ratio
        # height = base_size / aspect_ratio
        image_div_children.append(
            html.Img(
                src=filename,
                style={"width": "50%", "height": "50%"},
                id="spec-image-" + image_type,
            )
        )
        component = html.Div(
            children=image_div_children,
        )
    return component


def update_spectrum_images(
    event_data, *, spec_model, image_directory, static_image_url
):
    """
    just a callback-responsive wrapper to make_mspec_browse_image_components --
    probably we should actually put that function on the model
    """
    if not event_data:
        raise PreventUpdate
    # type checking just can't handle django class inheritance
    # noinspection PyTypeChecker
    spectrum = spectrum_from_graph_event(event_data, spec_model)
    # TODO: turn this into a dispatch function, if this ends up actually
    #  wanting distinct behavior
    if spec_model == ZSpec:
        return make_zspec_browse_image_components(
            spectrum, image_directory, static_image_url
        )
    return make_mspec_browse_image_components(
        spectrum, image_directory, static_image_url
    )


class SPlot:
    """
    class that holds a queryset of spectra with search parameters
    and axis relationships. probably also eventually visual settings.
    used for saving and restoring plots.
    """

    def __init__(self, arg_dict):
        # maybe eventually we define this more flexibly but better to be
        # strict for now
        self.y_axis = None
        self.x_axis = None
        for parameter in self.canonical_parameters:
            setattr(self, parameter, arg_dict[parameter])

    def axes(self):
        return self.x_axis, self.y_axis

    def graph(self):
        return self.graph_function(
            self.x_axis,
            self.y_axis,
            self.marker_properties,
            self.text,
            self.customdata,
        )

    def settings(self):
        return {
            parameter: getattr(self, parameter)
            for parameter in self.setting_parameters
        }

    canonical_parameters = (
        "x_axis",
        "y_axis",
        "marker_properties",
        "text",
        # customdata is typically a list of the pks of the spectra in axis
        # order
        "customdata",
        "main_search_ids",
        "search_parameters",
        "x_settings",
        "y_settings",
        "marker_settings",
        "graph_function",
        "main_highlight_parameters",
        "main_highlight_ids",
        "scale_to",
        "average_filters",
    )

    setting_parameters = (
        "search_parameters",
        "main_highlight_parameters",
        "x_settings",
        "y_settings",
        "marker_settings",
    )


def describe_current_graph(cget):
    """
    note this this relies on cached 'globals' from recalculate_graph
    and update_queryset! if this turns out to be an excessively ugly flow
    control solution, we could instead turn it into a callback that dynamically
    monitors state of the same objects they monitor the state of...but that
    parallel structure seems worse.
    """
    return {
        parameter: cget(parameter) for parameter in SPlot.canonical_parameters
    }


def open_viewer_tab(tabs, cget, cset):
    graph_params = describe_current_graph(cget)
    # don't open a tab on page load or if the graph is deformed
    if not graph_params["x_axis"]:
        raise PreventUpdate
    splot = SPlot(describe_current_graph(cget))
    # what viewers are open? check cache.
    graph_viewers = cget("open_graph_viewers")
    if graph_viewers:
        index = max(graph_viewers) + 1
        graph_viewers.append(index)
    else:
        index = 1
        graph_viewers = [1]
    # make a new tab at the first available index with the current graph
    new_tab = viewer_tab(index, splot)

    new_tabs = deepcopy(tabs)
    new_tabs.append(new_tab)
    # let cache know you opened a new viewer.
    graph_viewers.append(index)
    cset("open_graph_viewers", graph_viewers)
    return new_tabs, "viewer_tab_" + str(index)


def close_viewer_tab(index, tabs, cget, cset):
    # parse the frankly unfortunate dash input structure for the index of
    # the tab whose button this is
    tab_to_close = None
    for tab in tabs:
        if not isinstance(tab["props"]["id"], dict):
            continue
        if tab["props"]["id"]["type"] != "viewer-tab":
            continue
        if tab["props"]["id"]["index"] == index:
            tab_to_close = tab
    if not tab_to_close:
        raise ValueError("Got the wrong tab-to-close index from somewhere.")

    new_tabs = deepcopy(tabs)
    new_tabs.remove(tab_to_close)
    # let cache know you closed this in case anyone asks
    graph_viewers = cget("open_graph_viewers")
    graph_viewers.remove(index)
    cset("open_graph_viewers", graph_viewers)
    return new_tabs, "main_search_tab"


def save_search_tab_state(
    _n_clicks,
    save_name,
    trigger_value,
    cget,
    filename="saves/saved_searches.csv",
):
    """
    fairly permissive right now. this saves current search-tab state to a
    file as csv, which can then be reloaded by make_loaded_search_tab.
    """

    if not_triggered():
        raise PreventUpdate

    try:
        state_variable_names = [
            *cget("x_settings").keys(),
            *cget("y_settings").keys(),
            *cget("marker_settings").keys(),
            "search_parameters",
            "main_highlight_parameters",
            "scale_to",
            "average_filters",
        ]

        state_variable_values = [
            *cget("x_settings").values(),
            *cget("y_settings").values(),
            *cget("marker_settings").values(),
            str(cget("search_parameters")),
            str(cget("main_highlight_parameters")),
            str(cget("scale_to")),
            cget("average_filters"),
        ]
    except AttributeError as error:
        print(error)
        raise PreventUpdate
    try:
        saved_searches = pd.read_csv(filename)
    except FileNotFoundError:
        saved_searches = pd.DataFrame(columns=state_variable_names + ["name"])
    state_line = pd.DataFrame(
        {
            parameter: value
            for parameter, value in zip(
                state_variable_names, state_variable_values
            )
        },
        index=[0],
    )
    if save_name is None:
        save_name = dt.datetime.now().strftime("%D %H:%M:%S")
    state_line["name"] = save_name
    appended_df = pd.concat([saved_searches, state_line], axis=0)
    appended_df.to_csv(filename, index=False)
    return trigger_value + 1


def populate_saved_search_drop(*_triggers, search_file):
    try:
        options = [
            {"label": row["name"], "value": row_index}
            for row_index, row in enumerate(rows(pd.read_csv(search_file)))
        ]
    except FileNotFoundError:
        options = []
    return options


def make_loaded_search_tab(row, spec_model, search_file, cset):
    """makes a search tab with preset values from a saved search."""
    saved_searches = pd.read_csv(search_file)
    row_dict = rows(saved_searches)[row].to_dict()
    # TODO: doing this here might mean something is wrong in control flow
    cset("main_highlight_parameters", row_dict["main_highlight_parameters"])
    return search_tab(spec_model, row_dict)


def load_saved_search(tabs, row, spec_model, search_file, cset):
    """loads a search tab and replaces existing search tab with it"""
    new_tab = make_loaded_search_tab(row, spec_model, search_file, cset)
    if len(tabs) > 1:
        old_tabs = deepcopy(tabs[1:])
        new_tabs = [new_tab] + old_tabs
    else:
        new_tabs = [new_tab]
    return new_tabs, "main_search_tab"


def control_tabs(
    _n_clicks_open,
    _n_clicks_close,
    _n_clicks_load,
    tabs,
    load_row,
    load_trigger_index,
    *,
    spec_model,
    search_file,
    cget,
    cset,
):
    """
    control loading, opening, and closing viewer tabs.

    TODO, maybe: add ability to just open selectedData from main-graph
    """
    if not_triggered():
        raise PreventUpdate
    ctx = dash.callback_context
    trigger = ctx.triggered[0]["prop_id"]
    if "viewer-open-button" in trigger:
        return *open_viewer_tab(tabs, cget, cset), load_trigger_index
    if "tab-close-button" in trigger:
        index = trigger_index(ctx)
        new_tabs, active_tab_value = close_viewer_tab(index, tabs, cget, cset)
        return (
            new_tabs,
            active_tab_value,
            load_trigger_index,
        )
    if "load-search-load-button" in trigger:
        # don't try anything if nothing in the saved search dropdown is
        # selected
        if load_row is None:
            raise PreventUpdate
        # explicitly trigger graph recalculation call in update_queryset
        if not load_trigger_index:
            load_trigger_index = 0
        load_trigger_index = load_trigger_index + 1
        new_tabs, active_tab_value = load_saved_search(
            tabs, load_row, spec_model, search_file, cset
        )
        # this cache parameter is for semi-asynchronous flow control of the
        # load process without having to literally
        # have one dispatch callback function for the entire app
        cset("load_state", {"update_search_options": True})
        return new_tabs, active_tab_value, load_trigger_index


def pretty_print_search_params(search_parameters):
    string_list = []
    if not search_parameters:
        return ""
    for param in search_parameters:
        if "begin" in param.keys() or "end" in param.keys():
            string_list.append(
                param["field"]
                + ": "
                + str(param["begin"])
                + " -- "
                + str(param["end"])
            )
        else:
            term_list = param["term"]
            if len(term_list) > 1:
                term_string = ", ".join(term_list)
            else:
                term_string = term_list[0]
            string_list.append(param["field"] + ": " + term_string)
    if len(string_list) > 1:
        return "; ".join(string_list)
    return string_list[0]


def handle_main_highlight_save(
    _load_trigger, _save_button, trigger_value, *, cget, cset, spec_model
):
    if not_triggered():
        raise PreventUpdate
    ctx = dash.callback_context
    if "load-trigger" in str(ctx.triggered):
        # main highlight parameters are currently restored
        # in make_loaded_search_tab()
        metadata_df = cget("metadata_df")
        params = literal_eval(cget("main_highlight_parameters"))
        if params:
            cset(
                "main_highlight_ids",
                handle_graph_search(metadata_df, params, spec_model),
            )
        else:
            cset("main_highlight_ids", metadata_df.index)
        cset("main_highlight_parameters", params)
    else:
        highlight_ids = cget("main_highlight_ids")
        search_ids = cget("main_search_ids")
        if highlight_ids == search_ids:
            raise PreventUpdate
        cset("main_highlight_ids", search_ids)
        cset("main_highlight_parameters", cget("search_parameters"))
    return (
        "saved highlight: "
        + pretty_print_search_params(cget("main_highlight_parameters")),
        trigger_value + 1,
    )


def print_selected(selected):
    print(selected)
    return 0


# TODO: This should be reading from something in mcam_spect_data_conversion probably


def export_graph_csv(_clicks, selected, *, cget):
    if not_triggered():
        raise PreventUpdate
    metadata_df = cget("metadata_df").copy()
    filter_df = cget("main_graph_filter_df").copy()
    if selected is not None:
        search_ids = [point["customdata"] for point in selected["points"]]
    else:
        search_ids = cget("main_search_ids")
    filter_df.columns = [column.upper() for column in filter_df.columns]
    metadata_df.columns = [column.upper() for column in metadata_df.columns]
    metadata_df = metadata_df.reindex(
        columns=[
            "SOL",
            "SEQ_ID",
            "INSTRUMENT",
            "COLOR",
            "FEATURE",
            "FORMATION",
            "MEMBER",
            "FLOAT",
            "LTST",
            "ROVER_ELEVATION",
            "TARGET_ELEVATION",
            "INCIDENCE_ANGLE",
            "PHASE_ANGLE",
            "EMISSION_ANGLE",
            "TAU",
            "FOCAL_DISTANCE",
            "LAT",
            "LON",
            "ODOMETRY",
        ],
    )
    output_df = (
        pd.concat(
            [metadata_df, filter_df],
            axis=1,
        )
        .loc[search_ids]
        .sort_values(by="SEQ_ID")
    )
    filename = (
        cget("spec_model_name")
        + "_"
        + dt.datetime.now().strftime("%Y%M%dT%H%M%S")
        + "_"
        + re.sub(
            "[-;,:. ]+",
            "_",
            pretty_print_search_params(cget("search_parameters")),
        )
    )
    if cget("scale_to") != "None":
        filename += "_scaled_to_" + cget("scale_to")
    if cget("average_filters"):
        filename += "_near_filters_averaged"
    if selected is not None:
        filename += "_custom_selection"
    filename += ".csv"
    output_df.to_csv("exports/" + filename, index=None)
    return 1
