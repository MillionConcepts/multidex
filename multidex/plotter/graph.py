"""
functions used for selecting, manipulating, and drawing spectral data
within plotly-dash objects. this module is _separate_ from app structure
definition_ and, to the extent possible, components. these are lower-level
functions used by interface functions in callbacks.py
"""
import csv
import datetime as dt
from ast import literal_eval
from collections.abc import Iterable
from copy import deepcopy
from functools import reduce
from itertools import chain, cycle
from operator import or_
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence

import numpy as np
import pandas as pd
from dash import html
from dash.exceptions import PreventUpdate
from dustgoggles.pivot import split_on
from plotly import graph_objects as go
from toolz import keyfilter

from multidex_utils import (
    keygrab,
    not_blank,
    seconds_since_beginning_of_day,
    arbitrarily_hash_strings,
    none_to_quote_unquote_none,
    df_multiple_field_search,
    re_get,
    djget,
    insert_wavelengths_into_text,
)
from plotter import spectrum_ops
from plotter.colors import get_palette_from_scale_name, plotly_colorscale_type
from plotter.components.ui_components import (
    search_parameter_div,
)
from plotter.layout import search_div
from plotter.reduction import (
    default_multidex_pipeline,
    transform_and_explain_variance,
)
from plotter.styles.graph_style import COLORBAR_SETTINGS
from plotter.types import SpectrumModel, SpectrumModelInstance

if TYPE_CHECKING:
    from plotter.models import ZSpec, MSpec


# ### cache functions ###

# many of the other functions in this module take outputs of these two
# functions as arguments. returning a function that calls a specific cache
# backend defined in-app allows us to share data between defined clusters of
# dash objects. the specific cache is in some sense a set of pointers that
# serves as a namespace. this, rather than a global variable or variables, is
# used because Flask does not guarantee thread safety of globals.


def cache_set(cache) -> Callable[[str, Any], bool]:
    def cset(key: str, value: Any) -> bool:
        return cache.set(key, value)

    return cset


def cache_get(cache) -> Callable[[str], Any]:
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
    filters_are_averaged: bool,
):
    metadata_args = []
    filt_args = []
    indices = []
    for suffix in input_suffixes:
        axis_option = re_get(settings, "graph-option-" + suffix)
        model_property = keygrab(
            spec_model.graphable_properties(), "value", axis_option
        )
        if model_property["type"] == "decomposition":
            # assuming here for now all decompositions require all filters
            if filters_are_averaged is True:
                filters = list(spec_model.canonical_averaged_filters.keys())
            else:
                filters = list(spec_model.filters.keys())
            filt_args.append(filters)
        elif model_property["type"] == "method":
            # we assume here that 'methods' all take a spectrum's filter names
            # as arguments, and have arguments in an order corresponding to the
            # inputs.
            filt_args.append(
                [
                    re_get(settings, f"filter-{ix}-{suffix}")
                    for ix in range(1, model_property["arity"] + 1)
                ]
            )
        elif model_property["type"] == "computed":
            filt_args.append([axis_option])
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


def deframe(df_or_series):
    if isinstance(df_or_series, pd.DataFrame):
        assert len(df_or_series.columns == 1)
        return df_or_series.iloc[:, 0]
    return df_or_series


def perform_decomposition(id_list, filter_df, settings, props):
    # TODO: this is fairly inefficient and recomputes the decomposition
    #  every time any axis is changed. it might be better to cache this.
    #  at the moment, the performance concerns probably don't really matter,
    #  though; PCA on these sets is < 250ms per and generally much less.
    queryset_df = filter_df.loc[id_list]
    # TODO: temporary hack -- don't do PCA on tiny sets
    if len(queryset_df.index) < 8:
        raise ValueError("Won't do PCA on tiny sets.")
    # drop errors
    queryset_df = queryset_df[
        [c for c in queryset_df.columns if "err" not in c]
    ]
    component_ix = re_get(settings, "component")
    # TODO, maybe: placeholder for other decomposition methods
    # method = props["method"]
    pipeline = default_multidex_pipeline()
    transform, explained_variances = transform_and_explain_variance(
        queryset_df, pipeline
    )
    component = list(transform.iloc[:, component_ix].values)
    explained_variance = explained_variances.iloc[component_ix]
    title = "{}{} {}%".format(
        props["value"],
        str(component_ix + 1),
        str(round(explained_variance * 100, 2)),
    )
    return component, None, title


def perform_spectrum_op(
    id_list, spec_model, filter_df, settings, props, get_errors=False
):
    # we assume here that 'methods' all take a spectrum's filter names
    # as arguments, and have arguments in an order corresponding to
    # the inputs. also drop precalculated perperties -- a bit kludgey.
    queryset_df = (
        filter_df.loc[id_list].drop(["filter_avg", "err_avg"], axis=1).copy()
    )
    filt_args = [
        re_get(settings, "filter-" + str(ix))
        for ix in range(1, props["arity"] + 1)
    ]
    spectrum_op = getattr(spectrum_ops, props["value"])
    base_title = props["value"] + " " + str(" ".join(filt_args))
    # TODO, unfortunately: this probably needs more fiddly rules
    title = insert_wavelengths_into_text(base_title, spec_model)
    if get_errors == "none":
        get_errors = False
    try:
        if get_errors == "instrumental":
            vals, errors = spectrum_ops.compute_minmax_spec_error(
                queryset_df, spec_model, spectrum_op, *filt_args
            )
            if errors is not None:
                return (
                    list(deframe(vals).values),
                    {
                        "symmetric": False,
                        "arrayminus": list(
                            np.abs(np.array(deframe(errors[0])))
                        ),
                        "array": list(np.abs(np.array(deframe(errors[1])))),
                    },
                    title,
                )
            return list(deframe(vals).values), None, title
        vals, errors = spectrum_op(
            queryset_df, spec_model, *filt_args, get_errors
        )
        if get_errors and errors is not None:
            return (
                list(deframe(vals).values),
                {"array": list(deframe(errors).values)},
                title,
            )
        return list(deframe(vals).values), None, title
    except ValueError:  # usually representing intermediate input states
        raise PreventUpdate


def make_axis(
    settings: dict,
    id_list: Iterable,
    spec_model: Any,
    filter_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    get_errors: bool,
    filters_are_averaged,
    _color_clip,
) -> tuple[list[float], Optional[list[float]], str]:
    """
    make an axis for one of our graphs by looking at the appropriate rows from
    our big precalculated metadata / data dataframes; data has already been
    scaled and averaged as desired. expects a list that has been
    processed by truncate_id_list_for_missing_properties
    """
    axis_option, props = get_axis_option_props(settings, spec_model)
    if props["type"] == "decomposition":
        if filters_are_averaged is True:
            decomp_df = filter_df[
                list(spec_model.canonical_averaged_filters.keys())
            ]
        else:
            decomp_df = filter_df[list(spec_model.filters.keys())]
        return perform_decomposition(id_list, decomp_df, settings, props)

    if props["type"] == "computed":
        return filter_df.loc[id_list, props["value"]].values, None, axis_option

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
    return value_series.values, None, axis_option


def get_axis_option_props(settings, spec_model):
    # what is requested function or property?
    axis_option = re_get(settings, "graph-option-")
    # what are the characteristics of that function or property?
    props = keygrab(spec_model.graphable_properties(), "value", axis_option)
    return axis_option, props


# TODO: this is sloppy but cleanup would be better after everything's
#  implemented...probably...it would really be better to do this in components
#  but is difficult because you have to instantiate the colorbar somewhere
#  it would also be better to style with CSS but it seems like plotly
#  really wants to put element-level style declarations on graph ticks and
#  it is an unusually large hassle to get inside its svg rendering loop
def make_markers(
    settings,
    id_list,
    spec_model,
    filter_df,
    metadata_df,
    _get_errors,
    _filters_are_averaged,
    color_clip,
):
    """
    this expects an id list that has already
    been processed by truncate_id_list_for_missing_properties
    """
    marker_option, props = get_axis_option_props(settings, spec_model)

    if props["type"] == "decomposition":
        property_list, _, title = perform_decomposition(
            id_list, filter_df, settings, props
        )
    elif props["type"] == "computed":
        property_list, title = (
            filter_df.loc[id_list, props["value"]].values,
            props["value"],
        )
    elif props["type"] == "method":
        property_list, _, title = perform_spectrum_op(
            id_list, spec_model, filter_df, settings, props
        )
    else:
        property_list, title = (
            metadata_df.loc[id_list][props["value"]].values,
            props["value"],
        )
    palette_type = plotly_colorscale_type(
        re_get(settings, "palette-name-drop.value")
    )
    if palette_type is None:
        # solid color case
        color = re_get(settings, "palette-name-drop.value")
        colorbar = None
        colormap = None
    else:
        colorbar_dict = COLORBAR_SETTINGS.copy() | {"title_text": title}
        colormap = re_get(settings, "palette-name-drop.value")
        if props["value_type"] == "qual":
            string_hash = arbitrarily_hash_strings(
                none_to_quote_unquote_none(property_list)
            )
            # TODO: compute this one step later so that we can avoid
            #  including entirely-highlighted things in colorbar
            color = [string_hash[prop] for prop in property_list]
            colorbar_dict |= {
                "tickvals": list(string_hash.values()),
                "ticktext": list(string_hash.keys()),
            }
            if palette_type == "qualitative":
                # only do this for "qualitative" scales to trick plotly...
                # plotly's tricks for discretizing "continuous" scales are
                # better than mine
                colormap = get_palette_from_scale_name(
                    colormap, len(string_hash), qualitative=True
                )
        else:
            if len(property_list) > 0:
                if isinstance(property_list[0], dt.time):
                    property_list = list(
                        map(seconds_since_beginning_of_day, property_list)
                    )
            color = property_list
            if color_clip not in ([], [0, 100]):
                color = np.clip(color, *np.percentile(color, color_clip))
        # colorbar = go.scatter.marker.ColorBar(**colorbar_dict)
        colorbar = go.layout.coloraxis.ColorBar(**colorbar_dict)

    # set marker size and, if present, outline.
    # plotly demands that marker size be defined as a sequence
    # in order to be able to set marker outlines -- however, this also causes
    # markers to be drawn in a different (and more expensive) way -- hence
    # this silly-looking logic
    size = re_get(settings, "marker-size-radio.value")
    if re_get(settings, "outline-radio.value") == "off":
        outline = {}
    else:
        size = [size for _ in id_list]
        outline = {
            "color": re_get(settings, "outline-radio.value"),
            "width": 5,
        }

    # set marker symbol
    symbol = re_get(settings, "marker-symbol-drop.value")
    coloraxis = {"colorscale": colormap}
    marker_property_dict = {
        "marker": {
            "size": size,
            "opacity": 1,
            "symbol": symbol,
            "coloraxis": "coloraxis1",
        },
        "line": outline,
    }
    # colorbar = None causes plotly to draw undesirable fake ticks
    if colorbar is not None:
        # marker_property_dict["marker"]["colorbar"] = colorbar
        coloraxis["colorbar"] = colorbar
    return marker_property_dict, color, coloraxis, props["value_type"]


def format_display_settings(settings):
    settings_dict = {}
    axis_settings_dict = {}
    if re_get(settings, "graph-bg"):
        settings_dict["plot_bgcolor"] = re_get(settings, "graph-bg")
    if re_get(settings, "gridlines") == "off":
        axis_settings_dict["showgrid"] = False
    else:
        axis_settings_dict["showgrid"] = True
        axis_settings_dict["gridcolor"] = re_get(settings, "gridlines")
    return settings_dict, axis_settings_dict


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


def spectrum_values_range(metadata_df, field):
    """
    returns minimum and maximum values of property within id list of spectra.
    for cueing or aiding searches.
    """
    values = metadata_df[field]
    return values.min(), values.max()


def non_blank_search_parameters(parameters):
    entry_keys = ["term", "begin", "end", "value_list"]
    return [
        parameter
        for parameter in parameters
        if reduce(or_, [not_blank(parameter.get(key)) for key in entry_keys])
    ]


def handle_graph_search(search_df, parameters, spec_model):
    """
    dispatcher / manager for user-issued searches within the graph interface.
    fills fields from model definition and feeds resultant list to a general-
    purpose search function.
    """
    # nothing here? great
    if not parameters:
        return list(search_df.index)
    # add value_type and type information to dictionaries (based on
    # search properties defined in the model)
    for parameter in parameters:
        field = parameter.get("field")
        if field:
            props = keygrab(spec_model.searchable_fields(), "label", field)
            parameter["value_type"] = props["value_type"]
    # toss out blank strings, etc. -- they do not restrict the search
    parameters = non_blank_search_parameters(parameters)
    # do we have any actual constraints? if not, return the entire data set
    if not parameters:
        return list(search_df.index)
    # otherwise, actually perform a search
    return df_multiple_field_search(search_df, parameters)


def add_dropdown(children, spec_model, cget, cset):
    """
    adds another dropdown for search constraints.
    """
    # check with the cache in order to pick an index
    # this is because resetting the layout on load destroys n_clicks
    # could parse the page instead but i think this is better
    index = cget("search_parameter_index")
    if not index:
        index = 1
    else:
        index = index + 1

    searchable_fields = spec_model.searchable_fields()
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
    searchable_fields = spec_model.searchable_fields()
    return [search_parameter_div(0, searchable_fields, None)]


# TODO: inefficient -- but this may be irrelevant?
def make_mspec_browse_image_components(mspec: "MSpec", static_image_url):
    """
    MSpec object, size factor (viewport units), image directory ->
    pair of dash html.Img components containing the spectrum-reduced
    images associated with that object, pathed to the assets image
    route defined in the live app instance
    """
    image_div_children = []
    images = literal_eval(mspec.images)
    for eye in ["left", "right"]:
        try:
            eye_images = keyfilter(lambda key: eye in key, images)
            assert len(eye_images) >= 1
            filename = static_image_url + list(eye_images.values())[0]
        except AssertionError:
            filename = static_image_url + "missing.jpg"
        image_div_children.append(
            html.Img(
                src=filename,
                style={
                    "maxWidth": "55%",
                    "maxHeight": "55%",
                },
                id="spec-image-" + eye,
            )
        )

    return html.Div(children=image_div_children)


# TODO: assess whether this hack remains in, assess goodness of display in
#  new layout
def make_zspec_browse_image_components(
    zspec: "ZSpec", image_directory, static_image_url
):
    """
    ZSpec object, size factor (viewport units), image directory ->
    pair of dash html.Img components containing the rgb and enhanced
    images associated with that object, pathed to the assets image
    route defined in the live app instance -- silly hack rn
    """
    file_info = zspec.overlay_browse_file_info(image_directory)
    filename = None
    for eye in ["left", "right"]:
        try:
            filename = static_image_url + file_info[eye + "_file"]
        except KeyError:
            continue
    if filename is None:
        filename = static_image_url + "missing.jpg"
    # in this case we're just aggressively setting the appropriate
    # aspect ratio. this is probably not always a good idea.
    return html.Div(
        children=[
            html.Img(
                src=filename,
                style={"maxWidth": "100%", "maxHeight": "100%"},
                id="spec-image-" + eye,
            )
        ],
    )


def load_values_into_search_div(search_file, spec_model, cset):
    """makes a search tab with preset values from a saved search."""
    with open(search_file) as save_csv:
        search = next(csv.DictReader(save_csv))
        search = {k: literal_eval(v) for k, v in search.items()}
    # TODO: somewhat bad smell, might mean something is wrong in control flow
    if search["highlight_parameters"] is not None:
        cset("highlight_parameters", search["highlight_parameters"])
    if search["search_parameters"] is not None:
        cset("search_parameter_index", len(search["search_parameters"]))
    else:
        cset("search_parameter_index", 0)
    return search_div(spec_model, search)


def pretty_print_search_params(search_parameters):
    string_list = []
    if not search_parameters:
        return ""
    for param in search_parameters:
        if "begin" in param.keys() or "end" in param.keys():
            string_list.append(
                f"{param['field']}: {param.get('begin')} -- {param.get('end')}"
            )
        else:
            if param.get("term"):
                term_list = param["term"]
            else:
                term_list = param["value_list"]
            term_list = [
                str(term) if not str(term).endswith(".0") else str(term)[:-2]
                for term in term_list
            ]
            if len(term_list) > 1:
                term_string = ", ".join([str(term) for term in term_list])
            else:
                term_string = str(term_list[0])
            string_list.append(f"{param['field']}: {term_string}")
    if len(string_list) > 1:
        return "; ".join(string_list)
    return string_list[0]


def spectrum_from_graph_event(
    event_data: dict, spec_model: SpectrumModel
) -> SpectrumModelInstance:
    """
    dcc.Graph event data (e.g. hoverData), plotter.Spectrum class ->
    plotter.Spectrum instance
    this function assumes it's getting data from a browser event that
    highlights  a single graphed point (like clicking it or hovering on it),
    and returns the associated Spectrum object.
    """
    # the graph's customdata property should contain numbers corresponding
    # to database pks of spectra of associated points.
    return djget(
        spec_model, event_data["points"][0]["customdata"], "id", "get"
    )


def make_scatter_annotations(
    metadata_df: pd.DataFrame, truncated_ids: Sequence[int]
) -> np.ndarray:
    meta = metadata_df.loc[truncated_ids]
    descriptor = meta["feature"].copy()
    no_feature_ix = descriptor.loc[descriptor.isna()].index
    descriptor.loc[no_feature_ix] = meta["color"].loc[no_feature_ix]
    text = (
        "sol" + meta["sol"].astype(str) + " " + meta["name"] + " " + descriptor
    ).values
    return text


def retrieve_graph_data(
    cget: Callable[[str], Any]
) -> tuple[pd.DataFrame, pd.DataFrame, Sequence[int]]:
    search_ids = cget("search_ids")
    data_df = cget("data_df")
    metadata_df = cget("metadata_df")
    return data_df, metadata_df, search_ids


def halt_for_ineffective_highlight_toggle(ctx, highlight_settings):
    if isinstance(ctx.triggered[0]["prop_id"], dict):
        if ctx.triggered[0]["prop_id"]["type"] == "highlight-trigger":
            if highlight_settings["highlight-toggle.value"] == "off":
                raise PreventUpdate


def halt_to_debounce_palette_update(trigger, marker_settings, cget):
    if trigger != "palette-name-drop.value":
        return
    if (
        cget("marker_settings")["palette-name-drop.value"]
        == marker_settings["palette-name-drop.value"]
    ):
        raise PreventUpdate


def save_palette_memory(marker_settings, cget, cset):
    palette = re_get(marker_settings, "palette-name-drop.value")
    palette_type = plotly_colorscale_type(palette)
    palette_memory = cget("palette_memory")
    palette_memory[palette_type] = palette
    cset("palette_memory", palette_memory)


def halt_for_inappropriate_palette_type(marker_settings, spec_model):
    """
    this condition should only occur during transition from a qual to quant
    marker option when a qual map was selected. it may be a sloppy way to stop
    it.
    """
    palette_type = plotly_colorscale_type(
        re_get(marker_settings, "palette-name-drop.value")
    )
    if palette_type != "qualitative":
        return
    _, props = get_axis_option_props(marker_settings, spec_model)
    if props["value_type"] == "quant":
        raise PreventUpdate


def add_or_remove_label(cset, ctx, label_ids):
    """
    label point based on click. this adds the point id to the set of labels;
    it does not perform label rendering.
    """
    clicked_id = ctx.triggered[0]["value"]["points"][0]["customdata"]
    if clicked_id in label_ids:
        label_ids.remove(clicked_id)
    else:
        label_ids.append(clicked_id)
    cset("label_ids", label_ids)


def parse_main_graph_bounds_string(ctx):
    bounds_string = ctx.inputs["main-graph-bounds.value"]
    try:
        bounds_string = [
            float(bound) for bound in filter(None, bounds_string.split(" "))
        ]
        assert len(bounds_string) == 4
        return bounds_string
    except (ValueError, AssertionError, AttributeError):
        return None


def update_zoom_from_bounds_string(graph, bounds_string):
    graph.update_layout(
        {
            "xaxis": {
                "range": [bounds_string[0], bounds_string[1]],
                "autorange": False,
            },
            "yaxis": {
                "range": [bounds_string[2], bounds_string[3]],
                "autorange": False,
            },
        }
    )
    return graph


def explicitly_set_graph_bounds(ctx):
    """change graph bounds based on input to the 'set bounds' field"""
    bounds_string = parse_main_graph_bounds_string(ctx)
    if bounds_string is None:
        raise PreventUpdate
    graph = go.Figure(ctx.states["main-graph.figure"])
    update_zoom_from_bounds_string(graph, bounds_string)
    return graph, {}


def assemble_highlight_marker_dict(highlight_settings, base_marker_size):
    highlight_marker_dict = {}
    # iterate over values of all highlight UI elements, interpreting them as
    # marker values legible to go.Scatter & its relatives
    for prop, setting_input in zip(
        ("color", "size", "symbol"),
        ("color-drop", "size-radio", "symbol-drop"),
    ):
        setting = highlight_settings[f"highlight-{setting_input}.value"]
        if setting == "none":
            continue
        if prop == "size":
            # highlight size increase is relative, not absolute;
            # base_marker_size can be either int or list[int] --
            # need to retain its type to retain how outline works
            if isinstance(base_marker_size, list):
                setting = [setting * value for value in base_marker_size]
            else:
                setting = setting * base_marker_size
        highlight_marker_dict[prop] = setting
    return highlight_marker_dict


def branch_highlight_df(
    graph_df, highlight_ids, highlight_settings, base_marker_size
) -> tuple[pd.DataFrame, Optional[pd.DataFrame], dict]:
    """
    an element of the update_main_graph() flow. if highlight is active,
    split highlighted points out of main df and interpret styles as specified,
    to be drawn as a separate trace.
    """
    # don't do anything if there is nothing to do
    if highlight_settings["highlight-toggle.value"] == "off" or (
        len(highlight_ids) == 0
    ):
        return graph_df, None, {}
    highlight_df, graph_df = split_on(
        graph_df, graph_df["customdata"].isin(highlight_ids)
    )
    highlight_marker_dict = assemble_highlight_marker_dict(
        highlight_settings, base_marker_size
    )
    return graph_df, highlight_df, highlight_marker_dict
