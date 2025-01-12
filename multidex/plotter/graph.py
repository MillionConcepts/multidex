"""
functions used for selecting, manipulating, and drawing spectral data
within plotly-dash objects. this module is _separate_ from app structure
definition_ and, to the extent possible, components. these are lower-level
functions used by interface functions in callbacks.py
"""
from _testcapi import INT_MAX
from ast import literal_eval
from collections.abc import Iterable
from copy import deepcopy
import csv
import datetime as dt
from functools import reduce
from itertools import chain, cycle
from operator import or_
from typing import Any, Callable, Optional, Sequence, TYPE_CHECKING

from dash import html
from dash.exceptions import PreventUpdate
from dustgoggles.pivot import split_on
import numpy as np
import pandas as pd
from dustgoggles.structures import listify
from plotly import graph_objects as go

from multidex.multidex_utils import (
    keygrab,
    not_blank,
    seconds_since_beginning_of_day,
    hash_strings,
    none_to_quote_unquote_none,
    df_multiple_field_search,
    re_get,
    djget,
    insert_wavelengths_into_text,
    model_metadata_df,
    get_verbose_name,
)
from multidex.plotter import spectrum_ops
from multidex.plotter.colors import get_palette_from_scale_name, get_scale_type
from multidex.plotter.components.ui_components import (
    search_parameter_div,
)
from multidex.plotter.layout import primary_app_div
from multidex.plotter.models import INSTRUMENT_MODEL_MAPPING
from multidex.plotter.reduction import (
    default_multidex_pipeline,
    transform_and_explain,
)
from multidex.plotter.spectrum_ops import data_df_from_queryset
from multidex.plotter.config.graph_style import COLORBAR_SETTINGS
from multidex.plotter.types import SpectrumModel, SpectrumModelInstance
from multidex.plotter.components.graph_components import get_ordering

if TYPE_CHECKING:
    from multidex.plotter.models import ZSpec, MSpec


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
            # TODO: no longer a good assumption
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
        # allow null to be used as a category of color
        elif ("marker" in suffix) and (model_property["value_type"] == "qual"):
            continue
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


# TODO: this is fairly inefficient and recomputes the decomposition
#  every time any axis is changed. it might be better to cache this.
#  at the moment, the performance concerns probably don't really matter,
#  though; PCA on these sets is < 250ms per and generally much less.
def perform_decomposition(
    queryset_df, settings, props, spec_model, cset
):
    # TODO: temporary hack -- don't do PCA on tiny sets
    if len(queryset_df.index) < 8:
        raise ValueError("Won't do PCA on tiny sets.")
    # drop errors
    queryset_df = queryset_df[
        [c for c in queryset_df.columns if "std" not in c]
    ]
    component_ix = int(re_get(settings, "component"))
    # TODO, maybe: placeholder for other decomposition methods
    # method = props["method"]
    pipeline = default_multidex_pipeline()
    transform, eigenvectors, explained_variances = transform_and_explain(
        queryset_df, pipeline
    )
    eigenvectors = pd.DataFrame(eigenvectors)
    eigenvectors.columns = queryset_df.columns
    eigenvectors.index = explained_variances.index
    eigenvector_df = pd.concat([eigenvectors, explained_variances], axis=1)
    component = list(transform.iloc[:, component_ix].values)
    explained_variance = explained_variances.iloc[component_ix]
    title = "{}{} {}%".format(
        props["value"],
        str(component_ix + 1),
        str(round(explained_variance * 100, 2)),
    ).title()
    return component, title, eigenvector_df


def perform_spectrum_op(
    id_list, spec_model, filter_df, settings, props, get_errors=False
):
    # we assume here that 'methods' all take a spectrum's filter names
    # as arguments, and have arguments in an order corresponding to
    # the inputs. also drop precalculated perperties -- a bit kludgey.
    allowable = list(chain(*[(f, f"{f}_std") for f in spec_model.filters]))
    queryset_df = (
        filter_df.loc[id_list]
        .drop([c for c in filter_df if c not in allowable], axis=1)
        .copy()
    )
    filt_args = [
        re_get(settings, "filter-" + str(ix))
        for ix in range(1, props["arity"] + 1)
    ]
    spectrum_op = getattr(spectrum_ops, props["value"])
    base_title = props["value"] + " " + str(" ".join(filt_args))
    # TODO, unfortunately: this probably needs more fiddly rules
    title = insert_wavelengths_into_text(base_title.title(), spec_model)
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
    cset,
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
        return _decompose_for_axis(
            settings,
            cset,
            id_list,
            spec_model,
            filter_df,
            props,
            filters_are_averaged
        )
    if props["type"] == "computed":
        return filter_df.loc[id_list, props["value"]].values, None, axis_option

    # TODO, maybe: a hack
    if props["type"] == "non_filter_computed":
        return (
            metadata_df.loc[id_list][props["value"]].values,
            None,
            props["label"]
        )

    if props["type"] == "method":
        return perform_spectrum_op(
            id_list,
            spec_model,
            filter_df,
            settings,
            props,
            get_errors,
        )
    # qual case
    value_series = metadata_df.loc[id_list][
        props["value"]
    ].astype(str).str.title()
    return value_series.values, None, get_verbose_name(axis_option, spec_model)


def _decompose_for_axis(
    settings,
    cset,
    id_list,
    spec_model,
    filter_df,
    props,
    filters_are_averaged
):
    if filters_are_averaged is True:
        decomp_df = filter_df[
            list(spec_model.canonical_averaged_filters.keys())
        ]
    else:
        decomp_df = filter_df[list(spec_model.filters.keys())]
    # TODO: this is an operational-timeline hack to permit "cut the
    #  bayers" behavior. replace this with UI elements permitting
    #  decomposition op feature selection later.
    if "permissibly_explanatory_bandpasses" in dir(spec_model):
        queryset_df = decomp_df[
            spec_model.permissibly_explanatory_bandpasses(decomp_df.columns)
        ].loc[id_list]
    else:
        queryset_df = filter_df.loc[id_list]
    component, title, eigenvector_df = perform_decomposition(
        queryset_df, settings, props, spec_model, cset
    )
    cset("eigenvector_df", eigenvector_df)
    return component, None, title


def get_axis_option_props(settings, spec_model):
    # what is requested function or property?
    axis_option = re_get(settings, "graph-option-")
    # what are the characteristics of that function or property?
    props = keygrab(spec_model.graphable_properties(), "value", axis_option)
    return axis_option, props


def _maybeindex(x, seq):
    try:
        return -seq.index(x)
    except ValueError:
        return INT_MAX

# TODO: this is sloppy but cleanup would be better after everything's
#  implemented...probably...it would really be better to do this in components
#  but is difficult because you have to instantiate the colorbar somewhere
#  it would also be better to style with CSS but it seems like plotly
#  really wants to put element-level style declarations on graph ticks and
#  it is an unusually large hassle to get inside its svg rendering loop
def make_markers(
    settings,
    cset,
    id_list,
    spec_model,
    filter_df,
    metadata_df,
    _get_errors,
    filters_are_averaged,
    color_clip,
):
    """
    this expects an id list that has already been processed by
    truncate_id_list_for_missing_properties()
    """
    marker_option, props = get_axis_option_props(settings, spec_model)
    if props["type"] == "decomposition":
        property_list, _, title = _decompose_for_axis(
            settings,
            cset,
            id_list,
            spec_model,
            filter_df,
            props,
            filters_are_averaged
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
    # TODO, maybe: a hack
    elif props["type"] == "non_filter_computed":
        property_list, title = (
            metadata_df.loc[id_list][props["value"]].values,
            props["label"].title()
        )
    else:
        property_list, title = (
            metadata_df.loc[id_list][props["value"]].values,
            get_verbose_name(props["value"], spec_model).title(),
        )
    palette_type = get_scale_type(re_get(settings, "palette-name-drop.value"))
    if palette_type == "solid":
        # solid color case
        color = re_get(settings, "palette-name-drop.value")
        colorbar = None
        colormap = None
    else:
        colorbar_dict = COLORBAR_SETTINGS.copy() | {"title_text": title}
        colormap = re_get(settings, "palette-name-drop.value")
        if props["value_type"] == "qual":
            property_list = tuple(
                map(str.title, none_to_quote_unquote_none(property_list))
            )
            carr = get_ordering(
                props["value"], spec_model.instrument
            ).get("categoryarray")
            key = None if carr is None else lambda x: _maybeindex(x, carr)
            string_hash = hash_strings(property_list, key)
            # TODO: compute this one step later so that we can avoid
            #  including entirely-highlighted things in colorbar
            color = [string_hash[prop] for prop in property_list]
            colorbar_dict |= {
                "tickvals": list(string_hash.values()),
                "ticktext": list(map(str.title, string_hash.keys())),
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
            "width": 1,
        }

    # set marker symbol
    symbol = re_get(settings, "marker-symbol-drop.value")
    coloraxis = {"colorscale": colormap}
    opacity = settings['marker-opacity-input.value']
    if opacity is None:
        opacity = 100
    marker_property_dict ={
        "size": size,
        "opacity": opacity / 100,
        "symbol": symbol,
        "coloraxis": "coloraxis1",
        "line": outline,
    }
    # colorbar = None causes plotly to draw undesirable fake ticks
    if colorbar is not None:
        # marker_property_dict["colorbar"] = colorbar
        coloraxis["colorbar"] = colorbar
    return marker_property_dict, color, coloraxis, props["value_type"]


def format_display_settings(settings):
    settings_dict = {}
    axis_settings_dict = {}
    if re_get(settings, "graph-bg"):
        settings_dict["plot_bgcolor"] = re_get(settings, "graph-bg")
    if re_get(settings, "gridlines") == "#000000":
        axis_settings_dict["showgrid"] = False
    else:
        axis_settings_dict["showgrid"] = True
    axis_settings_dict["gridcolor"] = re_get(settings, "gridlines")
    if axis_settings_dict["showgrid"] is False:
        axis_settings_dict["zeroline"] = False
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


def spectrum_values_range(metadata_df, field, digits=2):
    """
    returns minimum and maximum values of property within id list of spectra.
    for cueing or aiding searches.
    """
    values = metadata_df[field].dropna()
    vstats= values.min(), values.max(), *np.quantile(values, (0.25, 0.75))
    if (values.round() == values).all():
        return tuple(map(lambda v: int(v), vstats))
    return tuple(map(lambda v: round(v, digits), vstats))


def non_blank_search_parameters(parameters):
    entry_keys = ["terms", "begin", "end", "free", "value_list"]
    # TODO: free semi-breaks this because it's no longer just an automatic
    #  distinction between quant and qual; bandaid fixes are in place
    #  downstream, but it would be better to actually fix it
    # TODO: is that TODO out of date?
    return [
        parameter
        for parameter in parameters
        if reduce(or_, [not_blank(parameter.get(key)) for key in entry_keys])
    ]


def handle_graph_search(
    search_df,
    tokens,
    parameters,
    logical_quantifier,
    spec_model,
) -> list[int]:
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
    # for AND or nothing for OR
    if not parameters:
        if logical_quantifier == "AND":
            return list(search_df.index)
        if logical_quantifier == "OR":
            return []
    # otherwise, actually perform a search
    return df_multiple_field_search(
        search_df, tokens, parameters, logical_quantifier
    )


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
    for side in ["left", "right"]:
        eye_image = images.get(side)
        if eye_image is None:
            eye_image = "missing.jpg"
        filename = static_image_url + eye_image
        image_div_children.append(
            html.Img(
                src=filename,
                style={"maxWidth": "70%", "maxHeight": "70%"},
                id=f"spec-image-{side}",
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
    dash html.Img component containing the natural-color image
    associated with that object, mapped to the assets image
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


#  new layout
def make_cspec_browse_image_components(
    cspec: "CSpec", image_directory, static_image_url
):
    """
    CSpec object, size factor (viewport units), image directory ->
    dash html.Img component containing the natural-color image
    associated with that object, mapped to the assets image
    route defined in the live app instance -- silly hack rn
    """
    image_div_children = []
    img = literal_eval(cspec.images)
    img = img[0]
    if img is None:
        img = "missing.jpg"
    filename = static_image_url + "/" + img
    image_div_children.append(
            html.Img(
                src=filename,
                style={"maxWidth": "100%", "maxHeight": "100%"},
                id="spec-image-cspec",
            )
        )
    return html.Div(children=image_div_children)


def load_state_into_application(search_file, spec_model, cget, cset):
    """loads saved application state into the primary application panel."""
    with open(search_file) as save_csv:
        settings = next(csv.DictReader(save_csv))
        settings = {k: literal_eval(v) for k, v in settings.items()}
    # TODO: all of this initialization special-case stuff has a somewhat
    #  bad smell, might mean something is fundamentally wrong in control flow
    # TODO: should really rectify types across cache, component, and loaded
    #  values -- although maybe this _is_ the load -> cache conversion step,
    #  which should just be siloed and made explicit?
    average_filters = settings["average_filters"] == "True"
    r_star = settings["r_star"] == "True"
    cache_data_df(
        spec_model=spec_model,
        cset=cset,
        average_filters=average_filters,
        r_star=r_star,
        scale_to=settings["scale_to"],
    )
    if settings["highlight_parameters"] is not None:
        cset("highlight_parameters", settings["highlight_parameters"])
    if settings["search_parameters"] is not None:
        cset("search_parameter_index", len(settings["search_parameters"]))
    else:
        cset("search_parameter_index", 0)
    # TODO: don't i have an abstraction for this
    palette_memory = cget("palette_memory")
    palette = settings["palette-name-drop.value"]
    palette_type = get_scale_type(palette)
    palette_memory[palette_type] = palette
    cset("palette_memory", palette_memory)
    # TODO: this is attempting to fix a race condition with
    #  allow_qualitative_palettes. it's messy but _maybe_ necessary?
    cset("loading_palette_type", palette_type)
    return primary_app_div(spec_model, settings)


def pretty_print_search_params(parameters, logical_quantifier):
    string_list = []
    if not parameters:
        return ""
    for param in parameters:
        if param.get('is_free') is True:
            description = f"{param['field']} LIKE {param['free']}"
        elif "begin" in param.keys() or "end" in param.keys():
            description = (
                f"{param['field']} from "
                f"{param.get('begin')} to {param.get('end')}"
            )
        else:
            if param.get("terms"):
                term_list = param["terms"]
            else:
                term_list = param["value_list"]
            term_list = [
                str(term) if not str(term).endswith(".0") else str(term)[:-2]
                for term in term_list
            ]
            if len(term_list) > 1:
                term_string = (
                    f"in ({', '.join([str(term) for term in term_list])})"
                )
            else:
                term_string = f"== {term_list[0]}"
            description = f"{param['field']} {term_string}"
        if param.get("null") is True:
            description += " (NULL is True)"
        if param.get("invert") is True:
            description = "NOT " + description
        string_list.append(description)
    if len(string_list) > 1:
        return f" {logical_quantifier} ".join(string_list)
    return string_list[0]


def spectrum_from_graph_event(
    event_data: dict, spec_model: SpectrumModel
) -> SpectrumModelInstance:
    """
    dcc.Graph event data (e.g. hoverData), plotter.Spectrum class ->
    plotter.Spectrum instance
    this function assumes it's getting data from a browser event that
    highlights a single graphed point (like clicking it or hovering on it),
    and returns the associated Spectrum object.
    """
    # the graph's customdata property should contain numbers corresponding
    # to database pks of spectra of associated points.
    return djget(
        spec_model, event_data["points"][0]["customdata"], "id", "get"
    )


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
    palette_type = get_scale_type(palette)
    palette_memory = cget("palette_memory")
    palette_memory[palette_type] = palette
    cset("palette_memory", palette_memory)


def halt_for_inappropriate_palette_type(marker_settings, spec_model):
    """
    this condition should only occur during transition from a qual to quant
    marker option when a qual map was selected. it may be a sloppy way to stop
    it.
    """
    palette_type = get_scale_type(
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


def assemble_highlight_marker_dict(
    highlight_settings, base_marker_size, highlight_ids
):
    highlight_marker_dict = {}
    # iterate over values of all highlight UI elements, interpreting them as
    # marker values legible to go.Scatter & its relatives
    for prop, setting_input in zip(
        ("color", "size", "symbol", "outline", "opacity"),
        (
            "color-drop",
            "size-radio",
            "symbol-drop",
            "outline-radio",
            "opacity-input"
        ),
    ):
        setting = highlight_settings[f"highlight-{setting_input}.value"]
        if setting == "none":
            continue
        elif prop == "outline" and setting == "off":
            prop, setting = "line", {}
        elif prop == "outline":
            prop, setting = "line", {"color": setting, "width": 1}
        elif prop == "size":
            # highlight size increase is relative, not absolute, and
            # base_marker_size can be either int or list[int] -- will be
            # list[int] if outline for the primary graph is on and int if
            # outline is off. we need to follow the same typing rules here
            # depending on the highlight outline setting (or lack thereof).
            base_size = listify(base_marker_size)[0]
            hout = highlight_settings.get("highlight-outline-radio.value")
            if hout is None and isinstance(base_marker_size, int):
                setting = setting * base_size
            elif hout is None:
                setting = [setting * base_size for _ in highlight_ids]
            elif hout == "off":
                setting = setting * base_size
            else:
                setting = [setting * base_size for _ in highlight_ids]
        elif prop == "opacity":
            setting = 1 if setting is None else setting / 100
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
        highlight_settings, base_marker_size, highlight_ids
    )
    return graph_df, highlight_df, highlight_marker_dict


# TODO: does this actually go here?
def dump_model_table(
    spec_model_code: str,
    filename: str,
    r_star: bool = False,
    include_lab_spectra: bool = False,
    dict_function: Optional[Callable] = None,
):
    spec_model = INSTRUMENT_MODEL_MAPPING[spec_model_code]
    data = data_df_from_queryset(spec_model.objects.all(), r_star=r_star)
    metadata = model_metadata_df(spec_model, dict_function=dict_function)
    data.columns = [column.upper() for column in data.columns]
    metadata.columns = [column.upper() for column in metadata.columns]
    # TODO: dumb hack, make this end-to-end smoother
    if "SOIL_LOCATION" in metadata.columns:
        metadata = metadata.rename(columns={"SOIL_LOCATION": "SOIL LOCATION"})
    metadata["UNITS"] = "R*" if r_star is True else "IOF"
    output = pd.concat([metadata, data], axis=1).sort_values(by="SCLK")
    if include_lab_spectra is False:
        output = output.loc[output["FEATURE"] != "lab spectrum"]
    output.to_csv(filename, index=None)


def cache_data_df(average_filters, cset, r_star, scale_to, spec_model):
    cset(
        "data_df",
        data_df_from_queryset(
            spec_model.objects.all(),
            average_filters=average_filters,
            scale_to=scale_to,
            r_star=r_star,
        ),
    )
    if scale_to != "none":
        scale_to_string = "_".join(scale_to)
    else:
        scale_to_string = scale_to
    cset("scale_to", scale_to_string)
    cset("average_filters", average_filters)
    cset("r_star", r_star)
