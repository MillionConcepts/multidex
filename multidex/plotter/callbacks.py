"""
these functions are partially defined and/or passed to callback
decorators in order to generate flow control within a dash app.
"""
import datetime as dt
import os
import re

from ast import literal_eval
from copy import deepcopy
from typing import Tuple

import dash
import pandas as pd
from dash.exceptions import PreventUpdate
from plotly import graph_objects as go

from multidex_utils import triggered_by, trigger_index, dict_to_paragraphs, pickctx, keygrab, field_values, not_blank, rows
from plotter.components import parse_model_quant_entry
from plotter.graph import load_values_into_search_div, add_dropdown, \
    remove_dropdown, clear_search, truncate_id_list_for_missing_properties, \
    make_axis, make_marker_properties, make_graph_display_settings, \
    spectrum_values_range, handle_graph_search, \
    make_zspec_browse_image_components, make_mspec_browse_image_components, \
    pretty_print_search_params, spectrum_from_graph_event, style_toggle
from plotter.spectrum_ops import filter_df_from_queryset


def handle_load(
    _n_clicks_load,
    load_row,
    load_trigger_index,
    *,
    spec_model,
    search_file,
    cset,
):
    """
    top-level handler for saved search loading process.
    """
    ctx = dash.callback_context
    trigger = ctx.triggered[0]["prop_id"]
    if "load-search-load-button" in trigger:
        # don't try anything if nothing in the saved search dropdown is
        # selected
        if load_row is None:
            raise PreventUpdate
        # explicitly trigger graph recalculation call in update_queryset
        if not load_trigger_index:
            load_trigger_index = 0
        load_trigger_index = load_trigger_index + 1
        loaded_div = load_values_into_search_div(
            load_row, spec_model, search_file, cset
        )
        # this cache parameter is for semi-asynchronous flow control of the
        # load process without having to literally
        # have one dispatch callback function for the entire app
        cset("load_state", {"update_search_options": True})
        return loaded_div, load_trigger_index


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
    if not event_data:
        raise PreventUpdate
    spectrum = spectrum_from_graph_event(event_data, spec_model)
    return spec_graph_function(
        spectrum,
        scale_to=scale_to,
        average_filters=average_filters,
        r_star=r_star,
        show_error=error_bar_value,
    )


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
    if scale_to not in ["None", None]:
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
    cset("r_star", r_star)
    if not scale_trigger_count:
        return 1
    return scale_trigger_count + 1


def update_main_graph(
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
    # handle explicit bounds changes
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
        return graph, {}

    # handle label addition / removal
    label_ids = cget("main_label_ids")
    if ctx.triggered[0]["prop_id"] == "main-graph.clickData":
        clicked_id = ctx.triggered[0]["value"]["points"][0]["customdata"]
        if clicked_id in label_ids:
            label_ids.remove(clicked_id)
        else:
            label_ids.append(clicked_id)
        cset("main_label_ids", label_ids)
    # TODO: performance increase is possible here by just returning the graph

    x_settings = pickctx(ctx, x_inputs)
    y_settings = pickctx(ctx, y_inputs)
    marker_settings = pickctx(ctx, marker_inputs)
    graph_display_settings = pickctx(ctx, graph_display_inputs)
    if isinstance(ctx.triggered[0]["prop_id"], dict):
        if ctx.triggered[0]["prop_id"]["type"] == "highlight-trigger":
            if marker_settings["main-highlight-toggle.value"] == "off":
                raise PreventUpdate
    search_ids = cget("search_ids")
    highlight_ids = cget("highlight_ids")
    filter_df = cget("main_graph_filter_df")
    metadata_df = cget("metadata_df")
    get_errors = ctx.inputs["main-graph-error.value"]
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

    graph_layout = ctx.states["main-graph.figure"]["layout"]

    # automatically reset graph zoom only if we're loading the page or
    # changing options and therefore scales
    if (
        ctx.triggered[0]["prop_id"]
        in ["main-graph-option-y.value", "main-graph-option-x.value", "."]
    ) or (len(ctx.triggered) > 1):
        zoom = None
    else:
        zoom = (graph_layout["xaxis"]["range"], graph_layout["yaxis"]["range"])

    # for functions that (perhaps asynchronously) fetch the state of the
    # graph.
    # this is another perhaps ugly flow control thing!
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
            "graph_display_dict",
            "axis_display_dict",
            "zoom",
            "x_errors",
            "y_errors",
            "x_title",
            "y_title",
            "get_errors",
        ):
            cset(parameter, locals()[parameter])

    # TODO: refactor (too many arguments)
    return (
        graph_function(
            x_axis,
            y_axis,
            marker_properties,
            graph_display_dict,
            axis_display_dict,
            text,
            customdata,
            label_ids,
            zoom,
            x_errors,
            y_errors,
            x_title,
            y_title,
        ),
        {},
    )


def update_search_options(
    field, _load_trigger_index, current_quant_search, *, cget, spec_model
):
    """
    populate term values and parameter range as appropriate when different
    fields are selected in the search interface
    currently this does _not_ cascade according to selected terms. this may
    or may not be desirable
    """
    if not field:
        raise PreventUpdate
    is_loading = (
        "search-load-trigger" in dash.callback_context.triggered[0]["prop_id"]
    )
    props = keygrab(spec_model.searchable_fields(), "label", field)
    search_df = pd.concat([cget("main_graph_filter_df"), cget("metadata_df")])
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
            "minimum/maximum: " + str(spectrum_values_range(search_df, field))
            # TODO: this should be a hover-over tooltip eventually
            + """ e.g., '100--200' or '100, 105, 110'""",
            search_text,
        ]

    # otherwise, populate the term interface and reset the range display and
    # searches
    return [field_values(search_df, field), "", ""]


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
    # don't do anything if a blank request is issued
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
    search_df = pd.concat(
        [cget("metadata_df"), cget("main_graph_filter_df")], axis=1
    )
    search = handle_graph_search(search_df, deepcopy(search_list), spec_model)
    ctx = dash.callback_context
    if (
        set(search) != set(cget("search_ids"))
        or "load-trigger" in ctx.triggered[0]["prop_id"]
    ):
        cset("search_ids", search)
        # save search parameters for graph description
        cset("search_parameters", search_list)
        return search_trigger_dummy_value + 1
    raise PreventUpdate


def change_calc_input_visibility(calc_type, *, spec_model):
    """
    turn visibility of filter + component dropdowns on and off in response to
    changes in arity / type of requested calc
    """
    props = keygrab(spec_model.graphable_properties(), "value", calc_type)
    # 'methods' are specifically those methods of spec_model
    # that take its filters as arguments
    if props["type"] == "method":
        return [
            {"display": "flex", "flexDirection": "column"}
            if x < props["arity"]
            else {"display": "none"}
            for x in range(3)
        ] + [{"display": "none"}]
    elif props["type"] == "decomposition":
        return [{"display": "none"} for _ in range(3)] + [
            {"display": "flex", "flexDirection": "column"}
        ]
    return [{"display": "none"} for _ in range(4)]


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
        keygrab(spec_model.searchable_fields(), "label", field)["value_type"]
        == "quant"
    ):
        return [{"display": "none"}, {}]
    return [{}, {"display": "none"}]


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
    if spec_model.instrument == "ZCAM":
        return make_zspec_browse_image_components(
            spectrum, image_directory, static_image_url
        )
    return make_mspec_browse_image_components(
        spectrum, image_directory, static_image_url
    )


def populate_saved_search_drop(*_triggers, search_file):
    try:
        options = [
            {"label": row["name"], "value": row_index}
            for row_index, row in enumerate(rows(pd.read_csv(search_file)))
        ]
    except FileNotFoundError:
        options = []
    return options


def handle_main_highlight_save(
    _load_trigger, _save_button, trigger_value, *, cget, cset, spec_model
):
    ctx = dash.callback_context
    if "load-trigger" in str(ctx.triggered):
        # main highlight parameters are currently restored
        # in make_loaded_search_tab()
        metadata_df = cget("metadata_df")
        filter_df = cget("main-graph-filter-df")
        params = cget("main_highlight_parameters")
        if params is not None:
            params = literal_eval(params)
            search_df = pd.concat([metadata_df, filter_df], axis=1)
            cset(
                "highlight_ids",
                handle_graph_search(search_df, params, spec_model),
            )
        else:
            cset("highlight_ids", metadata_df.index)
        cset("main_highlight_parameters", params)
    else:
        highlight_ids = cget("highlight_ids")
        search_ids = cget("search_ids")
        if highlight_ids == search_ids:
            raise PreventUpdate
        cset("highlight_ids", search_ids)
        cset("main_highlight_parameters", cget("search_parameters"))
    return (
        "saved highlight: "
        + pretty_print_search_params(cget("main_highlight_parameters")),
        trigger_value + 1,
    )


def toggle_panel_visibility(_click, panel_style, arrow_style, text_style):
    """
    switches collapsible panel between visible and invisible,
    and rotates and sets text on its associated arrow.
    """
    panel_style = style_toggle(panel_style, states=("none", "revert"))
    arrow_style = style_toggle(
        arrow_style, "WebkitTransform", ("rotate(45deg)", "rotate(-45deg)")
    )
    arrow_style = style_toggle(
        arrow_style, "transform", ("rotate(45deg)", "rotate(-45deg)")
    )
    text_style = style_toggle(text_style, states=("inline-block", "none"))
    return panel_style, arrow_style, text_style


def toggle_color_drop_visibility(type_selection: str) -> Tuple[dict, dict]:
    """
    just show or hide scale/solid color dropdowns per coloring type radio
    button selection. very unsophisticated for now.
    """
    if type_selection == "solid":
        return {"display": "none"}, {"display": "block"}
    return {"display": "block"}, {"display": "none"}


# TODO: This should be reading from something in marslab.compat probably
def export_graph_csv(_clicks, selected, *, cget):
    metadata_df = cget("metadata_df").copy()
    filter_df = cget("main_graph_filter_df").copy()
    if selected is not None:
        search_ids = [point["customdata"] for point in selected["points"]]
    else:
        search_ids = cget("search_ids")
    filter_df.columns = [column.upper() for column in filter_df.columns]
    metadata_df.columns = [column.upper() for column in metadata_df.columns]
    metadata_df = metadata_df.reindex(
        columns=[
            "NAME",
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
        + dt.datetime.now().strftime("%Y%m%dT%H%M%S")
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
    os.makedirs("exports", exist_ok=True)
    output_df.to_csv("exports/" + filename, index=None)
    return 1