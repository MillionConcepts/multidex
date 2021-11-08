"""
these functions are partially defined and/or passed to callback
decorators in order to generate flow control within a dash app.
"""
import datetime as dt
import json
import os

from ast import literal_eval
from copy import deepcopy
from itertools import cycle
from pathlib import Path
from typing import Tuple

import dash
import numpy as np
import pandas as pd
from dash.exceptions import PreventUpdate

from multidex_utils import (
    triggered_by,
    trigger_index,
    dict_to_paragraphs,
    pickctx,
    keygrab,
    field_values,
    not_blank,
)
from plotter.render_output.output_writer import save_main_scatter_plot
from plotter.components.ui_components import parse_model_quant_entry
from plotter.components.graph_components import (
    main_scatter_graph,
    spectrum_line_graph,
    failed_scatter_graph,
)
from plotter.graph import (
    load_values_into_search_div,
    add_dropdown,
    remove_dropdown,
    clear_search,
    truncate_id_list_for_missing_properties,
    make_axis,
    make_marker_properties,
    format_display_settings,
    spectrum_values_range,
    handle_graph_search,
    make_zspec_browse_image_components,
    make_mspec_browse_image_components,
    pretty_print_search_params,
    spectrum_from_graph_event,
    style_toggle,
    make_scatter_annotations,
    retrieve_graph_data,
    halt_for_ineffective_highlight_toggle,
    add_or_remove_label,
    explicitly_set_graph_bounds,
    parse_main_graph_bounds_string, get_axis_option_props,
)
from plotter.spectrum_ops import data_df_from_queryset


def trigger_search_update(_load_trigger, search_trigger):
    return search_trigger + 1


def handle_load(
    _n_clicks_load,
    selected_file,
    load_trigger_index,
    default_settings_checked,
    *,
    spec_model,
    search_path,
    cset,
):
    """
    top-level handler for saved search loading process.
    """
    ctx = dash.callback_context
    trigger = ctx.triggered[0]["prop_id"]
    if trigger == '.':
        if default_settings_checked is True:
            raise PreventUpdate
        try:
            selected_file = next(
                str(search) for search in Path(search_path).iterdir()
                if search.stem.lower() == 'default'
            )
        except (StopIteration, FileNotFoundError):
            raise PreventUpdate
    # don't try anything if nothing in the saved search dropdown is
    # selected / we're not loading
    elif selected_file is None:
        raise PreventUpdate
    # explicitly trigger graph recalculation call in update_queryset
    if not load_trigger_index:
        load_trigger_index = 0
    load_trigger_index = load_trigger_index + 1
    loaded_div = load_values_into_search_div(selected_file, spec_model, cset)
    # this cache parameter is for semi-asynchronous flow control of the
    # load process without having to literally
    # have one dispatch callback function for the entire app
    cset("load_state", {"update_search_options": True})
    return loaded_div, load_trigger_index, True


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
):
    if scale_to != "None":
        scale_to = spec_model.virtual_filter_mapping[scale_to]
    average_filters = True if average_input_value == ["average"] else False
    if not event_data:
        raise PreventUpdate
    spectrum = spectrum_from_graph_event(event_data, spec_model)
    return spectrum_line_graph(
        spectrum,
        scale_to=scale_to,
        average_filters=average_filters,
        r_star=r_star,
        show_error=error_bar_value,
    )


def update_data_df(
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
        "data_df",
        data_df_from_queryset(
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
    cget,
    cset,
    spec_model,
    record_settings=True,
):
    ctx = dash.callback_context
    trigger = ctx.triggered[0]["prop_id"]
    # handle explicit bounds changes
    if trigger == "main-graph-bounds.value":
        return explicitly_set_graph_bounds(ctx)
    bounds_string = parse_main_graph_bounds_string(ctx)
    try:
        color_clip = [
            ctx.inputs["color-clip-bound-low.value"],
            ctx.inputs["color-clip-bound-high.value"]
        ]
        if len(color_clip) != 2:
            raise ValueError("need two numerals in this field to do things")
        if color_clip[0] > color_clip[1]:
            raise ValueError("refusing to clip backwards")
    except (AttributeError, TypeError, ValueError):
        # don't change anything if they're in the middle of messing around
        # with the color clip
        if "color-clip-bound" in trigger:
            raise PreventUpdate
        # but if they have changed some other input with an invalid color clip
        # string in the dialog, render as if there were no color clip at all.
        color_clip = []
    # handle label addition / removal
    label_ids = cget("label_ids")
    # TODO: performance increase is possible here by just returning the graph
    if trigger == "main-graph.clickData":
        add_or_remove_label(cset, ctx, label_ids)
    elif trigger == "clear-labels.n_clicks":
        cset("label_ids", [])
        label_ids = []
    x_settings = pickctx(ctx, x_inputs)
    y_settings = pickctx(ctx, y_inputs)
    marker_settings = pickctx(ctx, marker_inputs)
    halt_for_ineffective_highlight_toggle(ctx, marker_settings)

    graph_display_dict, axis_display_dict = format_display_settings(
        pickctx(ctx, graph_display_inputs)
    )

    data_df, metadata_df, highlight_ids, search_ids = retrieve_graph_data(cget)
    if not search_ids:
        return (
            failed_scatter_graph(
                "no spectra match search parameters", graph_display_dict
            ),
            {},
        )
    for settings in x_settings, y_settings, marker_settings:
        axis_option, props = get_axis_option_props(settings, spec_model)
        if (props["type"] == "decomposition") and len(search_ids) <= 8:
            return (
                failed_scatter_graph(
                    "too few spectra for PCA", graph_display_dict
                ),
                {},
            )
    get_errors = ctx.inputs["main-graph-error.value"]
    filters_are_averaged = "average" in ctx.states["main-graph-average.value"]


    truncated_ids = truncate_id_list_for_missing_properties(
        x_settings | y_settings | marker_settings,
        search_ids,
        ["x.value", "y.value", "marker.value"],
        data_df,
        metadata_df,
        spec_model,
        filters_are_averaged,
    )
    if not truncated_ids:
        return (
            failed_scatter_graph(
                "matching spectra lack requested properties",
                graph_display_dict
            ),
            {},
        )
    graph_content = [
        truncated_ids,
        spec_model,
        data_df,
        metadata_df,
        get_errors,
        filters_are_averaged,
        highlight_ids,
        color_clip
    ]
    graph_df = pd.DataFrame({"customdata": truncated_ids})
    # storing these separately because the API for error bars is annoying
    errors = {}
    graph_df["x"], errors["x"], x_title = make_axis(x_settings, *graph_content)
    graph_df["y"], errors["y"], y_title = make_axis(y_settings, *graph_content)
    # similarly for marker properties
    marker_properties, marker_axis_type = make_marker_properties(
        marker_settings, *graph_content
    )

    # storing this to set draw order for highlights
    graph_df["size"] = marker_properties["marker"]["size"]
    graph_df["text"] = make_scatter_annotations(metadata_df, truncated_ids)

    # avoid resetting zoom for labels, color changes, etc.
    # TODO: continue assessing these conditions
    if ("marker" in trigger) or ("click" in trigger):
        layout = ctx.states["main-graph.figure"]["layout"]
        zoom = (layout["xaxis"]["range"], layout["yaxis"]["range"])
    else:
        zoom = None
    # for functions that (perhaps asynchronously) fetch the state of the
    # graph. this is another perhaps ugly flow control thing!
    if record_settings:
        for parameter in (
            "x_settings",
            "y_settings",
            "marker_settings",
            "marker_properties",
            "graph_display_dict",
            "axis_display_dict",
            "x_title",
            "y_title",
            "get_errors",
            "bounds_string",
        ):
            cset(parameter, locals()[parameter])

    return (
        main_scatter_graph(
            graph_df,
            errors,
            marker_properties,
            graph_display_dict,
            axis_display_dict,
            label_ids,
            x_title,
            y_title,
            zoom,
            marker_axis_type
            # bounds_string, # TODO: are we passing this ever or no?
        ),
        {},
    )


def update_search_options(
    field, _load_trigger_index, current_search, *, cget, spec_model
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
    search_df = pd.concat([cget("data_df"), cget("metadata_df")], axis=1)
    # if it's a field we do number interval searches on, reset term
    # interface and show number ranges in the range display. but don't reset
    # the number entries if we're in the middle of a load!
    if props["value_type"] == "quant":
        if is_loading:
            search_text = current_search
        else:
            search_text = ""
        return [
            [{"label": "any", "value": "any"}],
            "min/max: " + str(spectrum_values_range(search_df, field))
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
    try:
        entries = [
            parse_model_quant_entry(entry) for entry in quant_search_entries
        ]
    except ValueError:
        raise PreventUpdate
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
    search_df = pd.concat([cget("metadata_df"), cget("data_df")], axis=1)
    search = handle_graph_search(search_df, deepcopy(search_list), spec_model)
    # TODO: remove this cruft if in fact it remains so
    # ctx = dash.callback_context
    # if (
    #     set(search) != set(cget("search_ids"))
    #     or "load-trigger" in ctx.triggered[0]["prop_id"]
    # ):
    cset("search_ids", search)
    # save search parameters for graph description
    cset("search_parameters", search_list)
    return search_trigger_dummy_value + 1
    # raise PreventUpdate


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


def populate_saved_search_drop(*_triggers, search_path):
    try:
        options = [
            {"label": file.stem, "value": str(file)}
            for file in Path(search_path).iterdir()
            if file.name.endswith("csv")
        ]
    except FileNotFoundError:
        options = []
    return options


def handle_highlight_save(
    _load_trigger, _save_button, trigger_value, *, cget, cset, spec_model
):
    ctx = dash.callback_context
    if "load-trigger" in str(ctx.triggered):
        # main highlight parameters are currently restored
        # in make_loaded_search_tab()
        search_df = pd.concat([cget("metadata_df"), cget("data_df")], axis=1)
        params = cget("highlight_parameters")
        if params is not None:
            params = literal_eval(str(params))
            cset(
                "highlight_ids",
                handle_graph_search(search_df, params, spec_model),
            )
        else:
            cset("highlight_ids", search_df.index.to_list())
        cset("highlight_parameters", params)
    else:
        highlight_ids = cget("highlight_ids")
        search_ids = cget("search_ids")
        if np.all(highlight_ids == search_ids):
            raise PreventUpdate
        cset("highlight_ids", search_ids)
        cset("highlight_parameters", cget("search_parameters"))
    return (
        "saved highlight: "
        + pretty_print_search_params(cget("highlight_parameters")),
        trigger_value + 1,
    )


def toggle_panel_visibility(
    _click, _reset_click, panel_style, arrow_style, text_style
):
    """
    switches collapsible panel between visible and invisible,
    and rotates and sets text on its associated arrow.
    """
    ctx = dash.callback_context
    if ctx.triggered[0]["prop_id"] == "collapse-all.n_clicks":
        panel_style = {"display": "none"}
        arrow_style = {"WebkitTransform": "rotate(45deg)"}
        text_style = {"display": "inline-block"}
    else:
        panel_style = style_toggle(panel_style, states=("none", "revert"))
        arrow_style = style_toggle(
            arrow_style, "WebkitTransform", ("rotate(45deg)", "rotate(-45deg)")
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
def export_graph_csv(_clicks, selected, *, cget, spec_model):
    ctx = dash.callback_context
    if ctx.triggered[0]['value'] is None:
        raise PreventUpdate
    metadata_df = cget("metadata_df").copy()
    filter_df = cget("data_df").copy()
    if selected is not None:
        search_ids = [point["customdata"] for point in selected["points"]]
    else:
        search_ids = cget("search_ids")
    filter_df.columns = [column.upper() for column in filter_df.columns]
    metadata_df.columns = [column.upper() for column in metadata_df.columns]
    # TODO: dumb hack, make this end-to-end smoother
    if "SOIL_LOCATION" in metadata_df.columns:
        metadata_df = metadata_df.rename(
            columns={"SOIL_LOCATION": "SOIL LOCATION"}
        )
    if cget("r_star") is True:
        metadata_df['UNITS'] = 'R*'
    else:
        metadata_df['UNITS'] = 'IOF'
    output_df = (
        pd.concat(
            [metadata_df, filter_df],
            axis=1,
        )
        .loc[search_ids]
        .sort_values(by="SEQ_ID")
    )
    filename = dt.datetime.now().strftime("%Y%m%dT%H%M%S") + ".csv"
    output_path = Path("exports", "csv", spec_model.instrument.lower())
    os.makedirs(output_path, exist_ok=True)
    output_df.to_csv(Path(output_path, filename), index=None)
    # todo: huh?
    return 1


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


def save_search_state(
    _n_clicks, save_name, trigger_value, *, search_path, cget
):
    """
    fairly permissive right now. this saves current search-tab state to a
    file as csv, which can then be reloaded by make_loaded_search_tab.
    """
    if save_name is None:
        raise PreventUpdate
    state_variable_names = [
        *cget("x_settings").keys(),
        *cget("y_settings").keys(),
        *cget("marker_settings").keys(),
        *cget("graph_display_dict").keys(),
        *cget("axis_display_dict").keys(),
        "search_parameters",
        "highlight_parameters",
        "scale_to",
        "average_filters",
        "r_star",
        "errors",
    ]
    state_variable_values = [
        *cget("x_settings").values(),
        *cget("y_settings").values(),
        *cget("marker_settings").values(),
        *cget("graph_display_dict").values(),
        *cget("axis_display_dict").values(),
        str(cget("search_parameters")),
        str(cget("highlight_parameters")),
        str(cget("scale_to")),
        cget("average_filters"),
        cget("r_star"),
        cget("errors"),
    ]
    state_line = pd.DataFrame(
        {
            parameter: value
            for parameter, value in zip(
                state_variable_names, state_variable_values
            )
        },
        index=[0],
    )
    state_line["name"] = save_name
    save_name = save_name + ".csv"
    os.makedirs(search_path, exist_ok=True)
    state_line.to_csv(Path(search_path, save_name), index=False)
    return trigger_value + 1


def export_graph_png(clientside_fig_info, fig_dict, *, spec_model):
    # this condition occurs during saved search loading. search loading
    #  triggers a call from the clientside js snippet that triggers image
    #  export. this is an inelegant way to suppress that call.
    if not fig_dict.get("data"):
        raise PreventUpdate
    info = json.loads(clientside_fig_info)
    aspect = info["width"] / info["height"]
    save_main_scatter_plot(fig_dict, aspect, spec_model.instrument.lower())
