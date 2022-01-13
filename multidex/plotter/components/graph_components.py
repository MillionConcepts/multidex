"""dash components that are graphs and associated helper functions"""

from copy import deepcopy
from typing import Mapping, Optional

import numpy as np
import pandas as pd
from plotly import graph_objects as go

from plotter.colors import discretize_color_representations
from plotter.spectrum_ops import d2r
from plotter.styles.graph_style import (
    ANNOTATION_SETTINGS,
    GRAPH_DISPLAY_DEFAULTS,
    AXIS_DISPLAY_DEFAULTS,
    SEARCH_FAILURE_MESSAGE_SETTINGS,
)


def style_data(
    fig,
    graph_display_settings,
    axis_display_settings=None,
    x_title=None,
    y_title=None,
    marker_axis_type=None,
    marker_property_dict=None,
    zoom=None,
):
    axis_display_dict = AXIS_DISPLAY_DEFAULTS | axis_display_settings
    # noinspection PyTypeChecker
    fig.update_xaxes(axis_display_dict | {"title_text": x_title, "categoryorder": "category ascending"})
    fig.update_yaxes(axis_display_dict | {"title_text": y_title, "categoryorder": "category ascending"})
    # fig.update_traces(**marker_property_dict)
    if (
        (marker_axis_type == "qual")
        # don't try to discretize solid colors
        and not isinstance(marker_property_dict["marker"]["color"], str)
    ):
        fig = discretize_color_representations(fig)
    apply_canvas_style(fig, graph_display_settings)
    if zoom is not None:
        fig.update_layout(
            {
                "xaxis": {"range": [zoom[0][0], zoom[0][1]]},
                "yaxis": {"range": [zoom[1][0], zoom[1][1]]},
            }
        )


def draw_errors_on_figure(fig, errors):
    for axis in ["x", "y"]:
        key = f"error_{axis}"
        if errors[axis] is None:
            value = {"visible": False}
        else:
            value = errors[axis] | {
                "visible": True,
                "color": "rgba(0,0,0,0.3)",
            }
        fig.update_traces({key: value})
    return fig


def apply_canvas_style(fig, graph_display_settings):
    display_dict = GRAPH_DISPLAY_DEFAULTS | graph_display_settings
    fig.update_layout(display_dict)


def main_scatter_graph(
    graph_df: pd.DataFrame,
    highlight_df: Optional[pd.DataFrame],
    errors: Mapping,
    marker_property_dict: Mapping,
    marker_axis_type: str,
    coloraxis: dict,
    highlight_marker_dict: Mapping,
    graph_display_settings: Mapping,
    axis_display_settings: Mapping,
    label_ids: list[int],
    x_title: str = None,
    y_title: str = None,
    zoom: Optional[tuple[list[float, float]]] = None,
) -> go.Figure:
    """
    main graph component. this function creates the Plotly figure; data
    and metadata are filtered and formatted in callbacks.update_main_graph().
    """
    # TODO: go.Scattergl (WebGL) is noticeably worse-looking than go.Scatter
    #  (SVG), but go.Scatter may be inadequately performant with all the
    #  points in the data set. can we optimize a bit? hard with plotly...

    # TODO: refactor to build layout dictionaries first rather than
    #  using the update_layout pattern, for speed

    fig = go.Figure()
    # the click-to-label annotations
    draw_floating_labels(fig, graph_df, label_ids)
    # and the scattered points and their error bars
    # last-mile thing here to keep separate from highlight -- TODO: silly?
    marker_property_dict["marker"]["color"] = graph_df["color"].values
    fig.update_coloraxes(coloraxis)
    fig.add_trace(
        go.Scattergl(
            x=graph_df["x"],
            y=graph_df["y"],
            hovertext=graph_df["text"],
            # suppresses trace hoverover
            hovertemplate="%{hovertext}<extra></extra>",
            customdata=graph_df["customdata"],
            mode="markers + text",
            # marker={"color": "black", "size": 8},
            showlegend=False,
            **marker_property_dict,
        )
    )

    if highlight_df is not None:
        draw_floating_labels(fig, highlight_df, label_ids)
        # TODO: ...why are we doing this here?
        full_marker_dict = dict(deepcopy(marker_property_dict))
        full_marker_dict["marker"] = (
            full_marker_dict["marker"] | highlight_marker_dict
        )
        if "color" not in highlight_marker_dict:
            # just treat it like the graph df
            full_marker_dict["marker"]["color"] = highlight_df["color"].values
        fig.add_trace(
            go.Scattergl(
                x=highlight_df["x"],
                y=highlight_df["y"],
                hovertext=highlight_df["text"],
                customdata=highlight_df["customdata"],
                # suppresses trace display in hoverover
                hovertemplate="%{hovertext}<extra></extra>",
                mode="markers + text",
                showlegend=False,
                **full_marker_dict,
            )
        )
    fig = draw_errors_on_figure(fig, errors)
    # last-step canvas stuff: set bounds, gridline color, titles, etc.
    style_data(
        fig,
        graph_display_settings,
        axis_display_settings,
        x_title,
        y_title,
        marker_axis_type,
        marker_property_dict,
        zoom,
    )
    return fig


def failed_scatter_graph(message: str, graph_display_settings: Mapping):
    fig = go.Figure()
    fig.add_annotation(text=message, **SEARCH_FAILURE_MESSAGE_SETTINGS)
    apply_canvas_style(fig, graph_display_settings)
    return fig


# we are using 'annotations' rather than 'text' for this,
# because plotly requires redraws to show the text, and fails
# to do so every other time for ... reasons ... so it (falsely) appears to do
# nothing every other time unless you pan or whatever
def draw_floating_labels(fig, graph_df, label_ids):
    """
    add floating labels for clicked points
    """
    for database_id, string, xpos, ypos in graph_df[
        ["customdata", "text", "x", "y"]
    ].values:
        if database_id in label_ids:
            fig.add_annotation(
                x=xpos, y=ypos, text=string, **ANNOTATION_SETTINGS
            )


def spectrum_line_graph(
    spectrum: "MSpec",
    scale_to=("l1", "r1"),
    average_filters=True,
    show_error=True,
    r_star=True,
) -> go.Figure:
    """
    placeholder line graph for individual mastcam spectra.
    creates a plotly figure from the mspec's filter values and
    roi_color.
    """
    spectrum_data = spectrum.filter_values(
        scale_to=scale_to, average_filters=average_filters
    )
    x_axis = [filt_value["wave"] for filt_value in spectrum_data.values()]
    y_axis = [filt_value["mean"] for filt_value in spectrum_data.values()]
    y_error = [filt_value["err"] for filt_value in spectrum_data.values()]
    # TODO: this definitely shouldn't be happening here
    if r_star:
        if spectrum.incidence_angle:
            cos_theta_i = np.cos(d2r(spectrum.incidence_angle))
            y_axis = [mean / cos_theta_i for mean in y_axis]
            y_error = [err / cos_theta_i for err in y_error]
    text = [
        filt + ", " + str(spectrum_data[filt]["wave"])
        for filt in spectrum_data
    ]
    fig = go.Figure(
        layout={
            **GRAPH_DISPLAY_DEFAULTS,
            "xaxis": AXIS_DISPLAY_DEFAULTS
            | {"title_text": "wavelength", "title_standoff": 5},
            "yaxis": AXIS_DISPLAY_DEFAULTS
            | {
                "title_text": "reflectance",
                "range": [min(0, min(y_axis) - 0.05), max(min(y_axis) + max(y_axis), max(y_axis) + 0.05)],
                "title_standoff": 4,
                "side": "right",
            },
        }
    )
    # TODO: clean input to make this toggleable again
    show_error = True
    try:
        color = spectrum.roi_hex_code()
    except AttributeError:
        color = "#1777B6" # color for Chem Cam spectra
    scatter = go.Scattergl(
        x=x_axis,
        y=y_axis,
        mode="lines+markers",
        text=text,
        line={"color": color},
        error_y={"array": y_error, "visible": show_error},
    )
    fig.add_trace(scatter)
    return fig


def sort_by_marker_size(errors, graph_df, marker_property_dict):
    sort_indices = graph_df["size"].argsort()
    graph_df = graph_df.loc[sort_indices].reset_index(drop=True)
    for axis in ("x", "y"):
        if errors[axis] is None:
            continue
        for key in ("array", "arrayminus"):
            if errors[axis].get(key) is None:
                continue
            errors[axis][key] = np.array(errors[axis][key])[sort_indices]
    for key in ("color", "size"):
        if marker_property_dict["marker"].get(key) is None:
            continue
        # solid colors
        if isinstance(marker_property_dict["marker"].get(key), str):
            continue
        marker_property_dict["marker"][key] = np.array(
            marker_property_dict["marker"][key]
        )[sort_indices]
    return errors, graph_df, marker_property_dict
