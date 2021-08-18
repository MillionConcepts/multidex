"""dash components that are graphs and associated helper functions"""

from typing import Mapping, Optional

import numpy as np
import pandas as pd
from plotly import graph_objects as go

from plotter.spectrum_ops import d2r
from plotter.styles.components import (
    ANNOTATION_SETTINGS,
    GRAPH_DISPLAY_DEFAULTS,
    AXIS_DISPLAY_DEFAULTS,
    SEARCH_FAILURE_MESSAGE_SETTINGS,
)


def plot_and_style_data(
    axis_display_settings,
    errors,
    fig,
    graph_df,
    marker_property_dict,
    x_title,
    y_title,
):
    fig.add_trace(
        go.Scatter(
            x=graph_df["x"],
            y=graph_df["y"],
            hovertext=graph_df["text"],
            customdata=graph_df["customdata"],
            mode="markers + text",
            marker={"color": "black", "size": 8},
        )
    )
    axis_display_dict = AXIS_DISPLAY_DEFAULTS | axis_display_settings
    # noinspection PyTypeChecker
    fig.update_xaxes(axis_display_dict | {"title_text": x_title})
    fig.update_yaxes(axis_display_dict | {"title_text": y_title})
    fig.update_traces(**marker_property_dict)
    for axis in ["x", "y"]:
        if errors[axis] is None:
            fig.update_traces({"error_" + axis: {"visible": False}})
        else:
            fig.update_traces(
                {
                    "error_" + axis: errors[axis]
                    | {
                        "visible": True,
                        "color": "rgba(0,0,0,0.3)",
                    }
                }
            )


def apply_graph_style(fig, graph_display_settings, zoom):
    display_dict = GRAPH_DISPLAY_DEFAULTS | graph_display_settings
    fig.update_layout(display_dict)
    if zoom is not None:
        fig.update_layout(
            {
                "xaxis": {"range": [zoom[0][0], zoom[0][1]]},
                "yaxis": {"range": [zoom[1][0], zoom[1][1]]},
            }
        )


def main_scatter_graph(
    graph_df: pd.DataFrame,
    errors: Mapping,
    marker_property_dict: Mapping,
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

    # sort points by marker size so that we can draw highlighted points
    # last, and thus at a higher 'z-axis'
    if len(graph_df["size"].unique()) > 1:
        errors, graph_df, marker_property_dict = sort_by_marker_size(
            errors, graph_df, marker_property_dict
        )
    # the click-to-label annotations
    draw_floating_labels(fig, graph_df, label_ids)
    # and the scattered points and their error bars
    plot_and_style_data(
        axis_display_settings,
        errors,
        fig,
        graph_df,
        marker_property_dict,
        x_title,
        y_title,
    )
    apply_graph_style(fig, graph_display_settings, None)
    if zoom is not None:
        fig.update_layout(
            {
                "xaxis": {"range": [zoom[0][0], zoom[0][1]]},
                "yaxis": {"range": [zoom[1][0], zoom[1][1]]},
            }
        )
    return fig


def failed_scatter_graph(message: str, graph_display_settings: Mapping):
    fig = go.Figure()
    fig.add_annotation(text=message, **SEARCH_FAILURE_MESSAGE_SETTINGS)
    apply_graph_style(fig, graph_display_settings, None)
    # fig.to_image()
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
                "range": [0, min(y_axis) + max(y_axis)],
                "title_standoff": 4,
                "side": "right",
            },
        }
    )
    # TODO: clean input to make this toggleable again
    show_error = True
    scatter = go.Scatter(
        x=x_axis,
        y=y_axis,
        mode="lines+markers",
        text=text,
        line={"color": spectrum.roi_hex_code()},
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
