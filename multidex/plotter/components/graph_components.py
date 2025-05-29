"""dash components that are graphs and associated helper functions"""

from copy import deepcopy
from typing import Mapping, Optional

import numpy as np
import pandas as pd
from plotly import graph_objects as go

from multidex.plotter.colors import discretize_color_representations
from multidex.plotter.config.graph_style import (
    ANNOTATION_SETTINGS,
    GRAPH_DISPLAY_SETTINGS,
    AXIS_DISPLAY_SETTINGS,
    SEARCH_FAILURE_MESSAGE_SETTINGS,
)
from multidex.plotter.config.orderings import SPECIAL_ORDERINGS
from multidex.plotter.spectrum_ops import d2r


def add_regression(fig: go.Figure, unsplit_graph_df: pd.DataFrame):
    from scipy.stats import linregress

    try:
        reg = linregress(unsplit_graph_df["x"], unsplit_graph_df["y"])
        r_2, p = reg.rvalue ** 2, reg.pvalue
        m, b = reg.slope, reg.intercept
        r_2_text = f"{r_2:.1e}" if r_2 < 0.01 else round(r_2, 2)
        if p < 1e-30:
            p_text = "0"
        else:
            p_text = f"{p:.1e}" if p < 0.01 else round(p, 2)
        m_text =  f"{m:.1e}" if abs(m) < 0.01 else round(m, 2)
        b_text = f"{abs(b):.1e}" if abs(b) < 0.01 else round(abs(b), 2)
        op_text = "-" if b < 0 else "+"
        eqn_text = f"y = {m_text}x {op_text} {b_text}"
        xreg = [unsplit_graph_df["x"].min(), unsplit_graph_df["x"].max()]
        yreg = [xreg[0] * m + b, xreg[1] * m + b]
    except ValueError as ve:
        if "all x values are identical" not in str(ve):
            raise
        return
    except TypeError as te:
        # TODO: check
        return
    fig.add_trace(
        go.Scattergl(
            x=xreg,
            y=yreg,
            line={"color": "black", "width": 4},
            marker=None,
            showlegend=False,
            name="regression",
            mode="lines"
         )
    )
    # TODO, probably: make this configurable
    fig.update_layout(
        {
            "annotations": [
                {
                    "text": f"{eqn_text}; R^2 = {r_2_text}; p = {p_text}",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0,
                    "y": 0,
                    "showarrow": False,
                    "font": {"size": 28, "color": "black", "weight": 800},
                    "bgcolor": "rgba(0, 0, 0, 0.2)"
                }
            ]
        }
    )
    fig.update_traces(textposition="bottom center")


def get_ordering(field: Mapping, instrument: str):
    if (omap := SPECIAL_ORDERINGS.get(instrument)) is None:
        return {'categoryorder': 'category ascending'}
    if (ordering := omap.get(field)) is None:
        return {'categoryorder': 'category ascending'}
    return {'categoryarray': ordering}


def style_data(
    fig,
    graph_display_settings,
    axis_display_settings=None,
    x_title=None,
    y_title=None,
    marker_axis_type=None,
    marker_property_dict=None,
    zoom=None,
    instrument=None,
    ax_field_names=None
):
    axis_display_dict = AXIS_DISPLAY_SETTINGS | axis_display_settings
    # noinspection PyTypeChecker
    fig.update_xaxes(
        axis_display_dict
        | {"title_text": x_title}
        | get_ordering(ax_field_names['x'], instrument)
    )
    fig.update_yaxes(
        axis_display_dict
        | {"title_text": y_title}
        | get_ordering(ax_field_names['x'], instrument)
    )
    if (
        (marker_axis_type == "qual")
        # don't try to discretize solid colors
        and not isinstance(marker_property_dict["color"], str)
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


def construct_error_kwargs(errors):
    kwargs = {}
    for axis in ["x", "y"]:
        key = f"error_{axis}"
        if errors[axis].isna().all():
            value = {"visible": False}
        else:
            value = {
                "array": errors[axis].tolist(),
                "visible": True,
                "color": "rgba(0,0,0,0.3)",
            }
        kwargs[key] = value
    return kwargs


def apply_canvas_style(fig, graph_display_settings):
    display_dict = GRAPH_DISPLAY_SETTINGS | graph_display_settings
    fig.update_layout(display_dict)


def failed_scatter_graph(message: str, graph_display_settings: Mapping):
    fig = go.Figure()
    fig.add_annotation(text=message, **SEARCH_FAILURE_MESSAGE_SETTINGS)
    apply_canvas_style(fig, graph_display_settings)
    return fig


def main_scatter_graph(
    graph_df: pd.DataFrame,
    highlight_df: Optional[pd.DataFrame],
    graph_errors: pd.DataFrame,
    highlight_errors: pd.DataFrame,
    marker_property_dict: dict,
    marker_axis_type: str,
    coloraxis: dict,
    highlight_marker_dict: dict,
    graph_display_settings: Mapping,
    axis_display_settings: Mapping,
    label_ids: list[int],
    instrument: str,
    ax_field_names: dict[str, str],
    x_title: str = None,
    y_title: str = None,
    zoom: Optional[tuple[list[float, float]]] = None,
) -> go.Figure:
    """
    main graph component. this function creates the Plotly figure; data
    and metadata are filtered and formatted in callbacks.update_main_graph().
    """
    fig = go.Figure()
    # # the click-to-label annotations
    # draw_floating_labels(fig, graph_df, label_ids)
    # and the scattered points and their error bars
    # last-mile thing here to keep separate from highlight -- TODO: silly?
    marker_property_dict["color"] = graph_df["color"].values
    fig.update_coloraxes(coloraxis)
    error_kwargs = construct_error_kwargs(graph_errors)
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
            marker=marker_property_dict,
            **error_kwargs
        )
    )

    if highlight_df is not None:
        # draw_floating_labels(fig, highlight_df, label_ids)
        full_marker_dict = dict(deepcopy(marker_property_dict))
        full_marker_dict = full_marker_dict | highlight_marker_dict
        if "color" not in highlight_marker_dict:
            # just treat it like the graph df
            full_marker_dict["color"] = highlight_df["color"].values
        highlight_error_kwargs = construct_error_kwargs(highlight_errors)
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
                marker=full_marker_dict,
                **highlight_error_kwargs
            )
        )
    # last-step primary canvas stuff: set bounds, gridline color, titles, etc.
    style_data(
        fig,
        graph_display_settings,
        axis_display_settings,
        x_title,
        y_title,
        marker_axis_type,
        marker_property_dict,
        zoom,
        instrument,
        ax_field_names
    )
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
        scale_to=scale_to, average_filters=average_filters, show_bayers=False
    )
    x_axis = [filt_value["wave"] for filt_value in spectrum_data.values()]
    y_axis = [filt_value["mean"] for filt_value in spectrum_data.values()]
    y_error = [filt_value["std"] for filt_value in spectrum_data.values()]
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
    # create y_axis_range based on min y-value. Pin to zero unless there are
    # negative values
    if min(y_axis) < 0:
        y_axis_range = [min(y_axis) - 0.05, max(y_axis) + 0.05]
    else:
        y_axis_range = [0, min(y_axis) + max(y_axis)]
    fig = go.Figure(
        layout={
            **GRAPH_DISPLAY_SETTINGS,
            "xaxis": AXIS_DISPLAY_SETTINGS
                     | {"title_text": "wavelength", "title_standoff": 5},
            "yaxis": AXIS_DISPLAY_SETTINGS
                     | {
                "title_text": "reflectance",
                "range": y_axis_range,
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
        if marker_property_dict.get(key) is None:
            continue
        # solid colors
        if isinstance(marker_property_dict.get(key), str):
            continue
        marker_property_dict[key] = np.array(
            marker_property_dict[key]
        )[sort_indices]
    return errors, graph_df, marker_property_dict
