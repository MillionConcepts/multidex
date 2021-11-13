import inspect
from collections.abc import Sequence
from functools import partial
from itertools import chain, cycle
from operator import mul
from typing import Union

from dustgoggles.structures import dig_for_value
import matplotlib.colors as mcolors
import numpy as np
import plotly.colors as pcolors
import plotly.graph_objects as go
from more_itertools import windowed

from plotter.styles.marker_style import SOLID_MARKER_COLORS

PLOTLY_COLOR_MODULES = (
    pcolors.sequential,
    pcolors.cyclical,
    pcolors.diverging,
    pcolors.qualitative,
)


def get_plotly_colorscales(modules: tuple = PLOTLY_COLOR_MODULES) -> dict:
    return {
        # they're all so conveniently named!
        module.__name__.split(".")[-1]: {
            scale[0]: scale[1]
            for scale in inspect.getmembers(module)
            if isinstance(scale[1], Sequence) and not scale[0].startswith("_")
        }
        for module in modules
    }


def plotly_colorscale_type(
    scale_name: str, modules: tuple = PLOTLY_COLOR_MODULES
) -> str:
    scale_dict = get_plotly_colorscales(modules)
    for scale_type in scale_dict.keys():
        if scale_name in scale_dict[scale_type].keys():
            return scale_type


def rgbstring_to_rgb_percent(rgbstring: str) -> tuple[float]:
    # noinspection PyTypeChecker
    return tuple(
        map(
            partial(mul, 1 / 255),
            map(
                int, rgbstring.replace("rgb(", "").replace(")", "").split(",")
            ),
        )
    )


def plotly_color_to_percent(
    plotly_color: Union[str, tuple[float]]
) -> tuple[float]:
    if plotly_color.startswith("#"):
        return mcolors.to_rgb(plotly_color)
    if plotly_color.startswith("rgb"):
        return rgbstring_to_rgb_percent(plotly_color)
    return plotly_color


def scale_to_percents(scale: Sequence[str]) -> tuple[tuple[float]]:
    return tuple(map(plotly_color_to_percent, scale))


def percent_to_plotly_rgb(percent: Sequence[float]) -> str:
    return (
        f"rgb("
        f"{','.join(tuple(map(str, map(round, map(partial(mul, 255), percent)))))})"
    )


def scale_to_plotly_rgb(scale: Sequence[Sequence[float]]) -> tuple[str]:
    return tuple(map(percent_to_plotly_rgb, scale))


def get_lut(percent_scale, count):
    interp_points = np.linspace(0, len(percent_scale), count)
    return (
        np.array(
            [
                np.interp(
                    interp_points,
                    np.arange(len(percent_scale)),
                    percent_scale[:, ix],
                )
                for ix in range(3)
            ]
        )
        .astype(np.float64)
        .T
    )


def get_palette_from_scale_name(scale_name, count, qualitative=True):
    scale = dig_for_value(get_plotly_colorscales(), scale_name)
    # %rgb representation
    percents = np.array(scale_to_percents(scale))
    if qualitative is True:
        # i.e., take explicit color values from the palette
        wheel = cycle(percents)
        # prevent weird behavior from go.Scatter in some cases
        count = max(count, 2)
        lut = [next(wheel) for _ in range(count)]
    else:
        lut = get_lut(percents, count)
    return scale_to_plotly_rgb(lut)


def make_discrete_scale(percent_scale, count):
    discrete_scale = scale_to_plotly_rgb(
        get_lut(percent_scale, count).tolist()
    )
    positioned_discrete_scale = [
        (position, color)
        for position, color in zip(np.linspace(0, 1, count), discrete_scale)
    ]
    return list(
        chain.from_iterable(
            [
                ((bottom[0], bottom[1]), (top[0], bottom[1]))
                for bottom, top in windowed(positioned_discrete_scale, 2)
            ]
        )
    )


def discretize_color_representations(fig: go.Figure) -> go.Figure:
    """
    convert the first colorbar found in fig (if any) to "discrete"
    representation (chunky rather than smooth transitions between colors at
    tick boundaries).
    """
    # TODO, maybe: clean this up -- the obnoxiously-slightly-different syntax
    #  for update_traces() and update_coloraxes() makes it icky
    coloraxes, traces = fig.select_coloraxes(), fig.select_traces()
    for coloraxis in coloraxes:
        if "colorbar" not in coloraxis:
            continue
        coloraxis = discretize_colors(coloraxis)
        # don't keep doing this (selectors are complicated & not needed here)
        fig.update_coloraxes(coloraxis)
        return fig
    for trace in traces:
        if "marker" not in trace:
            continue
        marker = trace["marker"]
        if "colorbar" not in marker:
            continue
        marker = discretize_colors(marker)
        fig.update_traces(marker=marker)
        return fig
    # no colorbars? FINE
    return fig


def discretize_colors(colorbar_parent):
    tickvals = colorbar_parent["colorbar"]["tickvals"]
    # generally indicating explicitly-specified colors per point --
    # already as discrete as they can get!
    if ("colorscale" not in colorbar_parent) or (
        colorbar_parent["colorscale"] is None
    ):
        return colorbar_parent
    continuous_scale = [val[1] for val in colorbar_parent["colorscale"]]
    percent_scale = np.array(scale_to_percents(continuous_scale))
    discrete_scale = make_discrete_scale(percent_scale, len(tickvals) + 1)
    colorbar_parent["colorscale"] = discrete_scale
    # don't ask me why they define tick positions like this...
    # first, a special case:
    if len(tickvals) == 2:
        colorbar_parent["colorbar"]["tickvals"] = [0.25, 0.75]
    # otherwise, interpolate to the weird quasi-relative scale they use
    else:
        colorbar_parent["colorbar"]["tickvals"] = np.interp(
            tickvals,
            tickvals,
            np.linspace(0.5, len(tickvals) - 1.5, len(tickvals)),
        )
    return colorbar_parent


def generate_palette_options(
    scale_value, palette_value, remembered_value, allow_none=False
):
    if scale_value == "solid":
        output_options = list(SOLID_MARKER_COLORS)
    else:
        colormaps = get_plotly_colorscales()
        output_options = [
            {"label": colormap, "value": colormap}
            for colormap in colormaps[scale_value].keys()
        ]
    # "none" option used for some highlight features
    if allow_none is True:
        output_options = [{"label": "none", "value": "none"}] + output_options
    # fall back to first colormap / color in specified scale type if we've
    # really swapped scale types or on a clean load
    if (palette_value is None) or palette_value not in [
        option["value"] for option in output_options
    ]:
        if remembered_value is None:
            output_value = output_options[0]["value"]
        else:
            output_value = remembered_value
    else:
        output_value = palette_value
    return output_options, output_value
