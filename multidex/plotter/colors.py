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


def get_palette_from_scale_name(
    scale_name, count, qualitative=True
):
    scale = dig_for_value(get_plotly_colorscales(), scale_name)
    # %rgb representation
    percents = np.array(scale_to_percents(scale))
    if qualitative is True:
        # i.e., take explicit color values from the palette
        wheel = cycle(percents)
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


# note we're assuming this just has one -- or one relevant -- trace
def discretize_color_representations(fig):
    marker_dict = next(fig.select_traces())["marker"]
    marker_dict = discretize_marker_colors(marker_dict)
    fig.update_traces(marker=marker_dict)
    return fig


def discretize_marker_colors(marker_dict):
    tickvals = marker_dict["colorbar"]["tickvals"]
    continuous_scale = [val[1] for val in marker_dict["colorscale"]]
    percent_scale = np.array(scale_to_percents(continuous_scale))
    discrete_scale = make_discrete_scale(percent_scale, len(tickvals) + 1)
    marker_dict["colorscale"] = discrete_scale
    # don't ask me why they define tick positions like this...
    # first, a special case:
    if len(tickvals) == 2:
        marker_dict["colorbar"]["tickvals"] = [0.25, 0.75]
    # otherwise, interpolate to the weird quasi-relative scale they use
    else:
        marker_dict["colorbar"]["tickvals"] = np.interp(
            tickvals,
            tickvals,
            np.linspace(0.5, len(tickvals) - 1.5, len(tickvals)),
        )
    return marker_dict


def generate_palette_options(scale_value, palette_value):
    if scale_value == "solid":
        output_options = SOLID_MARKER_COLORS
    else:
        colormaps = get_plotly_colorscales()
        output_options = [
            {"label": colormap, "value": colormap}
            for colormap in colormaps[scale_value].keys()
        ]
    if (palette_value is None) or palette_value not in [option["value"] for option in output_options]:
        output_value = output_options[0]["value"]
    else:
        output_value = palette_value
    return output_options, output_value
