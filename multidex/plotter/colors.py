import inspect
from collections.abc import Sequence
from functools import partial
from itertools import chain
from operator import mul
from typing import Union

import matplotlib.colors as mcolors
import numpy as np
import plotly.colors as pcolors
from more_itertools import windowed

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
    fig.update_traces(marker=marker_dict)
    return fig


def generate_color_scale_options(scale_type, value):
    colormaps = get_plotly_colorscales()
    # this case should occur only on load of a saved state with a solid color
    if scale_type not in colormaps.keys():
        scale_type = "sequential"
    options = [
        {"label": colormap, "value": colormap}
        for colormap in colormaps[scale_type].keys()
    ]
    if (value is None) or value not in [option["value"] for option in options]:
        value = options[0]["value"]
    return options, value
