import inspect
import re
from collections.abc import Sequence
from functools import partial
from itertools import chain, cycle
from numbers import Number
from operator import mul
from typing import Union

from dustgoggles.structures import dig_for_value
import numpy as np
import plotly.colors as pcolors
import plotly.graph_objects as go
from more_itertools import windowed

from plotter.config.marker_style import SOLID_MARKER_COLOR_SETTINGS

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


def get_scale_type(
    scale_name: str, modules: tuple = PLOTLY_COLOR_MODULES
) -> str:
    scale_dict = get_plotly_colorscales(modules)
    for scale_type in scale_dict.keys():
        if scale_name in scale_dict[scale_type].keys():
            return scale_type


def rgbstring_to_rgb_percent(rgb: str) -> tuple[float]:
    division = map(
        partial(mul, 1 / 255),
        map(int, rgb.replace("rgb(", "").replace(")", "").split(",")),
    )
    # noinspection PyTypeChecker
    return tuple(division)


# the following function is vendored from matplotlib.colors and carries the
# matplotlib license. see the bottom of this module for a full copy of this license.
def to_rgba_no_colorcycle(c, alpha=None):
    """
    Convert *c* to an RGBA color, with no support for color-cycle syntax.

    If *alpha* is given, force the alpha value of the returned RGBA tuple
    to *alpha*. Otherwise, the alpha value from *c* is used, if it has alphaba
    information, or defaults to 1.

    *alpha* is ignored for the color value ``"none"`` (case-insensitive),
    which always maps to ``(0, 0, 0, 0)``.
    """
    orig_c = c
    if isinstance(c, str):
        # hex color in #rrggbb format.
        match = re.match(r"\A#[a-fA-F0-9]{6}\Z", c)
        if match:
            return (tuple(int(n, 16) / 255
                          for n in [c[1:3], c[3:5], c[5:7]])
                    + (alpha if alpha is not None else 1.,))
        # hex color in #rgb format, shorthand for #rrggbb.
        match = re.match(r"\A#[a-fA-F0-9]{3}\Z", c)
        if match:
            return (tuple(int(n, 16) / 255
                          for n in [c[1]*2, c[2]*2, c[3]*2])
                    + (alpha if alpha is not None else 1.,))
        # hex color with alpha in #rrggbbaa format.
        match = re.match(r"\A#[a-fA-F0-9]{8}\Z", c)
        if match:
            color = [int(n, 16) / 255
                     for n in [c[1:3], c[3:5], c[5:7], c[7:9]]]
            if alpha is not None:
                color[-1] = alpha
            return tuple(color)
        # hex color with alpha in #rgba format, shorthand for #rrggbbaa.
        match = re.match(r"\A#[a-fA-F0-9]{4}\Z", c)
        if match:
            color = [int(n, 16) / 255
                     for n in [c[1]*2, c[2]*2, c[3]*2, c[4]*2]]
            if alpha is not None:
                color[-1] = alpha
            return tuple(color)
        # string gray.
        try:
            c = float(c)
        except ValueError:
            pass
        else:
            if not (0 <= c <= 1):
                raise ValueError(
                    f"Invalid string grayscale value {orig_c!r}. "
                    f"Value must be within 0-1 range")
            return c, c, c, alpha if alpha is not None else 1.
        raise ValueError(f"Invalid RGBA argument: {orig_c!r}")
    # turn 2-D array into 1-D array
    if isinstance(c, np.ndarray):
        if c.ndim == 2 and c.shape[0] == 1:
            c = c.reshape(-1)
    # tuple color.
    if not np.iterable(c):
        raise ValueError(f"Invalid RGBA argument: {orig_c!r}")
    if len(c) not in [3, 4]:
        raise ValueError("RGBA sequence should have length 3 or 4")
    if not all(isinstance(x, Number) for x in c):
        # Checks that don't work: `map(float, ...)`, `np.array(..., float)` and
        # `np.array(...).astype(float)` would all convert "0.5" to 0.5.
        raise ValueError(f"Invalid RGBA argument: {orig_c!r}")
    # Return a tuple to prevent the cached value from being modified.
    c = tuple(map(float, c))
    if len(c) == 3 and alpha is None:
        alpha = 1
    if alpha is not None:
        c = c[:3] + (alpha,)
    if any(elem < 0 or elem > 1 for elem in c):
        raise ValueError("RGBA values should be within 0-1 range")
    return c


def plotly_color_to_percent(
    plotly_color: Union[str, tuple[float]]
) -> tuple[float]:
    if plotly_color.startswith("#"):
        return to_rgba_no_colorcycle(plotly_color)
    if plotly_color.startswith("rgb"):
        return rgbstring_to_rgb_percent(plotly_color)
    return plotly_color


def scale_to_percents(scale: Sequence[str]) -> tuple[tuple[float]]:
    return tuple(map(plotly_color_to_percent, scale))


def percent_to_plotly_rgb(percent: Sequence[float]) -> str:
    rgb = tuple(map(str, map(round, map(partial(mul, 255), percent))))
    return f"rgb({','.join(rgb)})"


def scale_to_plotly_rgb(scale: Sequence[Sequence[float]]) -> tuple[str]:
    return tuple(map(percent_to_plotly_rgb, scale))


def get_lut(percent_scale, count):
    interp_points = np.linspace(0, len(percent_scale), count)
    interp_channels = [
        np.interp(
            interp_points,
            np.arange(len(percent_scale)),
            percent_scale[:, ix],
        )
        for ix in range(3)
    ]
    return np.array(interp_channels).astype(np.float64).T


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
    coloraxes, traces = fig.select_coloraxes(), fig.select_traces()
    # note obnoxiously-slightly-different API syntax for trace and coloraxis
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
        output_options = list(SOLID_MARKER_COLOR_SETTINGS)
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


"""
matplotlib license, applicable to to_rgba_nocolorcycle()

License agreement for matplotlib versions 1.3.0 and later
=========================================================

1. This LICENSE AGREEMENT is between the Matplotlib Development Team
("MDT"), and the Individual or Organization ("Licensee") accessing and
otherwise using matplotlib software in source or binary form and its
associated documentation.

2. Subject to the terms and conditions of this License Agreement, MDT
hereby grants Licensee a nonexclusive, royalty-free, world-wide license
to reproduce, analyze, test, perform and/or display publicly, prepare
derivative works, distribute, and otherwise use matplotlib
alone or in any derivative version, provided, however, that MDT's
License Agreement and MDT's notice of copyright, i.e., "Copyright (c)
2012- Matplotlib Development Team; All Rights Reserved" are retained in
matplotlib  alone or in any derivative version prepared by
Licensee.

3. In the event Licensee prepares a derivative work that is based on or
incorporates matplotlib or any part thereof, and wants to
make the derivative work available to others as provided herein, then
Licensee hereby agrees to include in any such work a brief summary of
the changes made to matplotlib .

4. MDT is making matplotlib available to Licensee on an "AS
IS" basis.  MDT MAKES NO REPRESENTATIONS OR WARRANTIES, EXPRESS OR
IMPLIED.  BY WAY OF EXAMPLE, BUT NOT LIMITATION, MDT MAKES NO AND
DISCLAIMS ANY REPRESENTATION OR WARRANTY OF MERCHANTABILITY OR FITNESS
FOR ANY PARTICULAR PURPOSE OR THAT THE USE OF MATPLOTLIB
WILL NOT INFRINGE ANY THIRD PARTY RIGHTS.

5. MDT SHALL NOT BE LIABLE TO LICENSEE OR ANY OTHER USERS OF MATPLOTLIB
 FOR ANY INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES OR
LOSS AS A RESULT OF MODIFYING, DISTRIBUTING, OR OTHERWISE USING
MATPLOTLIB , OR ANY DERIVATIVE THEREOF, EVEN IF ADVISED OF
THE POSSIBILITY THEREOF.

6. This License Agreement will automatically terminate upon a material
breach of its terms and conditions.

7. Nothing in this License Agreement shall be deemed to create any
relationship of agency, partnership, or joint venture between MDT and
Licensee.  This License Agreement does not grant permission to use MDT
trademarks or trade name in a trademark sense to endorse or promote
products or services of Licensee, or any third party.

8. By copying, installing or otherwise using matplotlib ,
Licensee agrees to be bound by the terms and conditions of this License
Agreement.
"""