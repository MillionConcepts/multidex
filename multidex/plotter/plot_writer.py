import warnings

from dustgoggles.pivot import split_on
from marslab.imgops.pltutils import attach_axis
import matplotlib.pyplot as plt
import matplotlib.font_manager as mplf
import matplotlib.cm as cm
from matplotlib import colors
import pandas as pd

from multidex.multidex_utils import re_get, keygrab
from multidex.plotter.colors import plotly_color_to_percent, get_scale_type
from multidex.plotter.config.output_style import (
    FONT_PATH,
    FONT_PATH_BOLD,
    LABEL_TEXT_SIZE,
    TICK_TEXT_SIZE,
    FITLINE_TEXT_SIZE
)
from plotter.components.graph_components import get_ordering

# TODO: grab fonts from config instead of hardcoding them here

# TODO: similarly, make these sizes configurable
LABEL_FP = mplf.FontProperties(fname=FONT_PATH, size=LABEL_TEXT_SIZE)
TICK_FP = mplf.FontProperties(fname=FONT_PATH, size=TICK_TEXT_SIZE)
FITLINE_FP = mplf.FontProperties(fname=FONT_PATH_BOLD, size=FITLINE_TEXT_SIZE)

def plotly_to_matplotlib_symbol(plotly_symbol):
    # TODO: taking the first item after splitting by '-' oversimplifies some
    #  symbols (e.g. star, triangle, etc)
    plotly_base_symbol = plotly_symbol.split('-')[0]
    plotly_symbol_keywords = {
        'circle': 'o',
        'square': 's',
        'diamond': 'D',
        'cross': 'P',
        'x': 'X',
        'triangle': '^',
        'pentagon': 'p',
        'hexagon2': 'H',
        'hexagon': 'h',
        'octagon': '8',
        'star': '*',
    }
    try:
        pyplot_symbol = plotly_symbol_keywords[plotly_base_symbol]
    except:
        pyplot_symbol = 'o'  # default to circles if the above fails
    return pyplot_symbol


def _pick_colormap(marker_settings) -> tuple[
    colors.ListedColormap | None, str | None
]:
    from matplotlib import colormaps

    palette_name = re_get(marker_settings, "palette-name-drop.value")
    if get_scale_type(palette_name) == "solid":
        # NOTE: matplotlib recognizes all solid colors multidex offers,
        #  because they're all CSS color names
        return None, palette_name
    cmaps = [
        c for n, c in colormaps.items() if n.lower() == palette_name.lower()
    ]
    if len(cmaps) > 0:
        return cmaps[0], None
    warnings.warn(f"The colormap '{palette_name}' is not available for plot "
                  f"export. Defaulting to 'inferno'.")
    return colormaps["inferno"], None

def _draw_axis_labels(graph_df, label_fp, tick_fp, xrange, yrange,
                      axtick_names):
    plt.xlim(xrange)
    plt.xlabel(
        graph_df.keys()[0], fontproperties=label_fp, wrap=True, va="top",
    )
    if 'x' in axtick_names:
        plt.xticks(font=tick_fp, ticks=list(range(len(axtick_names['x']))),
                   labels=axtick_names['x'])
    else:
        plt.xticks(font=tick_fp)
    plt.ylim(yrange)
    plt.ylabel(
        graph_df.keys()[1], fontproperties=label_fp, wrap=True, va="bottom",
    )
    if 'y' in axtick_names:
        plt.yticks(font=tick_fp, ticks=list(range(len(axtick_names['y']))),
                   labels=axtick_names['y'])
    else:
        plt.yticks(font=tick_fp)

def _draw_colorbar(ax, cclip, cmap, norm, graph_contents, qual_ticks,
                   label_fp, marker_props, tick_fp):
    # Check if the m_axis is qualitative, then set the colormap and norm
    # Extend the colorbar to indicate when the color map has been clipped
    if cclip[0] > 0 and cclip[1] < 100:
        extend_cbar = 'both'
    elif cclip[0] > 0:
        extend_cbar = 'min'
    elif cclip[1] < 100:
        extend_cbar = 'max'
    else:
        extend_cbar = 'neither'
    # Create the colorbar
    cax = attach_axis(ax, size="3%", pad="0.5%")
    colorbar = plt.colorbar(
        # b/c we have a variable number of actually-graphed mappables, need to
        # use a 'dummy' mappable here
        cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, extend=extend_cbar,
    )
    plt.ylabel(graph_contents.keys()[2], fontproperties=label_fp, wrap=True)
    plt.yticks(font=tick_fp, rotation=-15)
    colorbar.ax.ticklabel_format(scilimits=(-3, 3), useMathText=True)
    colorbar.ax.yaxis.offsetText.set_fontproperties(tick_fp)
    # Change tick labels to their qualitative names
    if marker_props['value_type'] == 'qual':
        cbar_tick_labels = [s.title() for s in qual_ticks]
        colorbar.ax.set_yticks(ticks=range(len(cbar_tick_labels)),
                               labels=cbar_tick_labels)


def _maybe_draw_grid(ax, axis_display_settings):
    showgrid = re_get(axis_display_settings, "showgrid") is True
    zeroline = re_get(axis_display_settings, "zeroline") is not False
    if not (showgrid or zeroline):
        return
    grid_color = plotly_color_to_percent(
        re_get(axis_display_settings, "gridcolor")
    )
    if showgrid:
        plt.grid(color=grid_color, alpha=grid_color[3])
        ax.set_axisbelow(True)
    if zeroline:
        ax.axhline(color=grid_color, alpha=grid_color[3],
                   lw=2, zorder=0)
        ax.axvline(color=grid_color, alpha=grid_color[3],
                   lw=2, zorder=0)


def _draw_highlight_scatter_points(ax, highlight_df, highlight_settings,
                                   marker_settings, solid_color, cmap,
                                   norm):
    # Set the fill and outline colors based on whether the marker symbol is
    # "open" or filled
    color_kwargs = {
        "edgecolors":  plotly_color_to_percent(
            highlight_settings["highlight-outline-radio.value"]
        )
    }

    if (hcol := highlight_settings["highlight-color-drop.value"]) != 'none':
        if '-open' in highlight_settings["highlight-symbol-drop.value"]:
            color_kwargs["color"], color_kwargs["edgecolors"] = 'none', hcol
        else:
            color_kwargs["color"] = hcol
    elif solid_color is None:
        color_kwargs["cmap"], color_kwargs["c"] = cmap, highlight_df['cnum']
        color_kwargs["norm"] = norm
    else:
        color_kwargs["color"] = solid_color

    _highlights = ax.scatter(
        x=highlight_df.iloc[:, 0],
        y=highlight_df.iloc[:, 1],
        marker=plotly_to_matplotlib_symbol(
            re_get(highlight_settings, "highlight-symbol-drop.value")
        ),
        # 3x matches plotly marker size convention
        s=(marker_settings["marker-size-radio.value"] ** 2) * (
                highlight_settings["highlight-size-radio.value"] * 3),
        alpha=re_get(highlight_settings,
                     "highlight-opacity-input.value") / 100,
        zorder=2,  # Make sure the highlights plot above other markers
        **color_kwargs
    )


def _draw_fit_line(ax, fitline_fp, line):
    _fit_line = ax.plot(
        line['x'], line['y'],
        color='black', lw=3,
        marker='none',
    )
    plt.text(
        0.01, 0.01,
        f" {line['text']} ",
        transform=ax.transAxes,
        fontproperties=fitline_fp,
        va='bottom', ha='left',
        bbox=dict(fc='gray', ec='none', alpha=0.3,
                  boxstyle="Square, pad=0.3"),
    )


def _draw_main_scatter_points(ax, graph_df, marker_settings,
                              outline_color, solid_color, cmap, norm):
    plot_kwargs = {
        "x": graph_df['xnum'],
        "y": graph_df['ynum'],
        # marker sizes (s) in scatter plots are the square of their standard
        # matplotlib marker size
        "s": re_get(marker_settings, "marker-size-radio.value") ** 2,
        "alpha": re_get(marker_settings, "marker-opacity-input.value") / 100,
        "edgecolors": outline_color,
        "marker": plotly_to_matplotlib_symbol(
            re_get(marker_settings, "marker-symbol-drop.value")
        ),
    }
    if cmap is not None:
        plot_kwargs["c"], plot_kwargs["cmap"] = graph_df['cnum'], cmap
        plot_kwargs["norm"] = norm
    else:
        plot_kwargs["color"] = solid_color
    ax.scatter(**plot_kwargs)


def _draw_errors(errors, graph_contents, marker_settings):
    plt.errorbar(
        x=graph_contents.iloc[:, 0],
        y=graph_contents.iloc[:, 1],
        xerr=errors.loc[:, "x"],
        yerr=errors.loc[:, "y"],
        elinewidth=1,
        capsize=5,
        ecolor="gray",
        fmt="none",  # only draw error bars, no extra points/lines
        alpha=re_get(marker_settings, "marker-opacity-input.value") / 100,
        zorder=0,  # plot error bars under the existing markers
    )


def fig_from_main_graph(
    graph_contents,
    metadata_df,
    xrange,
    yrange,
    marker_settings,
    marker_props,
    highlight_settings,
    highlight_ids,
    graph_display_settings,
    axis_display_settings,
    cclip,
    errors,
    line,
    spec_model,
    ax_field_names
):
    if re_get(marker_settings, "marker-outline-radio.value") == "off":
        outline_color = "face"
    else:
        outline_color = plotly_color_to_percent(
            re_get(marker_settings, "marker-outline-radio.value")
        )
    cmap, solid_color = _pick_colormap(marker_settings)
    graph_contents = graph_contents.copy()
    axtick_names = {}
    for i, ax in enumerate(("x", "y")):
        if keygrab(
            spec_model.graphable_properties(), "value", ax_field_names[ax]
        )["value_type"] == "qual":
            axtick_names[ax], graph_contents[f'{ax}num'] = _catencode(
                graph_contents.iloc[:, i],
                graph_contents.columns[i],
                metadata_df,
                spec_model
            )
        else:
            graph_contents[f'{ax}num'] = graph_contents.iloc[:, i]
    # TODO, maybe: this color logic is very ugly but there aren't a lot of
    #  not-awkward ways to share colorscales across mappables in matplotlib
    if re_get(highlight_settings, "highlight-toggle.value") == "on":
        hcol = highlight_settings["highlight-color-drop.value"]
        is_highlight = graph_contents.index.isin(highlight_ids)
        highlight_df, graph_df = split_on(graph_contents, is_highlight)
    else:
        highlight_df, graph_df, hcol = None, graph_contents, "none"
    cref, qual_ticknames, norm, cnum = None, None, None, None
    if hcol == "none" and cmap is not None:
        cref = graph_contents
    elif cmap is not None:
        cref = graph_df
    if cref is not None:
        if marker_props["value_type"] == "qual":
            qual_ticknames, cnum = _catencode(
                cref, marker_props['value'], metadata_df, spec_model
            )
            cmap = cmap.resampled(len(qual_ticknames))
        else:
            cnum = cref.iloc[:, 2]
        norm = colors.Normalize(cnum.min(), cnum.max())
        graph_df['cnum'] = cnum.loc[cnum.index.isin(graph_df.index)]
        if highlight_df is not None:
            highlight_df['cnum'] = cref.loc[
                cref.index.isin(highlight_df.index)
            ]
    # the 'agg' backend produces more consistent output and also prevents
    # macOS-specific threading errors
    plt.switch_backend('agg')
    fig, ax = plt.subplots(figsize=(15, 12), layout='tight')
    _draw_main_scatter_points(ax, graph_df, marker_settings,
                              outline_color, solid_color, cmap, norm)
    bg_color = plotly_color_to_percent(
        re_get(graph_display_settings, "plot_bgcolor")
    )
    ax.set_facecolor(bg_color)
    _draw_axis_labels(graph_df, LABEL_FP, TICK_FP, xrange, yrange,
                      axtick_names)
    _maybe_draw_grid(ax, axis_display_settings)
    if line is not None:
        _draw_fit_line(ax, FITLINE_FP, line)
    if re_get(highlight_settings, "highlight-toggle.value") == "on":
        _draw_highlight_scatter_points(ax, highlight_df, highlight_settings,
                                       marker_settings, solid_color, cmap,
                                       norm)
    # TODO: i'm not sure all-not-null is the correct criterion
    # TODO: doublecheck alignment b/w highlight & main
    if not errors.isnull().values.any():
        _draw_errors(errors, graph_contents, marker_settings)
    if solid_color is None:
        _draw_colorbar(ax, cclip, cmap, norm, graph_contents, qual_ticknames,
                       LABEL_FP, marker_props, TICK_FP)
    return fig


def _catencode(ref, value, metadata_df, spec_model):
    names = metadata_df.loc[ref.index, value.lower()].fillna('none')
    ospec = get_ordering(value.lower(), spec_model.instrument)
    if (carry := ospec.get('categoryarray')) is None:
        order = reversed(sorted(set(names)))
    else:
        order = carry
        if 'none' not in map(str.lower, carry) and 'none' in names:
            order = ('none',) + order
    encoding = {n.lower(): i for i, n in enumerate(order)}
    qualticks = tuple(map(str.title, encoding.keys()))
    cnum = pd.Series([encoding[n.lower()] for n in names], index=ref.index)
    return qualticks, cnum
