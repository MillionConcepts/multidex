from pathlib import Path

from dustgoggles.pivot import split_on
import pandas as pd

from multidex.multidex_utils import re_get
from multidex.plotter.colors import plotly_color_to_percent


def plotly_to_matplotlib_symbol(plotly_symbol):
    # TODO: taking the first item after splitting by '-' oversimplifies some symbols (e.g. star, triangle, etc)
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
        line
):
    from marslab.imgops.pltutils import attach_axis
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as mplf
    import matplotlib.cm as cm
    import matplotlib.colors as colors

    # the 'agg' backend produces more consistent output and also prevents
    # macOS-specific threading errors
    plt.switch_backend('agg')

    # Create the matplotlib figure and its subplot
    fig, ax = plt.subplots(figsize=(15, 12), layout='tight')

    # Set fonts
    # TODO: maybe grab fonts from multidex/plotter/config/graph_style.py
    #  instead of hardcoding them here
    font_file = Path(
        Path(__file__).parent, "application/assets/fonts/TitilliumWeb-Light.ttf"
    )
    font_file_bold = Path(
        Path(__file__).parent, "application/assets/fonts/TitilliumWeb-Bold.ttf"
    )
    label_fp = mplf.FontProperties(fname=font_file, size=26)
    tick_fp = mplf.FontProperties(fname=font_file, size=22)
    fitline_fp = mplf.FontProperties(fname=font_file_bold, size=24)

    # Marker outline color
    if re_get(marker_settings, "marker-outline-radio.value") == "off":
        outline_color = "face"
    else:
        outline_color = plotly_color_to_percent(
            re_get(marker_settings, "marker-outline-radio.value")
        )

    if re_get(highlight_settings, "highlight-toggle.value") == "on":
        highlight_df, graph_df = split_on(
            graph_contents, graph_contents.index.isin(highlight_ids)
        )
    else:
        highlight_df, graph_df = None, graph_contents

    # Plot the data points
    plot = ax.scatter(
        x=graph_df.iloc[:, 0],
        y=graph_df.iloc[:, 1],
        c=graph_df.iloc[:, 2],
        # marker sizes (s) in scatter plots are the square of their standard
        # matplotlib marker size
        s=re_get(marker_settings, "marker-size-radio.value") ** 2,
        alpha=re_get(marker_settings, "marker-opacity-input.value") / 100,
        edgecolors=outline_color,
        marker=plotly_to_matplotlib_symbol(
            re_get(marker_settings, "marker-symbol-drop.value")
        ),
    )
    # Colormap
    cmap = re_get(marker_settings, "palette-name-drop.value")
    try:
        plot.set_cmap(cmap)
    except ValueError:
        cmap = cmap.lower()
        plot.set_cmap(cmap)
    # Background color
    bg_color = plotly_color_to_percent(
        re_get(graph_display_settings, "plot_bgcolor")
    )
    ax.set_facecolor(bg_color)
    # Axis limits and labels
    plt.xlim(xrange)
    plt.xlabel(
        graph_df.keys()[0], fontproperties=label_fp,
        wrap=True, va="top",
    )
    plt.xticks(font=tick_fp)
    plt.ylim(yrange)
    plt.ylabel(
        graph_df.keys()[1], fontproperties=label_fp,
        wrap=True, va="bottom",
    )
    plt.yticks(font=tick_fp)
    # Error bars
    if not errors.isnull().values.any():
        plt.errorbar(
            x=graph_df.iloc[:, 0],
            y=graph_df.iloc[:, 1],
            xerr=errors.loc[:, "x"],
            yerr=errors.loc[:, "y"],
            elinewidth=1,
            capsize=5,
            ecolor="gray",
            fmt="none",  # only draw error bars, no extra points/lines
            alpha=re_get(marker_settings, "marker-opacity-input.value") / 100,
            zorder=0,  # plot error bars under the existing markers
        )
    # Gridlines
    if re_get(axis_display_settings, "showgrid") == True:
        grid_color = plotly_color_to_percent(
            re_get(axis_display_settings, "gridcolor")
        )
        plt.grid(color=grid_color, alpha=grid_color[3])
        ax.set_axisbelow(True)
    # Zero lines
    if re_get(axis_display_settings, "zeroline") == False:
        pass
    else:
        ax.axhline(color=grid_color, alpha=grid_color[3],
                   lw=2, zorder=0)
        ax.axvline(color=grid_color, alpha=grid_color[3],
                   lw=2, zorder=0)
    # Fit line
    if line is not None:
        fit_line = ax.plot(
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
    # Highlighting
    if re_get(highlight_settings, "highlight-toggle.value") == "on":
        # Set the fill and outline colors based on whether the marker symbol is
        # "open" or filled
        if '-open' in highlight_settings["highlight-symbol-drop.value"]:
            hl_fillcolor = 'none'
            hl_outline = highlight_settings["highlight-color-drop.value"]
        else:
            hl_fillcolor = highlight_settings["highlight-color-drop.value"]
            hl_outline = plotly_color_to_percent(highlight_settings["highlight-outline-radio.value"])
        # Plot the highlighted points
        _highlights = ax.scatter(
            x=highlight_df.iloc[:, 0],
            y=highlight_df.iloc[:, 1],
            color=hl_fillcolor,
            edgecolors=hl_outline,
            marker=plotly_to_matplotlib_symbol(
                re_get(highlight_settings, "highlight-symbol-drop.value")
            ),
            # Hard coding in an extra 3x to mimic the plotly marker size
            s=(marker_settings["marker-size-radio.value"] ** 2) * (
                        highlight_settings["highlight-size-radio.value"] * 3),
            alpha=re_get(highlight_settings, "highlight-opacity-input.value") / 100,
            zorder=2,  # Make sure the highlights plot above other markers
        )

    # This whole last section is the colorbar:
    marker_axis = graph_df.iloc[:, 2]
    # Check if the m_axis is qualitative, then set the colormap and norm
    if marker_props['value_type'] == 'qual':
        cbar_cmap = plt.get_cmap(cmap, len(marker_axis.unique()))
        norm = colors.NoNorm(vmin=min(marker_axis), vmax=max(marker_axis))
    else:
        cbar_cmap = cmap
        norm = plt.Normalize(vmin=min(marker_axis), vmax=max(marker_axis))
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
        cm.ScalarMappable(norm=norm, cmap=cbar_cmap),
        cax=cax,
        extend=extend_cbar,
    )
    # Font properties and other label parameters
    plt.ylabel(graph_contents.keys()[2], fontproperties=label_fp,
               wrap=True)
    plt.yticks(font=tick_fp, rotation=-15)
    colorbar.ax.ticklabel_format(scilimits=(-3, 3), useMathText=True)
    colorbar.ax.yaxis.offsetText.set_fontproperties(tick_fp)
    # Change tick labels to their qualitative names
    if marker_props['value_type'] == 'qual':
        cbar_ticks_df = pd.DataFrame({
            'encoding': graph_df.iloc[:, 2],
            'name': metadata_df.loc[
                graph_df.index, marker_props['value']
            ].fillna('none')
        })
        cbar_tick_labels = list(
            cbar_ticks_df.drop_duplicates().sort_values(by='encoding')['name']
        )
        cbar_tick_labels = [s.title() for s in cbar_tick_labels]
        colorbar.ax.set_yticks(ticks=range(len(cbar_tick_labels)),
                               labels=cbar_tick_labels)

    return fig
