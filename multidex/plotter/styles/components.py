from multidex_utils import fetch_css_variables

css_variables = fetch_css_variables()

ANNOTATION_SETTINGS = {
    "font": {
        "family": "Fira Mono",
        "size": 14,
        "color": css_variables["clean-parchment"],
    },
    "bgcolor": "rgba(0,0,0,0.8)",
    "arrowwidth": 3,
    "xshift": -8,
    "yshift": 8,
    "captureevents": False,
}

SEARCH_FAILURE_MESSAGE_SETTINGS = {
    "font": {"family": "Fira Mono", "size": 32},
}

GRAPH_DISPLAY_DEFAULTS = {
    "margin": {"l": 10, "r": 10, "t": 25, "b": 0},
    "plot_bgcolor": css_variables["dark-tint-0"],
    "paper_bgcolor": css_variables["clean-parchment"],
}

AXIS_DISPLAY_DEFAULTS = {
    "showline": True,
    "showgrid": True,
    "mirror": True,
    "linewidth": 2,
    "gridcolor": css_variables["dark-tint-0"],
    "linecolor": css_variables["dark-tint-1"],
    "zerolinecolor": css_variables["dark-tint-1"],
    "spikecolor": css_variables["dark-tint-1"],
    "tickcolor": css_variables["midnight-ochre"],
    "tickfont": {"family": "Fira Mono"},
    "titlefont": {"family": "Fira Mono"},
    "title_text": None,
}

GRAPH_CONFIG_SETTINGS = {
    "modeBarButtonsToRemove": [
        "hoverCompareCartesian",
        "resetScale2d",
        "hoverClosestCartesian",
        "hoverCompareCartesian",
        "toImage"
    ],
    "displaylogo": False,
}

COLORBAR_SETTINGS = {
    "tickfont": {
        "family": "Fira Mono",
        "color": css_variables["midnight-ochre"],
    },
    "titlefont": {"family": "Fira Mono", "size": 14},
    "tickangle": 15,
    "title": {"side": "right"}
    # "x": -0.18,
    # "ticklabelposition": "outside top"
}
