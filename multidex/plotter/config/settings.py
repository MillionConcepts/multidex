"""
basic settings for fresh application load. some of these may be overridden
very quickly by callbacks and are essentially placeholders.
"""
import sys
from itertools import product

from multidex.plotter.config.graph_style import css_variables

SETTINGS = {
    "average_filters": "False",
    "highlight-toggle.value": "off",
    "highlight-size-radio.value": 1,
    "highlight-symbol-drop.value": "none",
    "highlight-color-drop.value": "none",
    "scale_to": "none",
    "r_star": "True",
    "logical_quantifier": "AND",
    "graph-option-x.value": "ratio",
    "graph-option-y.value": "ratio",
    "graph-option-marker.value": "feature",
    "marker-symbol-drop.value": "circle",
    "palette-name-drop.value": "Bold",
    "palette-type-drop.value": "qualitative",
    "marker-outline-radio.value": "rgba(0,0,0,1)",
    "highlight-outline-radio.value": "rgba(0,0,0,1)",
    "marker-size-radio.value": 11,
    "plot_bgcolor": css_variables["dark-tint-0"],
    "showgrid": True,
    "gridcolor": css_variables["dark-tint-0"],
    "color-clip-bound-high.value": 100,
    "color-clip-bound-low.value": 0,
    "search_parameters": [],
    "palette_memory": {
        "sequential": "Plasma",
        "diverging": "delta_r",
        "solid": "hotpink",
        "cyclical": "IceFire",
        "qualitative": "Bold",
    },
}

SETTINGS |= {
    f"filter-{ix}-{axis}.value": None
    for ix, axis in product((1, 2, 3), ("x", "y", "marker"))
}

SETTINGS |= {f"component-{axis}.value": 0 for axis in ("x", "y", "marker")}

# values given in mappings with variable names of the form
# X_SETTINGS will override the default values in SETTINGS
# in instances of MultiDEx launched with instrument code X
CCAM_SETTINGS = {"graph-option-marker.value": "sol"}


def instrument_settings(instrument: str) -> dict:
    try:
        return SETTINGS | getattr(
            sys.modules[__name__], f"{instrument}_SETTINGS"
        )
    except AttributeError:
        return SETTINGS
