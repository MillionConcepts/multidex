"""
basic settings for fresh application load. some of these may be overridden
very quickly by callbacks and are essentially placeholders.
"""
from itertools import product

from plotter.styles.graph_style import css_variables

DEFAULT_SETTINGS_DICTIONARY = {
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
        }
}

DEFAULT_SETTINGS_DICTIONARY |= {
    f"filter-{ix}-{axis}.value": None
    for ix, axis in product((1, 2, 3), ("x", "y", "marker"))
}

DEFAULT_SETTINGS_DICTIONARY |= {
    f"component-{axis}.value": 0 for axis in ("x", "y", "marker")
}

