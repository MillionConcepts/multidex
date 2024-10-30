"""
these are simply lists of inputs that refer to ids of components produced by
plotter.components. they are defined here for convenience in order to
avoid excessive repetition in app structure definition and function calls.

"""
from dash.dependencies import Input, Output

CALC_OPTION_DROPDOWNS = [
    "filter-1-marker",
    "filter-2-marker",
    "filter-3-marker",
    "filter-1-y",
    "filter-2-y",
    "filter-3-y",
    "filter-1-x",
    "filter-2-x",
    "filter-3-x",
    "component-x",
    "component-y",
    "component-marker",
]

X_INPUTS = [
    Input(dropdown, "value")
    for dropdown in CALC_OPTION_DROPDOWNS
    if dropdown.endswith("-x")
] + [Input("graph-option-x", "value")]

Y_INPUTS = [
    Input(dropdown, "value")
    for dropdown in CALC_OPTION_DROPDOWNS
    if dropdown.endswith("-y")
] + [Input("graph-option-y", "value")]

MARKER_INPUTS = [
    Input(dropdown, "value")
    for dropdown in CALC_OPTION_DROPDOWNS
    if dropdown.endswith("-marker")
] + [
    Input("graph-option-marker", "value"),
    Input("palette-name-drop", "value"),
    Input("color-clip-bound-low", "value"),
    Input("color-clip-bound-high", "value"),
    Input("marker-outline-radio", "value"),
    Input("marker-size-radio", "value"),
    Input("marker-symbol-drop", "value"),
    Input("marker-alpha-input", "value")
]

HIGHLIGHT_INPUTS = [
    Input("highlight-toggle", "value"),
    Input("highlight-size-radio", "value"),
    Input("highlight-symbol-drop", "value"),
    Input("highlight-color-drop", "value")
]

GRAPH_DISPLAY_INPUTS = [
    Input("main-graph-bg-radio", "value"),
    Input("main-graph-gridlines-radio", "value"),
]

FILTER_DROPDOWN_OUTPUTS = [
    Output(dropdown, "options")
    for dropdown in CALC_OPTION_DROPDOWNS
    if "filter" in dropdown
] + [
    Output(dropdown, "value")
    for dropdown in CALC_OPTION_DROPDOWNS
    if "filter" in dropdown
]

# client-side url for serving images to the user.
STATIC_IMAGE_URL = "/images/browse/"
