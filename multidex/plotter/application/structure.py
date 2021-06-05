"""
these are simply lists of inputs that refer to ids of components produced by
plotter.components. they are defined here for convenience in order to
avoid excessive repetition in app structure definition and function calls.

"""
from dash.dependencies import Input, Output

CALC_OPTION_DROPDOWNS = [
    "main-filter-1-marker",
    "main-filter-2-marker",
    "main-filter-3-marker",
    "main-filter-1-y",
    "main-filter-2-y",
    "main-filter-3-y",
    "main-filter-1-x",
    "main-filter-2-x",
    "main-filter-3-x",
    "main-component-x",
    "main-component-y",
    "main-component-marker",
]

X_INPUTS = [
    Input(dropdown, "value")
    for dropdown in CALC_OPTION_DROPDOWNS
    if dropdown.endswith("-x")
] + [Input("main-graph-option-x", "value")]

Y_INPUTS = [
    Input(dropdown, "value")
    for dropdown in CALC_OPTION_DROPDOWNS
    if dropdown.endswith("-y")
] + [Input("main-graph-option-y", "value")]

MARKER_INPUTS = [
    Input(dropdown, "value")
    for dropdown in CALC_OPTION_DROPDOWNS
    if dropdown.endswith("-marker")
] + [
    Input("main-graph-option-marker", "value"),
    Input("main-coloring-type", "value"),
    Input("main-color-scale", "value"),
    Input("main-color-solid", "value"),
    Input("main-highlight-toggle", "value"),
    Input("main-marker-outline-radio", "value"),
    Input("main-marker-base-size", "value"),
    Input("main-marker-symbol", "value"),
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
