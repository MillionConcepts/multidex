"""factory functions for UI dash components"""
import random
from ast import literal_eval
from functools import partial
from typing import Mapping, Optional, Iterable, Callable

import plotly.express as px
import plotly.graph_objects as go
from dash import dcc
from dash import html
from dash.html import Div

from multidex_utils import get_if, none_to_empty
from plotter.colors import get_plotly_colorscales
from plotter.styles.graph_style import (
    GRAPH_DISPLAY_DEFAULTS,
    GRAPH_CONFIG_SETTINGS,
    css_variables,
)
from plotter.styles.marker_style import SOLID_MARKER_COLORS, MARKER_SYMBOLS


# note that style properties are camelCased rather than hyphenated in
# compliance with conventions for React virtual DOM
def scale_to_drop(model, element_id, value=None):
    """dropdown for selecting a virtual filter to scale to"""
    return dcc.Dropdown(
        id=element_id,
        className="medium-drop",
        options=[{"label": "None", "value": "None"}]
        + [
            {"label": filt + " " + str(wave) + "nm", "value": filt}
            for filt, wave in model.virtual_filters.items()
        ],
        value=value,
        style={"maxWidth": "8rem"},
    )


def scale_controls_container(
    spec_model,
    id_prefix,
    scale_value=None,
    r_star_value=None,
    average_value=None,
    error_value="none",
):
    # TODO: this is a messy way to handle weird cases in loading.
    #  it should be cleaned up.
    if scale_value is None:
        scale_value = "None"
    if r_star_value is None:
        r_star_value = "None"
    if average_value in [None, False, ""]:
        average_value = ""
    else:
        average_value = "average"
    scale_container = html.Div(
        id=id_prefix + "-scale-controls-container-div",
        className="scale-controls-container",
        style={
            "display": "flex",
            "flexDirection": "row",
            "marginRight": "0.3rem",
        },
    )
    scale_container.children = [
        html.Div(
            style={
                "display": "flex",
                "flexDirection": "column",
                "marginRight": "0.3rem",
            },
            children=[
                dcc.Checklist(
                    id=id_prefix + "-average",
                    className="info-text",
                    options=[
                        {"label": "merge", "value": "average"},
                    ],
                    value=[average_value],
                ),
                html.Label(
                    className="info-text",
                    children=["scale to:"],
                    htmlFor=id_prefix + "-scale",
                ),
                scale_to_drop(spec_model, id_prefix + "-scale", scale_value),
            ],
        ),
        html.Div(
            style={"display": "flex", "flexDirection": "column"},
            children=[
                dcc.Checklist(
                    id=id_prefix + "-r-star",
                    className="info-text",
                    options=[
                        {"label": "R*", "value": "r-star"},
                    ],
                    value=[r_star_value],
                ),
                html.Label(
                    children=["show error"],
                    className="info-text",
                    htmlFor=id_prefix + "-error",
                ),
                dcc.Dropdown(
                    id=id_prefix + "-error",
                    className="medium-drop",
                    options=[
                        {"label": "None", "value": "none"},
                        {"label": "ROI", "value": "roi"},
                        {"label": "inst.", "value": "instrumental"},
                    ],
                    value=error_value,
                ),
            ],
        ),
    ]
    return scale_container


def dynamic_spec_div() -> html.Div:
    return html.Div(
        id="spec-container",
        style={
            "display": "flex",
            "flexDirection": "column",
            "height": "100%",
            "width": "33%",
        },
        children=[
            html.Div(
                children=[spec_graph("spec-graph")],
                id="spec-graph-container",
                style={
                    "display": "inline-block",
                    "width": "100%",
                    "height": "50%",
                },
            ),
            html.Div(
                id="spec-image",
                style={
                    "display": "flex",
                    "flexDirection": "row",
                    "justifyContent": "center",
                    "height": "48%",
                    "marginTop": "1%",
                    "marginRight": "3%",
                    "marginLeft": "1%",
                    "width": "96%",
                },
            ),
        ],
    )


# def dynamic_spec_div(
#     print_name: str, graph_name: str, image_name: str, index: int
# ) -> html.Div:
#     return html.Div(
#         children=[
#             html.Pre(
#                 children=[],
#                 id={"type": print_name, "index": index},
#                 style={
#                     "marginLeft": "5vw",
#                     "width": "15vw",
#                     "display": "inline-block",
#                     "verticalAlign": "top",
#                 },
#             ),
#             html.Div(
#                 children=[spec_graph(graph_name, index)],
#                 id={"type": graph_name + "-container", "index": index},
#                 style={
#                     "display": "inline-block",
#                 },
#             ),
#             html.Div(
#                 id={"type": image_name, "index": index},
#                 style={
#                     "display": "inline-block",
#                     "maxHeight": "20vw",
#                     "paddingTop": "1.5rem",
#                     "width": "30vw",
#                 },
#             ),
#         ],
#         id={"type": "spec-container", "index": index},
#         style={"display": "flex"},
#     )


def main_graph(style) -> dcc.Graph:
    """dash component factory for main graph"""
    fig = go.Figure(layout={**GRAPH_DISPLAY_DEFAULTS})
    # noinspection PyTypeChecker
    return dcc.Graph(
        id="main-graph",
        figure=fig,
        style=style,
        className="graph",
        config=GRAPH_CONFIG_SETTINGS,
    )


def spec_graph(name: str) -> dcc.Graph:
    """dash component factory for reflectance graphs"""
    fig = go.Figure(layout={**GRAPH_DISPLAY_DEFAULTS})
    # noinspection PyTypeChecker
    return dcc.Graph(
        id=name,
        figure=fig,
        style={"height": "100%", "width": "100%"},
        config=GRAPH_CONFIG_SETTINGS,
    )


def image_holder(index: int = 0) -> dcc.Graph:
    """dash component factory for zoomable assets images. maybe. placeholder"""
    return dcc.Graph(id="image-" + str(index))


def color_scale_drop(element_id: str, value: str = None) -> dcc.Dropdown:
    """
    dropdown for selecting calculation options for marker settings
    """
    options = [
        {"label": colormap, "value": colormap}
        for colormap in get_plotly_colorscales().keys()
    ]
    if not value:
        value = "haline"
    return dcc.Dropdown(
        id=element_id,
        className="filter-drop medium-drop",
        options=options,
        value=value,
        clearable=False,
    )


def collapse_arrow(id_for, title, off=False):
    if off:
        arrow_style = {
            "WebkitTransform": "rotate(45deg)",
        }
        text_style = {"display": "inline-block"}
    else:
        arrow_style = None
        text_style = None
    return html.Div(
        id={"type": "collapse-div", "index": id_for},
        className="collapse-div",
        children=[
            html.P(
                className="arrow",
                id={"type": "collapse-arrow", "index": id_for},
                style=arrow_style,
            ),
            html.P(
                className="collapse-text",
                id={"type": "collapse-text", "index": id_for},
                children=[title],
                style=text_style,
            ),
        ],
    )


def collapse(collapse_id, title, component=html.Div(), off=False):
    style_dict = {}
    if off is True:
        style_dict = {"display": "none"}
    return (
        collapse_arrow(collapse_id, title, off),
        html.Div(
            id={
                "type": "collapsible-panel",
                "index": collapse_id,
            },
            className="collapsible-panel",
            style=style_dict,
            children=[component],
        ),
    )


def axis_value_drop(spec_model, element_id, value=None, label_content=None):
    """
    dropdown for selecting calculation options for axes
    """
    options = [
        {"label": option["label"], "value": option["value"]}
        for option in spec_model.graphable_properties()
    ]
    if not value:
        if "marker" in element_id:
            value = "feature"
        else:
            value = "ratio"
    return html.Div(
        className="axis-title-text",
        id=element_id + "-container",
        style={
            "display": "flex",
            "flexDirection": "column",
        },
        children=[
            html.Label(children=[label_content], htmlFor=element_id),
            dcc.Dropdown(
                id=element_id,
                className="axis-value-drop medium-drop",
                options=options,
                value=value,
                clearable=False,
            ),
        ],
    )


def filter_drop(model, element_id, value, label_content=None, options=None):
    """dropdown for filter selection"""
    if options is None:
        options = [
            {"label": filt + " " + str(wave) + "nm", "value": filt}
            for filt, wave in model.filters.items()
        ]
    if not value:
        value = random.choice(options)["value"]

    classnames = ["dash-dropdown", "filter-drop", "medium-drop"]
    if label_content == "right":
        classnames.append("right-filter-drop")
    return html.Div(
        className="info-text",
        id=element_id + "-container",
        style={"display": "flex", "flexDirection": "column"},
        children=[
            html.Label(children=[label_content], htmlFor=element_id),
            dcc.Dropdown(
                id=element_id,
                options=options,
                value=value,
                className=" ".join(classnames),
                clearable=False
                # style={"width": "6rem", "display": "inline-block"},
                # style={"display":"inline-block"}
            ),
        ],
    )


def component_drop(element_id, value, label_content=None, options=None):
    if options is None:
        options = [
            {"label": str(component_ix + 1), "value": component_ix}
            for component_ix in range(8)
        ]
    if not value:
        value = 0
    if label_content is None:
        label_content = "component #"
    return html.Div(
        className="info-text",
        id=element_id + "-container",
        style={"display": "flex", "flexDirection": "column"},
        children=[
            html.Label(children=[label_content], htmlFor=element_id),
            dcc.Dropdown(
                id=element_id,
                options=options,
                value=value,
                className="dash-dropdown filter-drop",
                clearable=False,
            ),
        ],
    )


def field_drop(fields, element_id, index, value=None):
    """dropdown for field selection -- no special logic atm"""
    return dcc.Dropdown(
        id={"type": element_id, "index": index},
        className="medium-drop",
        options=[
            {"label": field["label"], "value": field["label"]}
            for field in fields
        ],
        value=none_to_empty(value),
    )


def model_options_drop(
    element_id: str,
    index: int,
    value: Optional[str] = None,
    className="medium-drop",
) -> dcc.Dropdown:
    """
    dropdown for selecting search values for a specific field
    could end up getting unmanageable as a UI element
    """
    # TODO: hacky.
    if value is not None:
        loaded_values = [{"label": item, "value": item} for item in value]
    else:
        loaded_values = []
    return dcc.Dropdown(
        id={"type": element_id, "index": index},
        options=[{"label": "any", "value": "any"}] + loaded_values,
        className=className,
        multi=True,
        value=none_to_empty(value),
    )


def parse_model_quant_entry(string: str) -> dict:
    value_dict = {}
    is_range = "--" in string
    is_list = "," in string
    if is_range and is_list:
        raise ValueError(
            "Entering both an explicit value list and a value range is "
            "currently not supported."
        )
    if is_range:
        range_list = string.split("--")
        if len(range_list) > 2:
            # try:
            raise ValueError(
                "Entering a value range with more than two numbers is "
                "currently not supported."
            )
        # allow either a blank beginning or end, but not both
        try:
            value_dict["begin"] = float(range_list[0])
        except ValueError:
            value_dict["begin"] = ""
        try:
            value_dict["end"] = float(range_list[1])
        except ValueError:
            value_dict["end"] = ""
        if not (value_dict["begin"] or value_dict["end"]):
            raise ValueError(
                "Either a beginning or end numerical value must be entered."
            )
    elif string != "":
        list_list = string.split(",")
        # do not allow ducks and rutabagas and such to be entered into the list
        try:
            value_dict["value_list"] = [float(item) for item in list_list]
        except ValueError:
            raise ValueError(
                "Non-numerical lists are currently not supported."
            )
    return value_dict


def unparse_model_quant_entry(value_dict: Mapping) -> str:
    if value_dict is None:
        text = ""
    elif ("value_list" in value_dict.keys()) and (
        ("begin" in value_dict.keys()) or ("end" in value_dict.keys())
    ):
        raise ValueError(
            "Entering both an explicit value list and a value range is "
            "currently not supported."
        )
    elif "value_list" in value_dict.keys():
        text = ",".join([str(val) for val in value_dict["value_list"]])
    elif ("begin" in value_dict.keys()) or ("end" in value_dict.keys()):
        text = str(value_dict["begin"]) + " -- " + str(value_dict["end"])
    else:
        text = ""
    return text


def model_range_entry(
    element_id: str, index: int, value_dict: Optional[Mapping] = None
) -> dcc.Input:
    """
    entry field for selecting a range of values for a
    quantitatively-valued field.
    """
    return dcc.Input(
        id={"type": element_id, "index": index},
        type="text",
        value=unparse_model_quant_entry(value_dict),
        style={"display": "none"},
    )


def model_range_display(element_id: str, index: int) -> html.P:
    """placeholder area for displaying range for number field searches"""
    return html.Span(
        className="tooltiptext",
        id={"type": element_id, "index": index},
    )


def search_parameter_div(
    index: int, searchable_fields: Iterable[str], preset_parameter=None
) -> html.Div:
    get_r = partial(get_if, preset_parameter is not None, preset_parameter)
    children = [
        html.Label(children=["search field"], className="axis-title-text"),
        field_drop(searchable_fields, "field-search", index, get_r("field")),
        model_options_drop(
            "term-search",
            index,
            value=get_r("term"),
            className="medium-drop term-search",
        ),
        html.Div(
            className="tooltipped",
            children=[
                model_range_display("number-range-display", index),
                model_range_entry("number-search", index, preset_parameter),
            ],
        ),
    ]
    if index != 0:
        children.append(
            html.Button(
                id={"type": "remove-param", "index": index},
                children="remove parameter",
            )
        ),
    else:
        children.append(
            html.Button("add parameter", id="add-param"),
        )
    return html.Div(
        className="search-parameter-container",
        children=children,
        id={"type": "search-parameter-div", "index": index},
    )


def search_container_div(spec_model, preset_parameters):
    search_container = html.Div(
        id="search-controls-container",
        className="search-controls-container",
    )
    # list was 'serialized' to string to put it in a single df cell
    if preset_parameters is None:
        preset_parameters = "None"  # doing a slightly goofy thing here
    searchable_fields = spec_model.searchable_fields()
    if literal_eval(preset_parameters) is not None:
        search_container.children = [
            search_parameter_div(ix, searchable_fields, parameter)
            for ix, parameter in enumerate(literal_eval(preset_parameters))
        ]
    else:
        search_container.children = [
            search_parameter_div(0, searchable_fields)
        ]
    return search_container


def trigger_div(prefix, number_of_triggers):
    """hidden div for semi-asynchronous callback triggers"""
    return html.Div(
        children=[
            dcc.Input(
                id={"type": prefix + "-trigger", "index": index}, value=0
            )
            for index in range(number_of_triggers)
        ],
        style={"display": "none"},
        id=prefix + "-trigger-div",
    )


def load_search_drop(element_id):
    return html.Div(
        className="load-button-container",
        children=[
            html.Label(children=["search name"], htmlFor=element_id + "-drop"),
            dcc.Dropdown(id=element_id + "-drop", className="medium-drop"),
            html.Button(
                id=element_id + "-load-button",
                children="load",
            ),
        ],
    )


def save_search_input(element_id):
    return html.Div(
        className="save-button-container",
        children=[
            html.Label(
                children=["save as"], htmlFor=element_id + "-name-input"
            ),
            dcc.Input(id=element_id + "-name-input", type="text"),
            html.Button(id=element_id + "-save-button", children="save"),
        ],
        style={"display": "flex", "flexDirection": "column"},
    )


def axis_controls_container(
    axis: str, spec_model, get_r: Callable, filter_options
) -> Div:
    children = [
        axis_value_drop(
            spec_model,
            "graph-option-" + axis,
            value=get_r("graph-option-" + axis + ".value"),
            label_content=axis + " axis",
        ),
        html.Div(
            className="filter-container",
            children=[
                filter_drop(
                    spec_model,
                    "filter-1-" + axis,
                    value=get_r("filter-1-" + axis + ".value"),
                    label_content="left",
                    options=filter_options,
                ),
                filter_drop(
                    spec_model,
                    "filter-3-" + axis,
                    value=get_r("filter-3-" + axis + ".value"),
                    label_content="center",
                    options=filter_options,
                ),
                filter_drop(
                    spec_model,
                    "filter-2-" + axis,
                    value=get_r("filter-2-" + axis + ".value"),
                    label_content="right",
                    options=filter_options,
                ),
                component_drop(
                    "component-" + axis,
                    value=get_r("component-" + axis + ".value"),
                    label_content="component #",
                ),
            ],
        ),
    ]
    return html.Div(className="axis-controls-container", children=children)


def marker_coloring_type_div(coloring_type: str) -> Div:
    return html.Div(
        style={
            "display": "flex",
            "flexDirection": "row",
        },
        children=[
            html.Label(
                className="info-text",
                children=["color: "],
                htmlFor="marker-outline-radio",
            ),
            dcc.RadioItems(
                id="coloring-type",
                className="radio-items",
                options=[
                    {"label": "scale", "value": "scale"},
                    {"label": "solid", "value": "solid"},
                ],
                value=coloring_type,
            ),
        ],
    )


def marker_size_div(marker_size) -> Div:
    return html.Div(
        style={
            "display": "flex",
            "flexDirection": "row",
        },
        children=[
            html.Label(
                children=["size: "],
                className="info-text",
                htmlFor="marker-base-size",
            ),
            dcc.RadioItems(
                id="marker-base-size",
                className="radio-items",
                options=[
                    {"label": "s", "value": 4},
                    {"label": "m", "value": 9},
                    {"label": "l", "value": 18},
                ],
                value=marker_size,
            ),
        ],
    )


def marker_outline_div(outline_color) -> Div:
    return html.Div(
        style={
            "display": "flex",
            "flexDirection": "row",
            "marginTop": "1.7rem",
        },
        children=[
            html.Label(
                className="info-text",
                children=["outline: "],
                htmlFor="marker-outline-radio",
            ),
            dcc.RadioItems(
                id="marker-outline-radio",
                className="radio-items",
                options=[
                    {
                        "label": "off",
                        "value": "off",
                    },
                    {"label": "b", "value": "rgba(0,0,0,1)"},
                    {
                        "label": "w",
                        "value": "rgba(255,255,255,1)",
                    },
                ],
                value=outline_color,
            ),
        ],
    )


def marker_clip_div(get_r: Callable) -> Div:
    high = get_r("color-clip-bound-high")
    if high is None:
        high = 100
    low = get_r("color-clip-bound-low")
    if low is None:
        low = 0
    return html.Div(
        [
            html.Label(
                children=["color clip"],
                className="axis-title-text",
                # TODO: wrap this more nicely
                htmlFor="color-clip-bound-low",
            ),
            dcc.Input(
                type="number",
                id="color-clip-bound-low",
                style={
                    "height": "1.4rem",
                    "width": "3rem",
                },
                value=low,
                min=0,
                max=100,
            ),
            dcc.Input(
                type="number",
                id="color-clip-bound-high",
                style={
                    "height": "1.4rem",
                    "width": "3rem",
                },
                value=high,
                min=0,
                max=100,
            ),
        ],
        style={
            "display": "flex",
            "flexDirection": "column",
            "marginRight": "0.3rem",
            "marginLeft": "0.3rem",
        },
    )


def marker_options_div(get_r: Callable) -> Div:
    coloring_type = get_r("coloring-type.value")
    if coloring_type is None:
        coloring_type = "scale"
    outline_color = get_r("marker-outline-radio.value")
    if outline_color is None:
        outline_color = "off"
    marker_size = get_r("marker-base-size.value")
    if marker_size is None:
        marker_size = 9
    return html.Div(
        id="marker-options-div",
        style={
            "display": "flex",
            "flexDirection": "column",
            "marginRight": "0.3rem",
        },
        children=[
            marker_outline_div(outline_color),
            marker_size_div(marker_size),
            marker_coloring_type_div(coloring_type),
        ],
    )


def marker_color_symbol_div(get_r: Callable) -> Div:
    marker_symbol = get_r("marker-symbol.value")
    if marker_symbol is None:
        marker_symbol = "circle"
    solid_color = get_r("color-solid.value")
    if solid_color is None:
        solid_color = "black"
    return html.Div(
        id="marker-color-symbol-container",
        style={"display": "flex", "flexDirection": "column", "width": "8rem"},
        className="axis-controls-container",
        children=[
            html.Label(
                children=["color"],
                htmlFor="color-scale",
                className="info-text",
            ),
            color_scale_drop(
                "color-scale",
                value=get_r("color-scale.value"),
            ),
            dcc.Dropdown(
                "color-solid",
                className="filter-drop medium-drop",
                value=solid_color,
                options=SOLID_MARKER_COLORS,
            ),
            html.Label(
                children=["marker symbol"],
                htmlFor="marker-symbol",
                className="info-text",
            ),
            dcc.Dropdown(
                "marker-symbol",
                className="medium-drop filter-drop",
                value=marker_symbol,
                options=MARKER_SYMBOLS,
            ),
        ],
    )


def search_controls_div(spec_model, get_r: Callable) -> html.Div:
    return html.Div(
        style={
            "display": "flex",
            "flexDirection": "row",
        },
        children=[
            search_container_div(
                spec_model,
                get_r("search_parameters"),
            ),
            html.Div(
                className="search-button-container",
                children=[
                    # hidden trigger for queryset
                    # update on dropdown removal
                    html.Button(
                        id={
                            "type": "submit-search",
                            "index": 1,
                        },
                        style={"display": "none"},
                    ),
                    html.Button(
                        "clear search",
                        id="clear-search",
                    ),
                    html.Button(
                        id={
                            "type": "submit-search",
                            "index": 0,
                        },
                        children="update graph",
                    ),
                ],
            ),
        ],
    )


def display_controls_div(get_r: Callable) -> html.Div:
    if get_r("plot_bgcolor") is None:
        bg_color = css_variables["dark-tint-0"]
    else:
        bg_color = get_r("plot_bgcolor")
    # TODO: these inconsistent variable names smell bad
    if get_r("showgrid") is None:
        gridlines_color = css_variables["dark-tint-0"]
    elif get_r("showgrid") is False:
        gridlines_color = False
    else:
        gridlines_color = get_r("gridcolor")
    return html.Div(
        children=[
            html.Label(
                children=["graph background"],
                className="info-text",
                htmlFor="main-graph-bg-color",
            ),
            dcc.RadioItems(
                id="main-graph-bg-radio",
                className="radio-items",
                options=[
                    {
                        "label": "white",
                        "value": "rgba(255,255,255,1)",
                    },
                    {
                        "label": "light",
                        "value": css_variables["dark-tint-0"],
                    },
                    {
                        "label": "dark",
                        "value": css_variables["dark-tint-2"],
                    },
                ],
                value=bg_color,
            ),
            html.Label(
                children=["gridlines"],
                className="info-text",
                htmlFor="main-graph-gridlines-radio",
            ),
            dcc.RadioItems(
                id="main-graph-gridlines-radio",
                className="radio-items",
                options=[
                    # note: setting value to Python False causes slightly-bad
                    # under-the-hood behavior in React
                    {"label": "off", "value": "off"},
                    {"label": "light", "value": css_variables["dark-tint-0"]},
                    {"label": "dark", "value": css_variables["dark-tint-2"]},
                ],
                value=gridlines_color,
            ),
            html.Button("clear labels", id="clear-labels"),
        ]
    )


def highlight_controls_div(get_r: Callable) -> html.Div:
    if get_r("highlight-toggle.value") is None:
        highlight = "off"
    else:
        highlight = get_r("highlight-toggle.value")
    return html.Div(
        className="axis-controls-container",
        children=[
            html.Button(
                "set highlight",
                id="highlight-save",
                style={"marginTop": "1rem"},
            ),
            dcc.RadioItems(
                id="highlight-toggle",
                className="info-text",
                options=[
                    {
                        "label": "highlight on",
                        "value": "on",
                    },
                    {"label": "off", "value": "off"},
                ],
                value=highlight,
            ),
            html.P(
                id="highlight-description",
                className="info-text",
                style={
                    "maxWidth": "12rem",
                },
            ),
        ],
    )


def scale_control_div(spec_model, get_r: Callable) -> html.Div:
    return html.Div(
        children=[
            html.Div(
                className="graph-bounds-axis-container",
                children=[
                    html.Label(
                        children=["set bounds"],
                        className="axis-title-text",
                        htmlFor="main-graph-bounds",
                    ),
                    dcc.Input(
                        type="text",
                        id="main-graph-bounds",
                        style={
                            "height": "1.4rem",
                            "width": "10rem",
                        },
                        placeholder="xmin xmax ymin ymax",
                    ),
                ],
                style={
                    "display": "flex",
                    "flexDirection": "column",
                    "marginRight": "0.3rem",
                    "marginLeft": "0.3rem",
                },
            ),
            scale_controls_container(
                spec_model,
                "main-graph",
                scale_value=get_r("scale_to"),
                average_value=get_r("average_filters"),
                # TODO: fix init issue, need extra layer
                #  somewhere
                r_star_value="r-star",
            ),
        ]
    )
