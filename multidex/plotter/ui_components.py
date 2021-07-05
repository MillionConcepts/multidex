"""factory functions for UI dash components"""
import random
from ast import literal_eval
from functools import partial
from typing import TYPE_CHECKING, Mapping, Optional, Iterable

import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import plotly.graph_objects as go

from plotter.proffered_markers import SOLID_MARKER_COLORS, MARKER_SYMBOLS
from multidex_utils import get_if, none_to_empty, fetch_css_variables

if TYPE_CHECKING:
    from plotter.models import Spectrum

# TODO: is this a terrible placeholder?
css_variables = fetch_css_variables()
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
    ],
    "displaylogo": False,
}
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


# note that style properties are camelCased rather than hyphenated in
# compliance with conventions for React virtual DOM


def scale_to_drop(model, element_id, value=None):
    """dropdown for selecting a virtual filter to scale to"""
    return dcc.Dropdown(
        id=element_id,
        options=[{"label": "None", "value": "None"}]
        + [
            {"label": filt + " " + str(wave) + "nm", "value": filt}
            for filt, wave in model.virtual_filters.items()
        ],
        value=value,
        style={"maxWidth": "10rem"},
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
    )
    scale_container.children = [
        html.Label(children=["scale to:"], htmlFor=id_prefix + "-scale"),
        scale_to_drop(spec_model, id_prefix + "-scale", scale_value),
        dcc.Checklist(
            id=id_prefix + "-average",
            options=[
                {"label": "average nearby", "value": "average"},
            ],
            value=[average_value],
        ),
        dcc.Checklist(
            id=id_prefix + "-r-star",
            options=[
                {"label": "R*", "value": "r-star"},
            ],
            value=[r_star_value],
        ),
        html.Label(children=["show error"], htmlFor=id_prefix + "-error"),
        dcc.Dropdown(
            id=id_prefix + "-error",
            options=[
                {"label": "none", "value": "none"},
                {"label": "ROI", "value": "roi"},
                {"label": "instrumental", "value": "instrumental"},
            ],
            value=error_value,
        ),
    ]
    return scale_container


def dynamic_spec_div(
    print_name: str, graph_name: str, image_name: str, index: int
) -> html.Div:
    return html.Div(
        children=[
            html.Pre(
                children=[],
                id={"type": print_name, "index": index},
                style={
                    "marginLeft": "5vw",
                    "width": "15vw",
                    "display": "inline-block",
                    "verticalAlign": "top",
                },
            ),
            html.Div(
                children=[spec_graph(graph_name, index)],
                id={"type": graph_name + "-container", "index": index},
                style={
                    "display": "inline-block",
                },
            ),
            html.Div(
                id={"type": image_name, "index": index},
                style={
                    "display": "inline-block",
                    "maxHeight": "20vw",
                    "paddingTop": "1.5rem",
                    "width": "30vw",
                },
            ),
        ],
        id={"type": "spec-container", "index": index},
        style={"display": "flex"},
    )


# TODO: determine if the following two component factories are necessary at all
def main_graph() -> dcc.Graph:
    """dash component factory for main graph"""
    fig = go.Figure(layout={**GRAPH_DISPLAY_DEFAULTS})
    # noinspection PyTypeChecker
    return dcc.Graph(
        id="main-graph",
        figure=fig,
        style={"height": "65vh"},
        className="graph",
        config=GRAPH_CONFIG_SETTINGS,
    )


def spec_graph(name: str, index: int) -> dcc.Graph:
    """dash component factory for reflectance graphs"""
    fig = go.Figure(layout={**GRAPH_DISPLAY_DEFAULTS})
    # noinspection PyTypeChecker
    return dcc.Graph(
        id={"type": name, "index": index},
        figure=fig,
        style={"height": "20vw", "width": "45vw"},
        config=GRAPH_CONFIG_SETTINGS,
    )


def image_holder(index: int = 0) -> dcc.Graph:
    """dash component factory for zoomable assets images. maybe. placeholder"""
    return dcc.Graph(
        id="image-" + str(index),
    )


def color_scale_drop(element_id: str, value: str = None) -> dcc.Dropdown:
    """
    dropdown for selecting calculation options for marker settings
    """
    options = [
        {"label": colormap, "value": colormap}
        for colormap in px.colors.named_colorscales()
    ]
    if not value:
        value = "haline"
    return dcc.Dropdown(
        id=element_id,
        className="color-drop",
        options=options,
        value=value,
        clearable=False,
    )


def collapse_arrow(id_for, title, off=False):
    if off:
        arrow_style = {
            "WebkitTransform": "rotate(45deg)",
            "transform": "rotate(45deg)",
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
        className="info-text",
        id=element_id + "-container",
        style={
            "display": "flex",
            "flexDirection": "column",
            "fontSize": "1rem",
        },
        children=[
            html.Label(children=[label_content], htmlFor=element_id),
            dcc.Dropdown(
                id=element_id,
                className="axis-value-drop",
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
        options=[
            {"label": field["label"], "value": field["label"]}
            for field in fields
        ],
        value=none_to_empty(value),
    )


def model_options_drop(
    element_id: str, index: int, value: Optional[str] = None
) -> dcc.Dropdown:
    """
    dropdown for selecting search values for a specific field
    could end up getting unmanageable as a UI element
    """
    return dcc.Dropdown(
        id={"type": element_id, "index": index},
        options=[{"label": "any", "value": "any"}],
        multi=True,
        value=none_to_empty(value),
    )


def model_range_entry(
    element_id: str,
    index: int,
    begin: Optional[float] = None,
    end: Optional[float] = None,
) -> list[dcc.Input]:
    """
    pair of entry fields for selecting a range of values for a
    quantitatively-valued field.
    """
    return [
        dcc.Input(
            id={"type": element_id + "-begin", "index": index},
            type="text",
            value=none_to_empty(begin),
        ),
        dcc.Input(
            id={"type": element_id + "-end", "index": index},
            type="text",
            value=none_to_empty(end),
        ),
    ]


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
    elif string == "":
        pass
    else:
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


def model_range_entry_2(
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
    return html.P(
        className="info-text",
        id={"type": element_id, "index": index},
    )


def search_parameter_div(
    index: int, searchable_fields: Iterable[str], preset_parameter=None
) -> html.Div:
    get_r = partial(get_if, preset_parameter is not None, preset_parameter)
    children = [
        html.Label(children=["search field"]),
        field_drop(searchable_fields, "field-search", index, get_r("field")),
        model_options_drop("term-search", index, get_r("term")),
        model_range_display("number-range-display", index),
        model_range_entry_2("number-search", index, preset_parameter),
        html.Div(
            children=[
                dcc.Input(
                    id={"type": "search-load-trigger", "index": index}, value=0
                )
            ],
            style={"display": "none"},
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


def search_container_div(searchable_fields, preset_parameters):
    search_container = html.Div(
        id="search-controls-container",
        className="search-controls-container",
    )
    # list was 'serialized' to string to put it in a single df cell
    if preset_parameters is None:
        preset_parameters = "None"  # doing a slightly goofy thing here
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
            dcc.Dropdown(id=element_id + "-drop"),
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


def axis_controls_container(axis, prefix, spec_model, get_r, filter_options):
    children = [
        axis_value_drop(
            spec_model,
            prefix + "graph-option-" + axis,
            value=get_r(prefix + "graph-option-" + axis + ".value"),
            label_content=axis + " axis",
        ),
        html.Div(
            className="filter-container",
            children=[
                filter_drop(
                    spec_model,
                    prefix + "filter-1-" + axis,
                    value=get_r(prefix + "filter-1-" + axis + ".value"),
                    label_content="left",
                    options=filter_options,
                ),
                filter_drop(
                    spec_model,
                    prefix + "filter-3-" + axis,
                    value=get_r(prefix + "filter-3-" + axis + ".value"),
                    label_content="center",
                    options=filter_options,
                ),
                filter_drop(
                    spec_model,
                    prefix + "filter-2-" + axis,
                    value=get_r(prefix + "filter-2-" + axis + ".value"),
                    label_content="right",
                    options=filter_options,
                ),
                component_drop(
                    prefix + "component-" + axis,
                    value=get_r(prefix + "component-" + axis + ".value"),
                    label_content="component #",
                ),
            ],
        ),
    ]
    return html.Div(className="axis-controls-container", children=children)


def marker_controls_container(axis, prefix, spec_model, get_r, filter_options):
    coloring_type = get_r(prefix + "coloring-type")
    if coloring_type is None:
        coloring_type = "scale"
    solid_color = get_r(prefix + "color-solid")
    if solid_color is None:
        solid_color = "black"
    marker_symbol = get_r(prefix + "marker-symbol")
    if marker_symbol is None:
        marker_symbol = "circle"
    children = [
        html.Div(
            className="axis-controls-container",
            children=[
                axis_controls_container(
                    axis, prefix, spec_model, get_r, filter_options
                )
            ],
        ),
        html.Div(
            style={
                "display": "flex",
                "flexDirection": "column",
                "marginRight": "0.6rem",
            },
            children=[
                dcc.RadioItems(
                    id=prefix + "coloring-type",
                    options=[
                        {"label": "color scale", "value": "scale"},
                        {"label": "solid", "value": "solid"},
                    ],
                    value=coloring_type,
                ),
                color_scale_drop(
                    prefix + "color-scale",
                    value=get_r(prefix + "color-scale.value"),
                ),
                dcc.Dropdown(
                    prefix + "color-solid",
                    className="color-drop",
                    value=solid_color,
                    options=SOLID_MARKER_COLORS,
                ),
                html.Label(
                    children=["marker outlines"],
                    htmlFor=prefix + "marker-outline-radio",
                ),
                dcc.RadioItems(
                    id=prefix + "marker-outline-radio",
                    options=[
                        {
                            "label": "off",
                            "value": "off",
                        },
                        {"label": "black", "value": "rgba(0,0,0,1)"},
                        {"label": "white", "value": "rgba(255,255,255,1)"},
                    ],
                    value="off",
                ),
            ],
        ),
        html.Div(
            style={
                "display": "flex",
                "flexDirection": "column",
                "marginRight": "0.3rem",
            },
            children=[
                html.Label(
                    children=["marker base size"],
                    htmlFor=prefix + "marker-base-size",
                ),
                dcc.RadioItems(
                    id=prefix + "marker-base-size",
                    options=[
                        {"label": "small", "value": 4},
                        {"label": "medium", "value": 9},
                        {"label": "large", "value": 18},
                    ],
                    value=9,
                ),
                html.Label(
                    children=["marker symbol"],
                    htmlFor=prefix + "marker-symbol",
                    style={"marginTop": "0.3rem"},
                ),
                dcc.Dropdown(
                    prefix + "marker-symbol",
                    className="color-drop",
                    value=marker_symbol,
                    options=MARKER_SYMBOLS,
                ),
            ],
        ),
    ]
    return html.Div(
        style={"display": "flex", "flexDirection": "row"},
        children=children,
    )


# primary search panel
# TODO: it is getting very obnoxious to keep track of indentation, even with
#  the level of abstraction currently in use, and it needs to be further
#  refactored -- maybe into a flat list of some kind?
def search_div(
    spec_model: "Spectrum", restore_dictionary: Optional[Mapping] = None
):
    # are we restoring from saved settings? if so, this function gets them;
    # if not, this function politely acts as None
    get_r = partial(get_if, restore_dictionary is not None, restore_dictionary)
    if get_r("average_filters"):
        filts = [
            {"label": filt, "value": filt}
            for filt in spec_model.canonical_averaged_filters
        ]
    else:
        filts = None
    search_children = [
        html.Div(
            className="graph-controls-container",
            children=[
                *collapse(
                    "main-graph-control-container-x",
                    "x axis",
                    axis_controls_container(
                        "x", "main-", spec_model, get_r, filts
                    ),
                ),
                *collapse(
                    "main-graph-control-container-y",
                    "y axis",
                    axis_controls_container(
                        "y", "main-", spec_model, get_r, filts
                    ),
                ),
                *collapse(
                    "main-graph-control-container-marker",
                    "markers",
                    marker_controls_container(
                        "marker", "main-", spec_model, get_r, filts
                    ),
                ),
                *collapse(
                    "highlight-controls",
                    "highlight",
                    html.Div(
                        className="axis-controls-container",
                        children=[
                            html.Button(
                                "set highlight",
                                id="main-highlight-save",
                                style={"marginTop": "1rem"},
                            ),
                            dcc.RadioItems(
                                id="main-highlight-toggle",
                                options=[
                                    {
                                        "label": "highlight on",
                                        "value": "on",
                                    },
                                    {"label": "off", "value": "off"},
                                ],
                                value="off",
                            ),
                            html.P(
                                id="main-highlight-description",
                                style={
                                    "maxWidth": "12rem",
                                },
                            ),
                        ],
                    ),
                    True,
                ),
                *collapse(
                    "search-controls",
                    "search",
                    html.Div(
                        style={
                            "display": "flex",
                            "flexDirection": "row",
                        },
                        children=[
                            search_container_div(
                                spec_model.searchable_fields(),
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
                                        children="Update graph",
                                    ),
                                ],
                            ),
                        ],
                    ),
                ),
                *collapse(
                    "numeric-controls",
                    "scaling",
                    html.Div(
                        children=[
                            html.Div(
                                className="graph-bounds-axis-container",
                                children=[
                                    html.Label(
                                        children=["set bounds"],
                                        htmlFor="main-graph-bounds",
                                    ),
                                    dcc.Input(
                                        type="text",
                                        id="main-graph-bounds",
                                        style={
                                            "height": "1.8rem",
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
                    ),
                    off=True,
                ),
                *collapse(
                    "load-panel",
                    "load",
                    html.Div(load_search_drop("load-search")),
                    off=True,
                ),
                *collapse(
                    "save-panel",
                    "save",
                    html.Div(
                        [
                            save_search_input("save-search"),
                            html.Button(
                                "Export CSV",
                                id="main-export-csv",
                                style={"marginTop": "1rem"},
                            ),
                        ]
                    ),
                    off=True,
                ),
                *collapse(
                    "graph-display-panel",
                    "display",
                    html.Div(
                        children=[
                            html.Label(
                                children=["graph background"],
                                htmlFor="main-graph-bg-color",
                            ),
                            dcc.RadioItems(
                                id="main-graph-bg-radio",
                                options=[
                                    {
                                        "label": "white",
                                        "value": "rgba(255,255,255,1)",
                                    },
                                    {
                                        "label": "light gray",
                                        "value": css_variables["dark-tint-0"],
                                    },
                                    {
                                        "label": "dark gray",
                                        "value": css_variables["dark-tint-1"],
                                    },
                                ],
                                value=css_variables["dark-tint-0"],
                            ),
                            html.Label(
                                children=["gridlines"],
                                htmlFor="main-graph-gridlines-radio",
                            ),
                            dcc.RadioItems(
                                id="main-graph-gridlines-radio",
                                options=[
                                    {"label": "off", "value": "off"},
                                    {
                                        "label": "on",
                                        "value": "on",
                                    },
                                ],
                                value="on",
                            ),
                        ]
                    ),
                    off=True,
                ),
            ],
        ),
        html.Div(children=[main_graph()], id="main-container"),
        dynamic_spec_div(
            "main-spec-print",
            "main-spec-graph",
            "main-spec-image",
            0,
        ),
        html.Div(
            children=[
                scale_controls_container(
                    spec_model,
                    "main-spec",
                    "L1_R1",
                    "r-star",
                    "average",
                    "error",
                ),
            ]
        ),
        # hidden divs for async triggers, dummy outputs, etc
        trigger_div("main-graph-scale", 1),
        trigger_div("search", 2),
        trigger_div("load", 1),
        trigger_div("save", 1),
        trigger_div("highlight", 1),
        html.Div(id="fire-on-load", children="2", style={"display": "none"}),
        html.Div(
            id="fake-output-for-callback-with-only-side-effects-0",
            style={"display": "none"},
        ),
        html.Div(
            id="fake-output-for-callback-with-only-side-effects-1",
            style={"display": "none"},
        ),
        html.Div(id="search-load-progress-flag"),
    ]
    return html.Div(
        children=search_children,
        # as opposed to regular DOM id
        id="search-div",
    )


def multidex_body(spec_model):
    """top-level "body" div of application"""
    # noinspection PyTypeChecker
    return html.Div(
        children=[
            search_div(spec_model),
            dcc.Interval(id="interval1", interval=1000, n_intervals=0),
        ],
        id="multidex",
    )
