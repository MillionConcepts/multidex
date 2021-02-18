"""factory functions for dash components."""

from ast import literal_eval
from functools import partial
import re
from typing import TYPE_CHECKING, Mapping, Optional, Iterable

import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from plotter_utils import get_if, none_to_empty

if TYPE_CHECKING:
    from plotter.models import Spectrum


# TODO: is this a terrible placeholder?
def fetch_css_variables(css_file: str = "assets/main.css") -> dict[str, str]:
    css_variable_dictionary = {}
    with open(css_file) as stylesheet:
        css_lines = stylesheet.readlines()
    for line in css_lines:
        if not re.match(r"\s+--", line):
            continue
        key, value = re.split(r":\s+", line)
        key = re.sub(r"(--)|[ :]", "", key)
        value = re.sub(r"[ \n;]", "", value)
        css_variable_dictionary[key] = value
    return css_variable_dictionary


css_variables = fetch_css_variables()
GRAPH_COLOR_SETTINGS = {
    "plot_bgcolor": css_variables["dark-tint-0"],
    "paper_bgcolor": css_variables["clean-parchment"],
}
GRAPH_AXIS_SETTINGS = {
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
}


# note that style properties are camelCased rather than hyphenated
# b/c React


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
                    "max-height": "20vw",
                    "padding-top": "1.5rem",
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
    fig = go.Figure()
    # noinspection PyTypeChecker
    fig.update_layout(
        {
            "margin": {"l": 10, "r": 10, "t": 25, "b": 0},
        }
        | GRAPH_COLOR_SETTINGS
    )
    return dcc.Graph(
        id="main-graph",
        figure=fig,
        style={"height": "65vh"},
        className="graph",
    )


def spec_graph(name: str, index: int) -> dcc.Graph:
    """dash component factory for reflectance graphs"""
    fig = go.Figure()
    # noinspection PyTypeChecker
    fig.update_layout(
        {
            "margin": {"l": 10, "r": 10, "t": 25, "b": 0},
        }
        | GRAPH_COLOR_SETTINGS
    )
    return dcc.Graph(
        id={"type": name, "index": index},
        figure=fig,
        style={"height": "20vw", "width": "45vw"},
    )


def image_holder(index: int = 0) -> dcc.Graph:
    """dash component factory for zoomable static images. maybe. placeholder"""
    return dcc.Graph(
        id="image-" + str(index),
    )


def main_graph_scatter(
    x_axis: list[float],
    y_axis: list[float],
    marker_property_dict: Mapping,
    text: list,
    customdata: list,
    zoom: Optional[tuple[list[float, float]]] = None,
) -> go.Figure:
    """
    partial placeholder scatter function for main graph.
    this function just creates the Plotly figure; data is read from db
    and formatted in make_axis.
    """
    fig = go.Figure()
    # TODO: go.Scattergl (WebGL) is noticeably worse-looking than
    # go.Scatter (SVG), but go.Scatter may be inadequately performant with all the points
    # in the data set. can we optimize a bit? hard with plotly...
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=y_axis,
            # change this to be hella popup text
            text=text,
            customdata=customdata,
            mode="markers",
            marker={"color": "black", "size": 8},
        )
    )

    # noinspection PyTypeChecker
    fig.update_layout(
        {
            "margin": {"l": 10, "r": 10, "t": 25, "b": 0},
        }
        | GRAPH_COLOR_SETTINGS
    )
    fig.update_xaxes(GRAPH_AXIS_SETTINGS)
    fig.update_yaxes(GRAPH_AXIS_SETTINGS)
    fig.update_traces(**marker_property_dict)
    if zoom is not None:
        fig.update_layout(
            {
                "xaxis": {"range": [zoom[0][0], zoom[0][1]]},
                "yaxis": {"range": [zoom[1][0], zoom[1][1]]},
            }
        )
    return fig


def mspec_graph_line(spectrum: "Spectrum") -> go.Figure:
    """
    placeholder line graph for individual mastcam spectra.
    creates a plotly figure from the mspec's filter values and
    roi_color.
    """
    spectrum_data = spectrum.as_dict()
    x_axis = list(spectrum_data.keys())
    y_axis = list(spectrum_data.values())
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=y_axis,
            mode="lines+markers",
            line={"color": spectrum.roi_hex_code()},
        )
    )
    # noinspection PyTypeChecker
    fig.update_layout(
        {"margin": {"l": 10, "r": 10, "t": 25, "b": 0}} | GRAPH_COLOR_SETTINGS
    )
    fig.update_xaxes(GRAPH_AXIS_SETTINGS)
    fig.update_yaxes(GRAPH_AXIS_SETTINGS)
    return fig


def marker_options_drop(
    spec_model: "Spectrum", element_id: str, value: str = None
) -> dcc.Dropdown:
    """
    dropdown for selecting calculation options for marker settings
    """
    options = [
        {"label": option["label"], "value": option["value"]}
        for option in spec_model.marker_value_properties
    ]
    if not value:
        value = options[0]["value"]
    return dcc.Dropdown(
        id=element_id,
        className="marker-value-drop",
        options=options,
        value=value,
    )


def color_drop(
    element_id: str, value: str = None
) -> dcc.Dropdown:
    """
    dropdown for selecting calculation options for marker settings
    """
    options = [
        {"label": colormap, "value": colormap}
        for colormap in px.colors.named_colorscales()
    ]
    if not value:
        value = options[0]["value"]
    return dcc.Dropdown(
        id=element_id,
        className="color-drop",
        options=options,
        value=value,
    )


def axis_value_drop(spec_model, element_id, value=None):
    """
    dropdown for selecting calculation options for axes
    """
    options = [
        {"label": option["label"], "value": option["value"]}
        for option in spec_model.marker_value_properties
    ]
    if not value:
        value = options[0]["value"]
    return dcc.Dropdown(
        id=element_id,
        className="axis-value-drop",
        options=options,
        value=value,
    )


def filter_drop(model, element_id, value):
    """dropdown for filter selection"""
    options = [{"label": filt, "value": filt} for filt in model.filters]
    if not value:
        value = options[0]["value"]
    return dcc.Dropdown(
        id=element_id,
        options=options,
        value=value,
        style={"width": "10rem", "display": "inline-block"},
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
    is_range = "-" in string
    is_list = "," in string
    if is_range and is_list:
        raise ValueError(
            "Entering both an explicit value list and a value range is "
            "currently not supported."
        )
    if is_range:
        range_list = string.split("-")
        if len(range_list) > 2:
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
        text = ",".join(value_dict["value_list"])
    elif ("begin" in value_dict.keys()) or ("end" in value_dict.keys()):
        text = value_dict["begin"] + "-" + value_dict["end"]
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
    )


def model_range_display(element_id: str, index: int) -> html.P:
    """placeholder area for displaying range for number field searches"""
    return html.P(id={"type": element_id, "index": index})


def search_parameter_div(
    index: int, searchable_fields: Iterable[str], preset_parameter=None
) -> html.Div:
    get_r = partial(get_if, preset_parameter is not None, preset_parameter)
    children = [
        field_drop(searchable_fields, "field-search", index, get_r("field")),
        model_options_drop("term-search", index, get_r("term")),
        model_range_entry_2("number-search", index, get_r("value_dict")),
        model_range_display("number-range-display", index),
    ]
    if index != 0:
        children.append(
            html.Button(
                id={"type": "remove-param", "index": index},
                children="remove parameter",
            )
        ),
    return html.Div(
        className="search-parameter-container",
        children=children,
        id={"type": "search-parameter-div", "index": index},
    )


def search_container_div(searchable_fields, preset_parameters):
    search_container = html.Div(
        id="search-controls-container-div",
        className="search-controls-container",
    )
    # list was 'serialized' to string to put it in a single df cell
    if preset_parameters is None:
        preset_parameters = "None" # doing a slightly goofy thing here
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


def viewer_tab(index, splot):
    return dcc.Tab(
        children=[
            html.Div(
                children=[
                    dcc.Graph(
                        id={"type": "view-graph", "index": index},
                        style={"height": "80vh"},
                        figure=splot.graph(),
                    )
                ],
                id={"type": "view-graph-container", "index": index},
            ),
            dynamic_spec_div(
                "view-spec-print", "view-spec-graph", "view-spec-image", index
            ),
            html.Div(
                children=[str(splot.settings())],
                id={"type": "view-settings", "index": index},
            ),
            html.Button(
                id={"type": "tab-close-button", "index": index},
                children="close this tab",
            ),
        ],
        label="GRAPH VIEWER " + str(index),
        value="viewer_tab_" + str(index),
        id={"type": "viewer-tab", "index": index},
    )


def trigger_div(prefix, number_of_triggers):
    """hidden div for semi-asynchronous callback triggers"""
    return html.Div(
        children=[
            dcc.Input(
                id={"type": prefix + "-trigger", "index": index}, value=""
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
            dcc.Dropdown(id=element_id + "-drop"),
            html.Button(id=element_id + "-load-button", children="load"),
            html.Button(id=element_id + "-save-button", children="save"),
        ],
    )


# primary search panel
def search_tab(
    spec_model: "Spectrum", restore_dictionary: Optional[Mapping] = None
):
    # are we restoring from saved settings? if so, this function gets them;
    # if not, this function politely acts as None
    get_r = partial(get_if, restore_dictionary is not None, restore_dictionary)
    return dcc.Tab(
        children=[
            html.Div(
                className="graph-controls-container",
                children=[
                    html.Div(
                        className="axis-controls-container",
                        children=[
                            axis_value_drop(
                                spec_model,
                                "main-graph-option-x",
                                value=get_r("graph-option-x"),
                            ),
                            filter_drop(
                                spec_model,
                                "main-filter-1-x",
                                value=get_r("main-filter-1-x.value"),
                            ),
                            filter_drop(
                                spec_model,
                                "main-filter-3-x",
                                value=get_r("main-filter-3-x.value"),
                            ),
                            filter_drop(
                                spec_model,
                                "main-filter-2-x",
                                value=get_r("main-filter-2-x.value"),
                            ),
                        ],
                    ),
                    html.Div(
                        className="axis-controls-container",
                        children=[
                            axis_value_drop(
                                spec_model,
                                "main-graph-option-y",
                                value=get_r("main-graph-option-y.value"),
                            ),
                            filter_drop(
                                spec_model,
                                "main-filter-1-y",
                                value=get_r("main-filter-1-y.value"),
                            ),
                            filter_drop(
                                spec_model,
                                "main-filter-3-y",
                                value=get_r("main-filter-3-y.value"),
                            ),
                            filter_drop(
                                spec_model,
                                "main-filter-2-y",
                                value=get_r("main-filter-2-y.value"),
                            ),
                        ],
                    ),
                    html.Div(
                        className="axis-controls-container",
                        children=[
                            marker_options_drop(
                                spec_model,
                                "main-graph-option-marker",
                                value=get_r("main-graph-option-marker.value"),
                            ),
                            filter_drop(
                                spec_model,
                                "main-filter-1-marker",
                                value=get_r("main-filter-1-marker.value"),
                            ),
                            filter_drop(
                                spec_model,
                                "main-filter-3-marker",
                                value=get_r("main-filter-3-marker.value"),
                            ),
                            filter_drop(
                                spec_model,
                                "main-filter-2-marker",
                                value=get_r("main-filter-2-marker.value"),
                            ),
                            color_drop(
                                'main-color',
                                value=get_r("main-color.value"),
                            )
                        ],
                    ),
                    trigger_div("search", 2),
                    trigger_div("load", 1),
                    search_container_div(
                        spec_model.searchable_fields,
                        get_r("search_parameters"),
                    ),
                    html.Div(
                        className="search-button-container",
                        children=[
                            # hidden trigger for queryset update on dropdown
                            # removal
                            html.Button(
                                id={"type": "submit-search", "index": 1},
                                style={"display": "none"},
                            ),
                            html.Button(
                                "add search parameter", id="add-param"
                            ),
                            html.Button(
                                id={"type": "submit-search", "index": 0},
                                children="Update graph",
                            ),
                            html.Button(
                                id="viewer-open-button",
                                children="open in viewer tab",
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(children=[main_graph()], id="main-container"),
            dynamic_spec_div(
                "main-spec-print", "main-spec-graph", "main-spec-image", 0
            ),
            html.Div(id="fake-output-for-callback-with-only-side-effects-0"),
            html.Div(id="fake-output-for-callback-with-only-side-effects-1"),
            html.Div(id="search-load-progress-flag"),
            load_search_drop("load-search"),
        ],
        # display title
        label="SEARCH",
        # used only by dcc.Tabs, apparently
        value="main_search_tab",
        # as opposed to regular DOM id
        id="main-search-tab",
    )
