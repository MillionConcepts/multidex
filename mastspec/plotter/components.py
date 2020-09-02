"""factory functions for dash components."""

from ast import literal_eval
from functools import partial

import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objects as go

from utils import get_if, none_to_empty, rows


# note that style properties are camelCased rather than hyphenated
# b/c React

def dynamic_spec_div(print_name, graph_name, image_name, index):
    return html.Div(
        children=[
            html.Pre(
                children=[],
                id={"type": print_name, "index": index},
                style={
                    "width": "20vw",
                    "marginLeft": "5vw",
                    "display": "inline-block",
                    "verticalAlign": "top",
                },
            ),
            html.Div(
                children=[spec_graph(graph_name, index)],
                id={"type": graph_name + "-container", "index": index},
                style={"display": "inline-block"},
            ),
            html.Div(
                id={"type": image_name, "index": index},
                style={"display": "inline-block"},
            ),
        ],
        id={"type": "spec-container", "index": index},
    )


def main_graph():
    """dash component factory for main graph"""
    fig = go.Figure()
    fig.update_layout(margin={"l": 10, "r": 10, "t": 25, "b": 0})
    return dcc.Graph(id="main-graph", figure=fig, style={"height": "45vh"})


def spec_graph(name, index):
    """dash component factory for reflectance graphs"""
    fig = go.Figure()
    fig.update_layout(margin={"l": 10, "r": 10, "t": 25, "b": 0})
    return dcc.Graph(
        id={"type": name, "index": index},
        figure=fig,
        style={"height": "30vh", "width": "45vw"},
    )


def image_holder(index=0):
    """dash component factory for zoomable static images. maybe. placeholder"""
    return dcc.Graph(id="image-" + str(index),)


def main_graph_scatter(x_axis, y_axis, text, customdata):
    """
    partial placeholder scatter function for main graph.
    this function just creates the Plotly figure; data is read from db
    and formatted in make_axis.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=x_axis,
            y=y_axis,
            # change this to be hella popup text
            text=text,
            customdata=customdata,
            mode="markers",
            marker={"color": "blue"},
        )
    )
    fig.update_layout(margin={"l": 10, "r": 10, "t": 25, "b": 0})
    return fig


def mspec_graph_line(spectrum):
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
        go.Scattergl(
            x=x_axis, y=y_axis, mode="lines+markers", line = {'color':spectrum.roi_hex_code()}
            )
        )
    fig.update_layout(margin={"l": 10, "r": 10, "t": 25, "b": 0})
    return fig



def axis_value_drop(spec_model, element_id, value=None):
    """
    dropdown for selecting calculation options for axes
    """

    options = [
        {"label": option["label"], "value": option["value"]}
        for option in spec_model.axis_value_properties
    ]
    if not value:
        value = options[0]["value"]
    return dcc.Dropdown(id=element_id, options=options, value=value)


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


def model_options_drop(field, element_id, index, value=None):
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


def model_range_entry(element_id, index, begin=None, end=None):
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


def parse_model_quant_entry(string):
    value_dict = {}
    is_range = "-" in string
    is_list = "," in string
    if is_range and is_list:       
        raise ValueError(
                "Entering both an explicit value list and a value range is currently not supported."
                )
    if is_range:
        range_list = string.split("-")
        if len(range_list) > 2:
            raise ValueError(
                "Entering a value range with more than two numbers is currently not supported."
                )
        # allow either a blank beginning or end, but not both
        try: 
            value_dict["begin"] = float(range_list[0])
        except:
            value_dict["begin"] = ""
        try: 
            value_dict["end"] = float(range_list[1])
        except:
            value_dict["end"] = ""
        if not (value_dict["begin"] or value_dict["end"]):
            raise ValueError("Either a beginning or end numerical value must be entered.")
    elif string == "":
        pass
    else:
        list_list = string.split(",")
        # do not allow ducks and rutabagas and such to be entered into the list
        try: 
            value_dict["value_list"] = [float(item) for item in list_list]
        except:
            raise ValueError("Non-numerical lists are currently not supported.")
    return value_dict


def unparse_model_quant_entry(value_dict):
    if value_dict is None:
        text = ""
    elif ("value_list" in value_dict.keys()) and (
            ("begin" in value_dict.keys()) or ("end" in value_dict.keys())
        ):
        raise ValueError(
            "Entering both an explicit value list and a value range is currently not supported."
            )
    elif "value_list" in value_dict.keys():
        text = ",".join(value_dict["value_list"])
    elif ("begin" in value_dict.keys()) or ("end" in value_dict.keys()):
        text = value_dict["begin"]+"-"+value_dict["end"]
    return text


def model_range_entry_2(element_id, index, value_dict=None):
    """
    entry field for selecting a range of values for a
    quantitatively-valued field.
    """
    return dcc.Input(
            id={"type": element_id, "index": index},
            type="text",
            value=unparse_model_quant_entry(value_dict)
        )


def model_range_display(element_id, index):
    """placeholder area for displaying range for number field searches"""
    return html.P(id={"type": element_id, "index": index})


def search_parameter_div(index, searchable_fields, preset_parameter=None):
    get_r = partial(get_if, preset_parameter is not None, preset_parameter)
    children = [
        field_drop(searchable_fields, "field-search", index, get_r("field")),
        model_options_drop("group", "term-search", index, get_r("term")),
        model_range_entry_2(
            "number-search", index, get_r("value_dict")
        ),
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
        children=children, id={"type": "search-parameter-div", "index": index}
    )


def search_container_div(searchable_fields, preset_parameters):
    search_container = html.Div(id="search-container")
    if preset_parameters:
        # list was 'serialized' to string to put it in a single df cell
        preset_parameters = literal_eval(preset_parameters)
        search_container.children = [
            search_parameter_div(ix, searchable_fields, parameter)
            for ix, parameter in enumerate(preset_parameters)
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
                        figure=splot.graph(),
                    )
                ],
                id={"type": "view-graph-container", "index": index},
            ),
            html.Div(
                children=[str(splot.settings())],
                id={"type": "view-settings", "index": index},
            ),
            html.Button(
                id={"type": "tab-close-button", "index": index},
                children="close this tab",
            ),
            dynamic_spec_div(
                "view-spec-print", "view-spec-graph", "view-spec-image", index
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
            dcc.Input(id={"type": prefix + "-trigger", "index": index}, value = "")
            for index in range(number_of_triggers)
        ],
        style={"display": "none"},
        id=prefix + "-trigger-div",
    )


def load_search_drop(element_id):
    return html.Div(children=[
        dcc.Dropdown(id = element_id+'-drop'),
        html.Button(id = element_id+'-load-button', children = 'load'),
        html.Button(id = element_id+'-save-button', children = 'save')
        ])


# primary search panel
def search_tab(spec_model, restore_dictionary=None):

    # are we restoring from saved settings? if so, this function gets them;
    # if not, this function politely acts as None
    get_r = partial(
        get_if, restore_dictionary is not None, restore_dictionary
    )

    return dcc.Tab(
        children=[
            html.Div(
                children=[
                    axis_value_drop(
                        spec_model,
                        "axis-option-x",
                        value=get_r("axis-option-x"),
                    ),
                    filter_drop(
                        spec_model, "filter-1-x", value=get_r("filter-1-x.value")
                    ),
                    filter_drop(
                        spec_model, "filter-3-x", value=get_r("filter-3-x.value")
                    ),
                    filter_drop(
                        spec_model, "filter-2-x", value=get_r("filter-2-x.value")
                    ),
                ]
            ),
            html.Div(
                children=[
                    axis_value_drop(
                        spec_model,
                        "axis-option-y",
                        value=get_r("axis-option-y.value"),
                    ),
                    filter_drop(
                        spec_model, "filter-1-y", value=get_r("filter-1-y.value")
                    ),
                    filter_drop(
                        spec_model, "filter-3-y", value=get_r("filter-3-y.value")
                    ),
                    filter_drop(
                        spec_model, "filter-2-y", value=get_r("filter-2-y.value")
                    ),
                ]
            ),
            trigger_div("search", 2),
            trigger_div("load", 1),
            search_container_div(
                spec_model.searchable_fields, get_r("search_parameters")
            ),
            html.Button("add search parameter", id="add-param"),
            html.Button(
                id={"type": "submit-search", "index": 0}, children="Submit",
            ),
            # hidden trigger for queryset update on dropdown removal
            html.Button(
                id={"type": "submit-search", "index": 1},
                style={"display": "none"},
            ),
            load_search_drop('load-search'),
            html.Div(children=[main_graph()], id="main-graph-container"),
            html.Button(
                id="viewer-open-button", children="open in graph viewer tab",
            ),
            dynamic_spec_div(
                "main-spec-print", "main-spec-graph", "main-spec-image", 0
            ),
            html.Div(id='fake-output-for-callback-with-only-side-effects-0'),
            html.Div(id='fake-output-for-callback-with-only-side-effects-1'),
            html.Div(id='search-load-progress-flag')
        ],
        # display title
        label="SEARCH",
        # used only by dcc.Tabs, apparently
        value="main_search_tab",
        # as opposed to regular DOM id
        id="main-search-tab",
    )
