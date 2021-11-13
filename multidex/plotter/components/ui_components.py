"""factory functions for dash UI components"""
import random
from typing import Mapping, Optional, Iterable, Union, Sequence

import plotly.graph_objects as go
from dash import dcc
from dash import html
from dash.html import Div

from multidex_utils import none_to_empty
from plotter.colors import generate_palette_options, get_scale_type
from plotter.styles.graph_style import (
    GRAPH_DISPLAY_DEFAULTS,
    GRAPH_CONFIG_SETTINGS,
    css_variables,
)
from plotter.styles.marker_style import MARKER_SYMBOLS


# note that style properties are camelCased rather than hyphenated in
# compliance with conventions for React virtual DOM
from plotter.types import SpectrumModel


def scale_to_drop(model, element_id, value=None):
    """dropdown for selecting a virtual filter to scale to"""
    return dcc.Dropdown(
        id=element_id,
        className="medium-drop",
        options=[{"label": "none", "value": "none"}]
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
    scale_value="none",
    r_star_value="none",
    average_value="",
    error_value="none",
):
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
                    options=[{"label": "merge", "value": "average"}],
                    value=[average_value] if average_value else [],
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
                    value=[r_star_value] if r_star_value else [],
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


def marker_color_drop(
    element_id: str, palette: str, palette_type: str, allow_none: bool = False
) -> dcc.Dropdown:
    """
    dropdown for selecting color scales / solid colors given a scale type
    """
    options, palette = generate_palette_options(
        palette_type, palette, None, allow_none
    )
    return dcc.Dropdown(
        id=element_id,
        className="filter-drop medium-drop",
        options=options,
        value=palette,
        clearable=False,
    )


def collapse_arrow(id_for, title, off=False):
    """arrow that toggles collapse state of collapsible div with id id_for"""
    if off:
        arrow_style = {"WebkitTransform": "rotate(45deg)"}
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
            id={"type": "collapsible-panel", "index": collapse_id},
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
    return html.Div(
        className="axis-title-text",
        id=element_id + "-container",
        style={"display": "flex", "flexDirection": "column"},
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
                clearable=False,
            ),
        ],
    )


def component_drop(element_id, value, label_content=None, options=None):
    """dropdown for PCA (etc.) component selection"""
    if options is None:
        options = [
            {"label": str(component_ix + 1), "value": component_ix}
            for component_ix in range(8)
        ]
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


def parse_model_quant_entry(
    string: str,
) -> dict[str, Union[float, list[float]]]:
    # annoyed by this type hint but using it anyway
    value_dict: dict[str, Union[float, list[float]]] = {}
    is_range = "--" in string
    is_list = "," in string
    # TODO: expose these errors to the user in some useful way.
    if is_range and is_list:
        raise ValueError(
            "Entering both an explicit value list and a value range is "
            "currently not supported."
        )
    if is_range:
        range_list = string.split("--")
        if len(range_list) > 2:
            raise ValueError(
                "Entering a value range with more than two numbers is "
                "currently not supported."
            )
        # allow either a blank beginning or end, but not both
        for name, position in zip(("begin", "end"), (0, 1)):
            try:
                value_dict[name] = float(range_list[position])
            except ValueError:
                continue
        if value_dict == {}:
            raise ValueError("At least one numerical value must be entered.")
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


def model_range_display(element_id: str, index: int) -> html.Span:
    """placeholder area for displaying range for number field searches"""
    return html.Span(
        className="tooltiptext",
        id={"type": element_id, "index": index},
    )


def search_parameter_div_drop_elements(index, searchable_fields, preset):
    """
    per-parameter dropdown-selection elements
    """
    return [
        html.Label(children=["search field"], className="axis-title-text"),
        field_drop(
            searchable_fields, "field-search", index, preset.get("field")
        ),
        model_options_drop(
            "term-search",
            index,
            value=preset.get("term"),
            className="medium-drop term-search",
        ),
        html.Div(
            className="tooltipped",
            children=[
                model_range_display("number-range-display", index),
                model_range_entry("number-search", index, preset),
            ],
        ),
    ]


def search_parameter_div_option_elements(index):
    """
    all per-parameter click elements but the add/remove button, which
    differs between first and subsequent parameter divs
    """
    return [
        html.Button("add new", id="add-param"),
        # TODO: restore from save
        dcc.Checklist(
            style={"marginLeft": "1rem"},
            id={"type": "param-logic-options", "index": index},
            className="info-text",
            options=[{"label": "allow null", "value": "allow null"}],
            value=[],
        ),
    ]


def search_parameter_div(
    index: int, searchable_fields: Iterable[str], preset=None
) -> html.Div:
    if preset is None:
        preset = {}
    children = search_parameter_div_drop_elements(
        index, searchable_fields, preset
    )
    if index == 0:
        button = html.Button("add new", id="add-param")
    else:
        button = html.Button(
            id={"type": "remove-param", "index": index},
            children="remove",
        )
    children.append(
        html.Div(
            style={"display": "flex", "flexDirection": "row"},
            children=[button] + search_parameter_div_option_elements(index),
        )
    )
    return html.Div(
        className="search-parameter-container",
        children=children,
        id={"type": "search-parameter-div", "index": index},
    )


def search_container_div(spec_model, preset):
    search_container = html.Div(
        id="search-controls-container",
        className="search-controls-container",
    )
    searchable_fields = spec_model.searchable_fields()
    # TODO: may no longer need None
    # list was 'serialized' to string to put it in a single csv field
    if not preset:
        search_container.children = [
            search_parameter_div(0, searchable_fields)
        ]
    else:
        search_container.children = [
            search_parameter_div(ix, searchable_fields, parameter)
            for ix, parameter in enumerate(preset)
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


def load_search_drop():
    return html.Div(
        className="load-button-container",
        children=[
            html.Label(children=["search name"], htmlFor="load-search-drop"),
            dcc.Dropdown(id="load-search-drop", className="medium-drop"),
            html.Button(
                id="load-search-load-button",
                children="load",
            ),
        ],
    )


def save_search_input():
    return html.Div(
        className="save-button-container",
        children=[
            html.Label(children=["save as"], htmlFor="save-search-name-input"),
            dcc.Input(id="save-search-name-input", type="text"),
            html.Button(id="save-search-save-button", children="save"),
        ],
        style={"display": "flex", "flexDirection": "column"},
    )


def axis_controls_container(
    axis: str, spec_model, settings: Mapping, filter_options
) -> Div:
    children = [
        axis_value_drop(
            spec_model,
            "graph-option-" + axis,
            value=settings["graph-option-" + axis + ".value"],
            label_content=axis + " axis",
        ),
        html.Div(
            className="filter-container",
            children=[
                filter_drop(
                    spec_model,
                    "filter-1-" + axis,
                    value=settings["filter-1-" + axis + ".value"],
                    label_content="left",
                    options=filter_options,
                ),
                filter_drop(
                    spec_model,
                    "filter-3-" + axis,
                    value=settings["filter-3-" + axis + ".value"],
                    label_content="center",
                    options=filter_options,
                ),
                filter_drop(
                    spec_model,
                    "filter-2-" + axis,
                    value=settings["filter-2-" + axis + ".value"],
                    label_content="right",
                    options=filter_options,
                ),
                component_drop(
                    "component-" + axis,
                    value=settings["component-" + axis + ".value"],
                    label_content="component #",
                ),
            ],
        ),
    ]
    return html.Div(className="axis-controls-container", children=children)


def marker_coloring_type_div(coloring_type: str) -> Div:
    palette_types = ("sequential", "solid", "diverging", "cyclical")
    return html.Div(
        style={"display": "flex", "flexDirection": "column"},
        children=[
            html.Label(
                className="info-text",
                children=["palette type"],
                htmlFor="palette-type-drop",
            ),
            dcc.Dropdown(
                id="palette-type-drop",
                className="filter-drop medium-drop",
                options=[
                    {"label": c_type, "value": c_type}
                    for c_type in palette_types
                ],
                value=coloring_type,
            ),
        ],
    )


def marker_size_div(marker_size) -> Div:
    return html.Div(
        style={"display": "flex", "flexDirection": "row"},
        children=[
            html.Label(
                children=["size: "],
                className="info-text",
                htmlFor="marker-size-radio",
            ),
            dcc.RadioItems(
                id="marker-size-radio",
                className="radio-items",
                options=[
                    {"label": "s", "value": 8},
                    {"label": "m", "value": 11},
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
            "marginTop": "0.5rem",
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
                    {"label": "off", "value": "off"},
                    {"label": "b", "value": "rgba(0,0,0,1)"},
                    {"label": "w", "value": "rgba(255,255,255,1)"},
                ],
                value=outline_color,
            ),
        ],
    )


def clip_input(element_id, value):
    return dcc.Input(
        type="number",
        id=element_id,
        style={"height": "1.4rem", "width": "3rem"},
        value=value,
        min=0,
        max=100,
    )


def marker_clip_div(settings: Mapping) -> Div:
    high = int(settings["color-clip-bound-high.value"])
    low = int(settings["color-clip-bound-low.value"])
    return html.Div(
        children=[
            html.Label(
                children=["color clip"],
                className="axis-title-text",
                # TODO: wrap this more nicely
                htmlFor="color-clip-bound-low",
            ),
            # TODO: make this and other number fields less visually hideous
            clip_input("color-clip-bound-low", low),
            clip_input("color-clip-bound-high", high),
        ],
        style={
            "display": "flex",
            "flexDirection": "column",
            "marginRight": "0.3rem",
            "marginLeft": "0.3rem",
        },
    )


def marker_options_div(settings: Mapping) -> Div:
    marker_symbol = settings["marker-symbol-drop.value"]
    outline_color = settings["marker-outline-radio.value"]
    marker_size = int(settings["marker-size-radio.value"])
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
            html.Label(
                children=["marker symbol"],
                htmlFor="marker-symbol-drop",
                className="info-text",
            ),
            dcc.Dropdown(
                "marker-symbol-drop",
                className="medium-drop filter-drop",
                value=marker_symbol,
                options=MARKER_SYMBOLS,
            ),
        ],
    )


def highlight_size_div(highlight_size: str) -> Div:
    return html.Div(
        style={
            "display": "flex",
            "flexDirection": "column",
            "marginTop": "0.5rem",
        },
        children=[
            html.Label(
                children=["embiggen: "],
                className="info-text",
                htmlFor="highlight-size-radio",
            ),
            dcc.RadioItems(
                id="highlight-size-radio",
                className="info-text",
                options=[
                    {"label": "none", "value": 1},
                    {"label": "some", "value": 2},
                    {"label": "lots", "value": 4},
                ],
                value=highlight_size,
            ),
        ],
    )


def marker_color_symbol_div(settings: Mapping) -> Div:
    palette = settings["palette-name-drop.value"]
    palette_type = get_scale_type(palette)
    return html.Div(
        id="marker-color-symbol-container",
        style={"display": "flex", "flexDirection": "column", "width": "8rem"},
        className="axis-controls-container",
        children=[
            html.Label(
                children=["palette"],
                htmlFor="palette-name-drop",
                className="info-text",
            ),
            marker_color_drop(
                "palette-name-drop",
                palette=palette,
                palette_type=palette_type,
            ),
            marker_coloring_type_div(get_scale_type(palette)),
        ],
    )


def search_controls_div(spec_model, settings: Mapping) -> html.Div:
    return html.Div(
        style={"display": "flex", "flexDirection": "row"},
        children=[
            search_container_div(
                spec_model,
                settings["search_parameters"],
            ),
            html.Div(
                className="search-button-container",
                children=[
                    html.Button("clear search", id="clear-search"),
                    html.Button(
                        id={"type": "submit-search", "index": 0},
                        children="update graph",
                    ),
                    dcc.RadioItems(
                        id="logical-quantifier-radio",
                        className="radio-items",
                        options=[
                            {"label": "AND", "value": "AND"},
                            {"label": "OR", "value": "OR"},
                        ],
                        value="AND",
                    ),
                    # hidden trigger for queryset update on dropdown removal
                    html.Button(
                        id={"type": "submit-search", "index": 1},
                        style={"display": "none"},
                    ),
                ],
            ),
        ],
    )


def display_controls_div(settings: Mapping) -> html.Div:
    if settings["showgrid"] is False:
        gridcolor = False
    else:
        gridcolor = settings["gridcolor"]
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
                    {"label": "white", "value": "rgba(255,255,255,1)"},
                    {"label": "light", "value": css_variables["dark-tint-0"]},
                    {"label": "dark", "value": css_variables["dark-tint-2"]},
                ],
                value=settings["plot_bgcolor"],
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
                value=gridcolor,
            ),
            html.Button("clear labels", id="clear-labels"),
        ]
    )


def highlight_options_div(size, color, symbol) -> html.Div:
    return html.Div(
        id="highlight-options-div",
        style={
            "display": "flex",
            "flexDirection": "column",
            "marginRight": "0.3rem",
        },
        children=[
            highlight_size_div(size),
            html.Label(
                children=["highlight color"],
                htmlFor="highlight-color-drop",
                className="info-text",
            ),
            marker_color_drop(
                "highlight-color-drop",
                palette=color,
                palette_type="solid",
                allow_none=True,
            ),
            html.Label(
                children=["highlight symbol"],
                htmlFor="highlight-symbol-drop",
                className="info-text",
            ),
            dcc.Dropdown(
                "highlight-symbol-drop",
                className="medium-drop filter-drop",
                value=symbol,
                options=[{"label": "none", "value": "none"}]
                + list(MARKER_SYMBOLS),
            ),
        ],
    )


def highlight_controls_div(settings: Mapping) -> html.Div:
    status = settings["highlight-toggle.value"]
    size = int(settings["highlight-size-radio.value"])
    symbol = settings["highlight-symbol-drop.value"]
    color = settings["highlight-color-drop.value"]
    return html.Div(
        children=[
            html.Div(
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
                            {"label": "highlight off", "value": "off"},
                            {"label": "on", "value": "on"},
                        ],
                        value=status,
                    ),
                    html.P(
                        id="highlight-description",
                        className="info-text",
                        style={"maxWidth": "12rem"},
                        children="no highlight presently set.",
                    ),
                ],
            ),
            highlight_options_div(size, color, symbol),
        ],
        style={"display": "flex", "flexDirection": "row"},
    )


def scale_control_div(spec_model, settings: Mapping) -> html.Div:
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
                        style={"height": "1.4rem", "width": "10rem"},
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
                scale_value=settings["scale_to"],
                average_value=settings["average_filters"],
                # TODO: fix init issue, need extra layer somewhere
                r_star_value="r-star",
            ),
        ]
    )


def fake_output_divs(n_divs: int) -> list[html.Div]:
    return [
        html.Div(
            id=f"fake-output-for-callback-with-only-side-effects-{ix}",
            style={"display": "none"},
        )
        for ix in range(n_divs)
    ]


def save_div():
    return html.Div(
        [
            save_search_input(),
            html.Div(
                style={
                    "display": "flex",
                    "flexDirection": "row",
                    "marginTop": "0.5rem",
                },
                children=[
                    html.Button(
                        "CSV",
                        id="export-csv",
                        style={"marginRight": "0.8rem"},
                    ),
                    html.Button("image", id="export-image"),
                ],
            ),
        ]
    )


def graph_controls_div(
    spec_model: SpectrumModel,
    settings: Mapping,
    filts: Sequence,
    spectrum_scale: str,
) -> Div:
    """factory for top-level graph controls div at top of screen"""
    return html.Div(
        className="graph-controls-container",
        children=[
            *collapse(
                "control-container-x",
                "x axis",
                axis_controls_container("x", spec_model, settings, filts),
            ),
            *collapse(
                "control-container-y",
                "y axis",
                axis_controls_container("y", spec_model, settings, filts),
            ),
            *collapse(
                "control-container-marker",
                "m axis",
                axis_controls_container("marker", spec_model, settings, filts),
            ),
            *collapse(
                "color-controls",
                "m style",
                marker_color_symbol_div(settings),
                off=True,
            ),
            *collapse(
                "marker-options",
                "m options",
                marker_options_div(settings),
                off=True,
            ),
            *collapse(
                "marker-clip",
                "m clip",
                marker_clip_div(settings),
                off=True,
            ),
            *collapse(
                "highlight-controls",
                "h controls",
                highlight_controls_div(settings),
                off=True,
            ),
            *collapse(
                "search-controls",
                "search",
                search_controls_div(spec_model, settings),
            ),
            # TODO: at least the _nomenclature_ of these two separate
            #  'scaling' divs should be clarified
            *collapse(
                "numeric-controls",
                "scaling",
                scale_control_div(spec_model, settings),
                off=True,
            ),
            *collapse(
                "spec-controls",
                "spectrum",
                scale_controls_container(
                    spec_model,
                    "main-spec",
                    spectrum_scale,
                    "r-star",
                    "average",
                    "error",
                ),
                off=True,
            ),
            *collapse("load-panel", "load", load_search_drop(), off=True),
            *collapse(
                "save-panel",
                "save",
                html.Div([save_search_input(), save_div()]),
                off=True,
            ),
            *collapse(
                "graph-display-panel",
                "display",
                display_controls_div(settings),
                off=True,
            ),
        ],
    )
