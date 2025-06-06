"""factory functions for dash UI components"""
import random
from typing import Mapping, Optional, Iterable, Union, Sequence, Literal

import plotly.graph_objects as go
from dash import dcc
from dash import html
from dash.html import Div

from multidex.multidex_utils import none_to_empty
from multidex.plotter.colors import generate_palette_options, get_scale_type
from multidex.plotter.config.graph_style import (
    GRAPH_DISPLAY_SETTINGS,
    GRAPH_CONFIG_SETTINGS,
    css_variables,
)
from multidex.plotter.config.marker_style import MARKER_SYMBOL_SETTINGS


# note that style properties are camelCased rather than hyphenated in
# compliance with conventions for React virtual DOM
from multidex.plotter.types import SpectrumModel


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


def spec_controls_div(
    spec_model,
    id_prefix,
    scale_value="none",
    r_star=True,
    average=True,
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
                "width": "6rem",
            },
            children=[
                dcc.Checklist(
                    id=id_prefix + "-average",
                    className="info-text",
                    options=[{"label": "merge", "value": "average"}],
                    value=["average"] if average is True else [],
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
                    value=["r-star"] if r_star else [],
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
    fig = go.Figure(layout={**GRAPH_DISPLAY_SETTINGS})
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
    fig = go.Figure(layout={**GRAPH_DISPLAY_SETTINGS})
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
            for component_ix in range(6)
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


def model_range_display(element_id: str, index: int) -> html.Pre:
    """placeholder area for displaying range for number field searches"""
    return html.Pre(
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
        dcc.Input(
            id={"type": "free-search", "index": index},
            type="search",
            value=preset.get("free")
        ),
        model_options_drop(
            "term-search",
            index,
            value=preset.get("terms"),
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


def search_parameter_div(
    index: int, searchable_fields: Iterable[str], preset=None
) -> html.Div:
    if preset is None:
        preset = {}
    children = search_parameter_div_drop_elements(
        index, searchable_fields, preset
    )
    if index == 0:
        button = html.Button("add", id="add-param")
    else:
        button = html.Button(
            id={"type": "remove-param", "index": index},
            children="remove",
        )
    checklist_values = []
    for option in ("null", "invert", "is_free"):
        if preset.get(option) is True:
            checklist_values.append(option)
    checklist = dcc.Checklist(
        style={
            "marginLeft": "0.1rem",
            "display": "flex",
            "max-width": "8rem",
            "flex-wrap": "wrap"
        },
        id={"type": "param-logic-options", "index": index},
        className="info-text",
        options=[
            {"label": "null", "value": "null"},
            {"label": "flip", "value": "invert"},
            {"label": "free", "value": "is_free"},
        ],
        value=checklist_values,
    )
    children.append(
        html.Div(
            style={"display": "flex", "flexDirection": "row"},
            children=[button, checklist],
        )
    )
    return html.Div(
        className="search-parameter-container",
        children=children,
        id={"type": "search-parameter-div", "index": index},
    )


def search_container_div(spec_model, settings):
    search_container = html.Div(
        id="search-controls-container",
        className="search-controls-container",
    )
    searchable_fields = spec_model.searchable_fields()
    if not settings["search_parameters"]:
        search_container.children = [
            search_parameter_div(0, searchable_fields)
        ]
    else:
        search_container.children = [
            search_parameter_div(ix, searchable_fields, parameter)
            for ix, parameter in enumerate(settings["search_parameters"])
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


def state_div():
    return html.Div(children=[save_state_input(), load_state_input()])


def load_state_input():
    return html.Div(
        className="load-button-container",
        children=[
            dcc.Dropdown(id="load-state-drop", className="medium-drop"),
            html.Button(id="load-state-button", children="load"),
        ],
    )


def save_state_input():
    return html.Div(
        className="save-button-container",
        children=[
            html.Label(children=["state"], htmlFor="save-state-input"),
            dcc.Input(id="save-state-input", type="text"),
            html.Button(id="save-state-button", children="save"),
        ],
        # style={"display": "flex", "flexDirection": "column"},
    )


def axis_controls_div(
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
                    value=int(settings["component-" + axis + ".value"]),
                    label_content="component #",
                ),
            ],
        ),
    ]
    return html.Div(className="axis-controls-container", children=children)


def marker_coloring_type_div(coloring_type: str) -> Div:
    palette_types = ["sequential", "solid", "diverging", "cyclical"]
    if coloring_type == "qualitative":
        palette_types.append("qualitative")
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
        style={
            "display": "flex",
            "flexDirection": "column",
        },
        children=[
            html.Label(
                children=["size"],
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
        ]
    )


def marker_outline_div(
    outline_color, which: Literal["marker", "highlight"] = "marker"
) -> Div:
    return html.Div(
        style={
            "display": "flex",
            "flexDirection": "column",
        },
        children=[
            html.Label(
                className="info-text",
                children=["outline"],
                htmlFor=f"{which}-outline-radio",
            ),
            dcc.RadioItems(
                id=f"{which}-outline-radio",
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
        style={"height": "1rem", "width": "3rem", "margin-left": "0.3rem"},
        value=value,
        min=0,
        max=100,
    )


def marker_clip_opacity_div(settings: Mapping) -> Div:
    high = float(settings["color-clip-bound-high.value"])
    low = float(settings["color-clip-bound-low.value"])
    return html.Div(
        children=[
            html.Label(
                children=["color clip"],
                className="info-text",
                htmlFor="color-clip-bound-low",
            ),
            # TODO, maybe: make this and other number fields less visually
            #  hideous (very hard to override browser style)
            html.Div(
                children=[
                    clip_input("color-clip-bound-low", low),
                    clip_input("color-clip-bound-high", high),
                ],
                style={"display": "flex", "flexDirection": "row"}
            ),
            html.Label(
                children=["opacity"],
                className="info-text",
                htmlFor="marker-opacity-input"
            ),
            dcc.Input(
                type="number",
                id="marker-opacity-input",
                style={"height": "1.4rem", "width": "3rem"},
                value=int(settings.get('marker-opacity-input.value', 100)),
                min=0,
                max=100,
            )
        ],
        style={
            "display": "flex",
            "flexDirection": "column",
            "marginRight": "0.3rem",
            "marginLeft": "0.3rem",
        },
    )


def marker_shape_div(settings: Mapping) -> Div:
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
                id="marker-symbol-drop",
                className="medium-drop filter-drop",
                value=marker_symbol,
                options=MARKER_SYMBOL_SETTINGS,
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
                children=["embiggen"],
                className="info-text",
                htmlFor="highlight-size-radio",
            ),
            dcc.RadioItems(
                id="highlight-size-radio",
                className="radio-items",
                options=[
                    {"label": "x1", "value": 1},
                    {"label": "x2", "value": 2},
                    {"label": "x4", "value": 4},
                ],
                value=highlight_size,
            ),
        ],
    )


def highlight_color_symbol_div(color, symbol, outline_color) -> html.Div:
    return html.Div(
        children=[
            marker_outline_div(outline_color, "highlight"),
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
                id="highlight-symbol-drop",
                className="medium-drop filter-drop",
                value=symbol,
                options=[{"label": "none", "value": "none"}]
                + list(MARKER_SYMBOL_SETTINGS),
            )
        ]
    )


def marker_color_div(settings: Mapping) -> Div:
    palette = settings["palette-name-drop.value"]
    palette_type = get_scale_type(palette)
    return html.Div(
        id="marker-color-container",
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
            marker_coloring_type_div(palette_type),
        ],
    )


def search_controls_div(spec_model, settings: Mapping) -> html.Div:
    return html.Div(
        style={"display": "flex", "flexDirection": "row"},
        children=[
            search_container_div(spec_model, settings),
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
                        value=settings["logical_quantifier"],
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
    if settings["showgrid"] == "False":
        gridcolor = "#000000"
    # defensive backwards-compatibility thing
    elif "gridcolor" not in settings.keys():
        gridcolor = GRAPH_DISPLAY_SETTINGS["gridcolor"]
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
                    {"label": "off", "value": "#000000"},
                    {"label": "light", "value": css_variables["dark-tint-0"]},
                    {"label": "dark", "value": css_variables["dark-tint-2"]},
                ],
                value=gridcolor,
            ),

            html.Label(
                className="info-text",
                htmlFor="main-graph-regression-check",
            ),
            dcc.Checklist(
                ["fit line"],
                [],
                id="main-graph-regression-check",
                className="info-text",
                inline=True
            ),
            html.Button("clear labels", id="clear-labels"),
        ]
    )


def highlight_size_opacity_div(size, opacity) -> html.Div:
    return html.Div(
        id="highlight-size-opacity-div",
        style={
            "display": "flex",
            "flexDirection": "column",
            "marginRight": "0.3rem",
        },
        children=[
            highlight_size_div(size),
            html.Label(
                children=["opacity"],
                className="info-text",
                htmlFor="highlight-opacity-input"
            ),
            dcc.Input(
                type="number",
                id="highlight-opacity-input",
                style={"height": "1.4rem", "width": "3rem"},
                value=opacity,
                min=0,
                max=100,
            )
        ],
    )


def highlight_controls_div(settings: Mapping) -> html.Div:
    status = settings["highlight-toggle.value"]
    size = int(settings["highlight-size-radio.value"])
    symbol = settings["highlight-symbol-drop.value"]
    color = settings["highlight-color-drop.value"]
    opacity = int(settings.get('highlight-opacity-input.value', 100))
    outline_color = settings.get(
        "highlight-outline-radio.value", "rgba(0, 0, 0, 1)"
    )
    return html.Div(
        children=[
            html.Div(
                className="axis-controls-container",
                children=[
                    html.Button(
                        "set highlight",
                        id="highlight-save",
                        # style={"marginTop": "1rem"},
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
            highlight_size_opacity_div(size, opacity),
            highlight_color_symbol_div(color, symbol, outline_color)
        ],
        style={"display": "flex", "flexDirection": "row"},
    )


def scale_controls_div(spec_model, settings: Mapping) -> html.Div:
    average = settings["average_filters"] == "True"
    r_star = settings["r_star"] == "True"
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
                        value=settings.get("bounds_string"),
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
            spec_controls_div(
                spec_model,
                "main-graph",
                scale_value=settings["scale_to"],
                average=average,
                r_star=r_star,
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


def export_div():
    return html.Div(
        style={
            "display": "flex",
            "flexDirection": "column",
            "marginTop": "0.5rem",
        },
        children=[
            html.Label(children=["export"]),
            html.Button(
                "CSV",
                id="export-csv",
                style={"marginRight": "0.8rem"},
            ),
            dcc.Download(id="csv-export-endpoint"),
            dcc.Download(id="csv-export-endpoint-2"),
            html.Button("plot", id="export-plot"),
            dcc.Download(id="plot-export-endpoint"),
        ],
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
                axis_controls_div("x", spec_model, settings, filts),
            ),
            *collapse(
                "control-container-y",
                "y axis",
                axis_controls_div("y", spec_model, settings, filts),
            ),
            *collapse(
                "control-container-marker",
                "m axis",
                axis_controls_div("marker", spec_model, settings, filts),
            ),
            *collapse(
                "marker-options",
                "m options",
                marker_div(settings),
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
                scale_controls_div(spec_model, settings),
                off=True,
            ),
            *collapse("state-panel", "state", state_div(), off=True),
            *collapse("export-panel", "export", export_div(), off=True),
            *collapse(
                "display-controls",
                "display",
                display_controls_div(settings),
                off=True,
            ),
            *collapse(
                "spectrum-panel",
                "spectrum",
                spec_controls_div(
                    spec_model,
                    "main-spec",
                    spectrum_scale,
                    True,
                    True,
                    "error",
                ),
                off=True,
            ),
        ],
    )


def marker_div(settings):
    return html.Div(
        id="marker-div",
        style={
            "display": "flex",
            "flexDirection": "row",
            "marginRight": "0.3rem",
        },
        children=[
            marker_color_div(settings),
            marker_shape_div(settings),
            marker_clip_opacity_div(settings),
        ],
    )
