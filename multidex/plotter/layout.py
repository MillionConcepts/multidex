from functools import partial
from typing import Optional, Mapping

import dash_core_components as dcc
import dash_html_components as html

from multidex_utils import get_if
from plotter.components.ui_components import (
    collapse,
    axis_controls_container,
    marker_color_symbol_div,
    marker_options_div,
    highlight_controls_div,
    search_controls_div,
    scale_control_div,
    scale_controls_container,
    load_search_drop,
    save_search_input,
    display_controls_div,
    main_graph,
    dynamic_spec_div,
    trigger_div,
)


# primary search panel
def search_div(
    spec_model: "Spectrum", restore_dictionary: Optional[Mapping] = None
):
    # are we restoring from saved settings? if so, this function gets them;
    # if not, this function politely acts as None
    # TODO: refactor this horror to load from an external set of defaults
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
                    "control-container-x",
                    "x axis",
                    axis_controls_container("x", spec_model, get_r, filts),
                ),
                *collapse(
                    "control-container-y",
                    "y axis",
                    axis_controls_container("y", spec_model, get_r, filts),
                ),
                *collapse(
                    "control-container-marker",
                    "m axis",
                    axis_controls_container(
                        "marker", spec_model, get_r, filts
                    ),
                ),
                *collapse(
                    "color-controls",
                    "m style",
                    marker_color_symbol_div(get_r),
                    off=True,
                ),
                *collapse(
                    "marker-options",
                    "m options",
                    marker_options_div(get_r),
                    off=True,
                ),
                *collapse(
                    "highlight-controls",
                    "highlight",
                    highlight_controls_div(get_r),
                    off=True,
                ),
                *collapse(
                    "search-controls",
                    "search",
                    search_controls_div(spec_model, get_r),
                ),
                # TODO: at least the _nomenclature_ of these two separate
                #  'scaling' divs should be clarified
                *collapse(
                    "numeric-controls",
                    "scaling",
                    scale_control_div(spec_model, get_r),
                    off=True,
                ),
                *collapse(
                    "spec-controls",
                    "spectrum",
                    scale_controls_container(
                        spec_model,
                        "main-spec",
                        "L1_R1",
                        "r-star",
                        "average",
                        "error",
                    ),
                    off=True,
                ),
                *collapse(
                    "load-panel",
                    "load",
                    load_search_drop("load-search"),
                    off=True,
                ),
                *collapse(
                    "save-panel",
                    "save",
                    html.Div(
                        [
                            save_search_input("save-search"),
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
                                    html.Button(
                                        "image",
                                        id="export-image",
                                    ),
                                ],
                            ),
                        ]
                    ),
                    off=True,
                ),
                *collapse(
                    "graph-display-panel",
                    "display",
                    display_controls_div(get_r),
                    off=True,
                ),
            ],
        ),
        html.Div(
            style={
                "display": "flex",
                "flexDirection": "row",
                "height": "85vh",
            },
            children=[
                main_graph(
                    style={"height": "100%", "width": "66%", "flexShrink": 0}
                ),
                dynamic_spec_div(),
            ],
            id="main-container",
        ),
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
            html.Button(
                id="collapse-all",
                style={
                    "background": "mediumseagreen",
                    "position": "fixed",
                    "bottom": "0.7rem",
                    "borderWidth": "5px",
                    "height": "1.2rem",
                    "width": "1.2rem",
                    "zIndex": "9999",
                },
            ),
            html.Div(
                id="spec-print-div",
                style={
                    "position": "absolute",
                    "zIndex": 9999,
                    "background": "aliceblue",
                    "left": "90vw",
                },
                children=[
                    html.Div(
                        id="spec-print-handle",
                        style={
                            "background": "lightpink",
                            "width": "1.1rem",
                            "height": "1.1rem",
                        },
                    ),
                    html.P(
                        id="spec-print", style={"display": "none", "margin": 0}
                    ),
                ],
            ),
            search_div(spec_model),
            # hidden divs for async triggers, dummy outputs, etc
            trigger_div("main-graph-scale", 1),
            trigger_div("search", 2),
            trigger_div("load", 1),
            trigger_div("save", 1),
            trigger_div("highlight", 1),
            html.Div(
                id="fire-on-load", children="2", style={"display": "none"}
            ),
            html.Div(
                id="fake-output-for-callback-with-only-side-effects-0",
                style={"display": "none"},
            ),
            html.Div(
                id="fake-output-for-callback-with-only-side-effects-1",
                style={"display": "none"},
            ),
            html.Div(
                id="fake-output-for-callback-with-only-side-effects-2",
                style={"display": "none"},
            ),
            html.Div(
                id="fake-output-for-callback-with-only-side-effects-3",
                style={"display": "none"},
            ),
            html.Div(
                id="default-settings-checked-div", style={"display": "none"}
            ),
            html.Div(id="graph-size-record-div", style={"display": "none"}),
            html.Div(
                id="search-load-progress-flag", style={"display": "none"}
            ),
            html.Div(
                children=[
                    dcc.Input(
                        id="search-load-trigger",
                        value=0
                    )
                ],
                style={"display": "none"},
            ),
            # dcc.Interval(id="interval1", interval=1000, n_intervals=0),
        ],
        id="multidex",
    )
