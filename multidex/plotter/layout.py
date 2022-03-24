from typing import Optional, Mapping

from dash import dcc
from dash import html

from plotter.components.ui_components import (
    dynamic_spec_div,
    fake_output_divs,
    graph_controls_div,
    main_graph,
    trigger_div,
)
from plotter.config.settings import instrument_settings
from plotter.types import SpectrumModel


def primary_app_div(
    spec_model: SpectrumModel, settings: Optional[Mapping] = None
) -> html.Div:
    """
    generates the primary application div.
    """
    if settings is None:
        settings = instrument_settings(spec_model.instrument)
    # TODO: this feels bad
    if settings.get("average_filters") == "True":
        filts = [
            {"label": filt, "value": filt}
            for filt in spec_model.canonical_averaged_filters
        ]
    else:
        filts = None
    # TODO: dumb hack, shift into config and make saveable
    if spec_model.instrument == "MCAM":
        spectrum_scale = "L6_R6"
    elif spec_model.instrument == "ZCAM":
        spectrum_scale = "L1_R1"
    else:
        spectrum_scale = None
    search_children = [
        graph_controls_div(spec_model, settings, filts, spectrum_scale),
        html.Div(
            style={
                "display": "flex",
                "flexDirection": "row",
                "height": "83vh",
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
    return html.Div(children=search_children, id="search-div")


def multidex_body(spec_model: SpectrumModel) -> html.Div:
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
                    html.Div(
                        id="spec-print", style={"display": "none", "margin": 0}
                    ),
                ],
            ),
            primary_app_div(spec_model),
            # hidden divs for async triggers, dummy outputs, etc
            trigger_div("main-graph-scale", 1),
            trigger_div("search", 2),
            trigger_div("load", 1),
            trigger_div("save", 1),
            trigger_div("highlight", 1),
            html.Div(
                id="fire-on-load", children="2", style={"display": "none"}
            ),
            *fake_output_divs(4),
            html.Div(
                id="default-settings-checked-div", style={"display": "none"}
            ),
            html.Div(id="graph-size-record-div", style={"display": "none"}),
            html.Div(
                id="search-load-progress-flag", style={"display": "none"}
            ),
            html.Div(
                children=[dcc.Input(id="search-load-trigger", value=0)],
                style={"display": "none"},
            ),
            # tick-tock
            # dcc.Interval(id="interval1", interval=1000, n_intervals=0),
        ],
        id="multidex",
    )
