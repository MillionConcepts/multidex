"""
callback registry: register functions from plotter.callbacks with app i/o
"""

from dash.dependencies import Input, Output, State, MATCH, ALL

from plotter.application.structure import (
    Y_INPUTS,
    MARKER_INPUTS,
    GRAPH_DISPLAY_INPUTS,
    FILTER_DROPDOWN_OUTPUTS,
    X_INPUTS,
)


def register_change_calc_input_visibility(
    app, configured_function, value_class
):
    app.callback(
        [
            Output("filter-1-" + value_class + "-container", "style"),
            Output("filter-2-" + value_class + "-container", "style"),
            Output("filter-3-" + value_class + "-container", "style"),
            Output("component-" + value_class + "-container", "style"),
        ],
        [Input("graph-option-" + value_class, "value")],
    )(configured_function)


def register_trigger_search_update(app, configured_function):
    # trigger updates on page load
    app.callback(
        Output({"type": "search-load-trigger", "index": ALL}, "value"),
        Input({"type": "load-trigger", "index": 0}, "value"),
        [State({"type": "search-load-trigger", "index": ALL}, "value")],
    )(configured_function)


def register_update_main_graph(app, configured_function):
    # trigger redraw of main graph on new search, axis calculation change, etc
    app.callback(
        [Output("main-graph", "figure"), Output("main-graph", "clickData")],
        # maybe later add an explicit recalculate button?
        [
            *X_INPUTS,
            *Y_INPUTS,
            *MARKER_INPUTS,
            *GRAPH_DISPLAY_INPUTS,
            Input({"type": "search-trigger", "index": ALL}, "value"),
            Input({"type": "main-graph-scale-trigger", "index": 0}, "value"),
            Input({"type": "highlight-trigger", "index": 0}, "value"),
            Input("main-graph-bounds", "value"),
            Input("main-graph-error", "value"),
            Input("main-graph", "clickData"),
            Input("clear-labels", "n_clicks")
            # Input({'type': 'load-trigger', 'index': 0}, 'value')
        ],
        [State("main-graph", "figure"), State("main-graph-average", "value")],
    )(configured_function)


def register_handle_highlight_save(app, configured_function):
    app.callback(
        [
            Output("highlight-description", "children"),
            Output({"type": "highlight-trigger", "index": 0}, "value"),
        ],
        [
            Input({"type": "load-trigger", "index": 0}, "value"),
            Input("highlight-save", "n_clicks"),
        ],
        [State({"type": "highlight-trigger", "index": 0}, "value")],
        prevent_initial_call=True,
    )(configured_function)


def register_toggle_panel_visibility(app, configured_function):
    app.callback(
        [
            Output({"type": "collapsible-panel", "index": MATCH}, "style"),
            Output({"type": "collapse-arrow", "index": MATCH}, "style"),
            Output({"type": "collapse-text", "index": MATCH}, "style"),
        ],
        [
            Input({"type": "collapse-div", "index": MATCH}, "n_clicks"),
            Input("collapse-all", "n_clicks")
        ],
        [
            State({"type": "collapsible-panel", "index": MATCH}, "style"),
            State({"type": "collapse-arrow", "index": MATCH}, "style"),
            State({"type": "collapse-text", "index": MATCH}, "style"),
        ],
        prevent_initial_call=True,
    )(configured_function)


def register_toggle_color_drop_visibility(app, configured_function):
    app.callback(
        [
            Output("color-scale", "style"),
            Output("color-solid", "style"),
        ],
        [Input("coloring-type", "value")],
    )(configured_function)


# change visibility of search filter inputs
# based on whether a 'quantitative' or 'qualitative'
# search field is selected
def register_toggle_search_input_visibility(app, configured_function):
    app.callback(
        [
            Output({"type": "term-search", "index": MATCH}, "style"),
            Output({"type": "number-search", "index": MATCH}, "style"),
        ],
        [Input({"type": "field-search", "index": MATCH}, "value")],
    )(configured_function)


def register_update_search_options(app, configured_function):
    # update displayed search options based on selected search field
    app.callback(
        [
            Output({"type": "term-search", "index": MATCH}, "options"),
            Output(
                {"type": "number-range-display", "index": MATCH}, "children"
            ),
            Output({"type": "number-search", "index": MATCH}, "value"),
        ],
        [
            Input({"type": "field-search", "index": MATCH}, "value"),
            Input({"type": "search-load-trigger", "index": MATCH}, "value"),
        ],
        [
            State({"type": "number-search", "index": MATCH}, "value"),
        ],
        prevent_initial_call=True,
    )(configured_function)


def register_update_search_ids(app, configured_function):
    # trigger active queryset / df update on new searches
    # or scaling / averaging requests
    app.callback(
        Output({"type": "search-trigger", "index": 0}, "value"),
        [
            Input({"type": "submit-search", "index": ALL}, "n_clicks"),
            Input({"type": "load-trigger", "index": 0}, "value"),
        ],
        [
            State({"type": "field-search", "index": ALL}, "value"),
            State({"type": "term-search", "index": ALL}, "value"),
            State({"type": "number-search", "index": ALL}, "value"),
            State({"type": "search-trigger", "index": 0}, "value"),
        ],
        prevent_initial_call=True,
    )(configured_function)


def register_toggle_averaged_filters(app, configured_function):
    # change and reset options on averaging request
    app.callback(
        FILTER_DROPDOWN_OUTPUTS,
        [
            Input("main-graph-average", "value"),
            # Input('interval1', 'n_intervals')
        ],
    )(configured_function)


def register_update_data_df(app, configured_function):
    app.callback(
        Output({"type": "main-graph-scale-trigger", "index": 0}, "value"),
        [
            Input({"type": "load-trigger", "index": 0}, "value"),
            Input("main-graph-scale", "value"),
            Input("main-graph-average", "value"),
            Input("main-graph-r-star", "value"),
        ],
        [State({"type": "main-graph-scale-trigger", "index": 0}, "value")],
    )(configured_function)


def register_control_search_dropdowns(app, configured_function):
    # handle creation and removal of search filters
    app.callback(
        [
            Output("search-controls-container", "children"),
            Output({"type": "submit-search", "index": 1}, "n_clicks"),
        ],
        [
            Input("add-param", "n_clicks"),
            Input("clear-search", "n_clicks"),
            Input({"type": "remove-param", "index": ALL}, "n_clicks"),
        ],
        [
            State("search-controls-container", "children"),
            State({"type": "submit-search", "index": 1}, "n_clicks"),
        ],
        prevent_initial_call=True,
    )(configured_function)


def register_handle_load(app, configured_function):
    app.callback(
        [
            Output("search-div", "children"),
            Output({"type": "load-trigger", "index": 0}, "value"),
            Output("default-settings-checked-div", "children")
        ],
        [
            Input("load-search-load-button", "n_clicks"),
        ],
        [
            State("load-search-drop", "value"),
            State({"type": "load-trigger", "index": 0}, "value"),
            State("default-settings-checked-div", "children")
        ],
    )(configured_function)


def register_update_spectrum_images(app, configured_function):
    app.callback(
        Output("spec-image", "children"),
        [Input("main-graph", "hoverData")],
    )(configured_function)


def register_graph_point_to_metadata(app, configured_function):
    app.callback(
        Output("spec-print", "children"),
        [Input("main-graph", "hoverData")],
    )(configured_function)


def register_update_spectrum_graph(app, configured_function):
    app.callback(
        Output("spec-graph", "figure"),
        [
            Input("main-graph", "hoverData"),
            Input("main-spec-scale", "value"),
            Input("main-spec-r-star", "value"),
            Input("main-spec-average", "value"),
            Input("main-spec-error", "value"),
        ],
    )(configured_function)


def register_save_search_state(app, configured_function):
    app.callback(
        Output({"type": "save-trigger", "index": 0}, "value"),
        [Input("save-search-save-button", "n_clicks")],
        [
            State("save-search-name-input", "value"),
            State({"type": "save-trigger", "index": 0}, "value"),
        ],
        prevent_initial_call=True,
    )(configured_function)


def register_populate_saved_search_drop(app, configured_function):
    app.callback(
        Output("load-search-drop", "options"),
        [
            Input({"type": "save-trigger", "index": 0}, "value"),
            Input({"type": "load-trigger", "index": 0}, "value"),
            Input("fire-on-load", "children"),
        ],
    )(configured_function)


def register_export_graph_csv(app, configured_function):
    app.callback(
        Output(
            "fake-output-for-callback-with-only-side-effects-0", "children"
        ),
        [Input("export-csv", "n_clicks")],
        [State("main-graph", "selectedData")],
        prevent_initial_call=True,
    )(configured_function)


def register_export_graph_png(app, configured_function):
    app.callback(
        Output(
            "fake-output-for-callback-with-only-side-effects-1", "children"
        ),
        [Input("graph-size-record-div", "children")],
        [State("main-graph", "figure")],
        prevent_initial_call=True,
    )(configured_function)


def register_record_graph_size_and_trigger_save(app):
    app.clientside_callback(
        """
        function() {
            const main_graph = document.getElementById("main-graph");
            const info_object = {
                'width': main_graph.clientWidth, 
                'height': main_graph.clientHeight
            }
            return JSON.stringify(info_object)
        }
        """,
        Output(
            "graph-size-record-div", "children"
        ),
        [Input("export-image", "n_clicks")],
        prevent_initial_call=True
    )


def register_drag_spec_print(app):
    app.clientside_callback(
        """function() {makeDraggable('spec-print-handle', 'spec-print-div')}""",
        Output(
            'fake-output-for-callback-with-only-side-effects-2', 'children'
        ),
        [Input('fire-on-load', 'children')]
    )


def register_hide_spec_print(app):
    app.clientside_callback(
        """function() {makeHider('spec-print-handle', 'spec-print')}""",
        Output(
            'fake-output-for-callback-with-only-side-effects-3', 'children'
        ),
        [Input('fire-on-load', 'children')]
    )

    # from dash.dependencies import Output, Input
    # app.clientside_callback(
    #     ClientsideFunction(namespace="clientside",
    #     function_name="make_draggable"),
    #     Output("marker-options", "data-drag"),
    #     [Input("marker-options", "id")]
    # )

# debug printer
# app.callback(
#     Output('fake-output-for-callback-with-only-side-effects-1', 'children'),
#     [Input('load-search-drop', 'value')]
# )(print_callback)
