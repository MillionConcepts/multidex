import os

import django
import flask

# import pylibmc
from dash import dash
from dash.dependencies import Input, Output, State, MATCH, ALL
import dash_core_components as dcc
import dash_html_components as html
from flask_caching import Cache

from plotter.debug import debug_check_cache
from plotter.spectrum_ops import filter_df_from_queryset
from plotter_utils import partially_evaluate_from_parameters, qlist, \
    model_metadata_df

# note: ignore any PEP 8-based linter / IDE complaints about import order: the
# following statements _must_ come before we import all the django dependencies
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "multidex.settings")
os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"
django.setup()

from plotter.components import (
    main_graph,
    main_graph_scatter,
    mspec_graph_line,
    search_tab,
)
from plotter.models import MSpec
from plotter.graph import (
    cache_set,
    cache_get,
    update_spectrum_images,
    update_spectrum_graph,
    control_tabs,
    control_search_dropdowns,
    recalculate_main_graph,
    update_search_options,
    update_search_ids,
    change_calc_input_visibility,
    toggle_search_input_visibility,
    graph_point_to_metadata,
    populate_saved_search_drop,
    save_search_tab_state,
    toggle_averaged_filters,
    update_filter_df, handle_main_highlight_save, export_graph_csv,
    toggle_panel_visibility,
)

# initialize the app itself. HTML / react objects must be described in this
# object
# as dash components.
# callback functions that handle user input must also be described in this
# object.
# adding __name__ preserves relative path when and if this is not the  __main__
# module (i.e. if served by gunicorn)
app = dash.Dash(__name__)

# we are using flask-caching to share state between callbacks,
# because dash refuses to enforce thread safety in python globals and
# so causes problems when globals are set within the execution tree
# of callbacks.
# this is separate from in-browser persistence we will likely add later to
# protect against, e.g., stray page refreshes.
# this also allows us to cache some expensive things like database lookups.
# (performance has not been profiled here, though.)
# we could also instead consider memoizing these using standard library
# tools like lru_cache.
# note that we are _not_ presently attempting to make this a multiuser
# application,
# but a separate cache per user might easily allow that.
# flask-caching requires an explicitly-set 'backend.'
# Our current backend of choice is memcached, which requires a memcached
# server running on the host.
# if a memcached install is not desired, don't run this cell,
# and change 'cache_type' in the cell below.

# initialize a client for caching
# make sure memcached is running with correct parameters
# systemctl edit memcached.service --full
# systemctl start memcached
# on non systemd systems, just run from command line
# current options:
# [Service]
# Environment=OPTIONS="-I 24m, -m 1200"
# 1 meg is the default for -I / slab size, which
# was fine for the test set but too small for this set.
# the whole prefetched database is probably around 6M in
# memory, and then individual parameters may it up further.
# see also memcached-tool 127.0.0.1:11211 settings
# (this can easily be set to start at runtime in a container)
# client = pylibmc.Client(["127.0.0.1"], binary=True)

# initialize the cache itself and register it with the app

# change this to 'filesystem' if you don't want to install memcached.

# I'm getting some bad spookiness I don't currently understand,
# so presently switching back to filesystem with a tmpfs.
# (with root:)
# mount -o size=1024M -t tmpfs mastspeccache .cache

cache_type = "filesystem"

CACHE_CONFIG = {
    "CACHE_TYPE": cache_type,
    "CACHE_DIR": ".cache",
    "CACHE_DEFAULT_TIMEOUT": 0,
    "CACHE_THRESHOLD": 0,  # important for filesystem backend
    # 'SERVERS': client  # filesystem backend will just ignore this
}
cache = Cache()
cache.init_app(app.server, config=CACHE_CONFIG)
cache.clear()  # this may or may not be desirable in prod, but gives us a
# clean slate

# cache_set and cache_get are factory functions for setter and getter
# functions for a flask_caching.Cache object.
# in the context of this app initialization, they can be thought
# of as defining a namespace.
cset = cache_set(cache)
cget = cache_get(cache)

spec_model = MSpec
cset("spec_model_name", "Mastcam")

# main_graph is a factory function for a locally-defined dash component.
# dash apps use dash components to automagically generate HTML and
# react components and handle user I/O at runtime.
# in general our convention is to encapsulate component definitions
# in order to separate them from app definition / initialization.
# we put them in factory functions so that things like html attributes
# can be defined dynamically.
# the exceptions, currently, are the flask route (which, somehow, is
# a component?) and the very top level of app layout.
# this makes it easier for us to move a single component around within
# the app layout without worrying about changing the component.
# our factory functions for components are stored in plotter.components.
fig = main_graph()

# active queryset is explicitly stored in global cache, as are
# many other app runtime values
cset("main_search_ids", qlist(spec_model.objects.all(), "id"))

cset(
    "main_highlight_ids",
    qlist(spec_model.objects.all(), "id"),
)

cset("main_graph_filter_df", filter_df_from_queryset(spec_model.objects.all()))

cset("metadata_df", model_metadata_df(spec_model, ['observation']))

# this variable is a list of open graph-view tabs;
# these are separate from the 'main' / 'search' tab.
cset("open_graph_viewers", [])
cset("scale_to", "None")
cset("average_filters", False)

# these are simply lists of inputs that refer to
# components produced by plotter.components.filter_drop.
# they are defined here for convenience in order to avoid excessive
# repetition in app structure definition and function calls.
# it's possible these should actually be in plotter.components.

filter_dropdowns = [
    "main-filter-1-marker",
    "main-filter-2-marker",
    "main-filter-3-marker",
    "main-filter-1-y",
    "main-filter-2-y",
    "main-filter-3-y",
    "main-filter-1-x",
    "main-filter-2-x",
    "main-filter-3-x",
]

x_inputs = [
    Input(dropdown, "value")
    for dropdown in filter_dropdowns
    if dropdown.endswith("-x")
] + [Input("main-graph-option-x", "value")]

y_inputs = [
    Input(dropdown, "value")
    for dropdown in filter_dropdowns
    if dropdown.endswith("-y")
] + [Input("main-graph-option-y", "value")]

marker_inputs = [
    Input(dropdown, "value")
    for dropdown in filter_dropdowns
    if dropdown.endswith("-marker")
] + [
    Input("main-graph-option-marker", "value"),
    Input("main-color", "value"),
    Input("main-highlight-toggle", "value"),
]

filter_dropdown_outputs = [
    Output(dropdown, "options") for dropdown in filter_dropdowns
] + [Output(dropdown, "value") for dropdown in filter_dropdowns]


# client-side url for serving images to the user.
static_image_url = "/images/browse/"

# host-side directory containing those images.
# note this is just ROI browse images for now
# TODO: add a link
image_directory = "./static_in_pro/our_static/img/roi_browse/"

# insert 'settings' / 'global' values for this app into callback functions.
# in Dash, callback functions encapsulate I/O behavior for components and
# are defined separately from components.
# we store our callback functions in plotter.graph.
# typically our convention is that 'global' variables in these functions
# are keyword-only arguments and callback inputs / states are positional
# arguments.


settings = {
    "x_inputs": x_inputs,
    "y_inputs": y_inputs,
    "marker_inputs": marker_inputs,
    "cget": cget,
    "cset": cset,
    # factory functions for plotly figures (which Dash
    # fairly-transparently treats as components).
    # they do not actually fetch or format data;
    # they just do the visual representation.
    "graph_function": main_graph_scatter,
    "spec_graph_function": mspec_graph_line,
    # django model (SQL table + methods) (see plotter.models)
    # containing our spectra.
    # note that insertion of this into functions may end up being
    # a way to generate separate function 'namespaces'
    # in the possible case of, say, wanting to mix mastcam / z data
    # within a single app instance.
    "spec_model": MSpec,
    "image_directory": image_directory,
    # scale factor, in viewport units, for spectrum images
    "base_size": 20,
    "static_image_url": static_image_url,
    # file containing saved searches
    "search_file": "./saves/saved_searches.csv",
}
functions_requiring_settings = [
    control_tabs,
    control_search_dropdowns,
    update_filter_df,
    recalculate_main_graph,
    update_search_options,
    update_search_ids,
    change_calc_input_visibility,
    toggle_search_input_visibility,
    update_spectrum_graph,
    graph_point_to_metadata,
    update_spectrum_images,
    populate_saved_search_drop,
    save_search_tab_state,
    toggle_averaged_filters,
    debug_check_cache,
    handle_main_highlight_save,
    export_graph_csv
]
for function in functions_requiring_settings:
    globals()[function.__name__] = partially_evaluate_from_parameters(
        function, settings
    )


# serve static images using a flask 'route.'
# does defining this function here violate my conventions a little bit? not
# sure.
@app.server.route(static_image_url + "<path:path>")
def static_image_link(path):
    static_folder = os.path.join(os.getcwd(), image_directory)
    return flask.send_from_directory(static_folder, path)


# app layout definition
# the layout property of a dash.Dash object defines how it will
# lay out its components in the browser at runtime.
# it is nominally equivalent to HTML / DOM tree structure,
# although this gets fuzzy at the pointy end.

# very top level, containing the tabs.
# initialize a 'main' / search tab
# (using a component factory function from plotter.components).
# all of its children are created within that function
# by calling other component factory functions.

# noinspection PyTypeChecker
app.layout = html.Div(
    children=[
        dcc.Tabs(
            children=[search_tab(spec_model)],
            value="main_search_tab",
            id="tabs",
        ),
        dcc.Interval(id="interval1", interval=1000, n_intervals=0),
    ]
)

# callback creation section: register functions from plotter.graph with app i/o

# dash.Dash.callback is a factory function
# that returns an impure function that associates the function
# that is its single argument with the components of that dash.Dash
# object whose ids match the ids of the Output, Input, and State
# objects passed to the callback function.

# syntax for this is:
# app.callback(
# list of outputs or single output,
# list of inputs,
# optional list of states
# )(callback_function)
# when the user interacts with the app, changes in inputs and states
# are passed to callback_function as positional arguments
# in the order they are given to app.callback.
# elements of whatever callback_function returns are then passed to outputs,
# also in order.
# changes in inputs trigger function evaluation and subsequent output.
# changes in states do not.

# change visibility of x / y axis calculation inputs
# based on arity of calculation function

# app.callback(
#     Output("fake-output-for-callback-with-only-side-effects-0", "value"),
#     [Input("interval1", "n_intervals")]
# )(debug_check_cache)

for value_class in ["x", "y", "marker"]:
    app.callback(
        [
            Output("main-filter-1-" + value_class + "-container", "style"),
            Output("main-filter-2-" + value_class + "-container", "style"),
            Output("main-filter-3-" + value_class + "-container", "style"),
        ],
        [Input("main-graph-option-" + value_class, "value")],
    )(change_calc_input_visibility)

# trigger redraw of main graph
# on new search, axis calculation change, etc
app.callback(
    Output("main-graph", "figure"),
    # maybe later add an explicit recalculate button?
    [
        *x_inputs,
        *y_inputs,
        *marker_inputs,
        Input({"type": "search-trigger", "index": ALL}, "value"),
        Input({"type": "main-graph-scale-trigger", "index": 0}, "value"),
        Input({'type': 'highlight-trigger', 'index': 0}, 'value'),
        Input('main-graph-bounds', 'value'),
        Input("main-graph-error", "value")
        # Input({'type': 'load-trigger', 'index': 0}, 'value')
    ],
    [State("main-graph", "figure")],
)(recalculate_main_graph)


app.callback(
    [
        Output('main-highlight-description', 'children'),
        Output({"type": "highlight-trigger", "index": 0}, "value")
    ],
    [
        Input({"type": "load-trigger", "index": 0}, "value"),
        Input('main-highlight-save', 'n_clicks'),
    ],
    [State({"type": "highlight-trigger", "index": 0}, "value")]
)(handle_main_highlight_save)


# change visibility of search filter inputs
# based on whether a 'quantitative' or 'qualitative'
# search field is selected
app.callback(
    [
        Output({"type": "term-search", "index": MATCH}, "style"),
        Output({"type": "number-search", "index": MATCH}, "style"),
    ],
    [Input({"type": "field-search", "index": MATCH}, "value")],
)(toggle_search_input_visibility)

# update displayed search options based on selected search field
app.callback(
    [
        Output({"type": "term-search", "index": MATCH}, "options"),
        Output({"type": "number-range-display", "index": MATCH}, "children"),
        Output({"type": "number-search", "index": MATCH}, "value"),
    ],
    [
        Input({"type": "field-search", "index": MATCH}, "value"),
        Input({"type": "load-trigger", "index": 0}, "value"),
    ],
    [
        State({"type": "number-search", "index": MATCH}, "value"),
    ],
)(update_search_options)

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
)(update_search_ids)

# change and reset options on averaging request
app.callback(
    filter_dropdown_outputs,
    [
        Input("main-graph-average", "value"),
        # Input('interval1', 'n_intervals')
    ],
)(toggle_averaged_filters)

app.callback(
    Output({"type": "main-graph-scale-trigger", "index": 0}, "value"),
    [
        Input({"type": "load-trigger", "index": 0}, "value"),
        Input("main-graph-scale", "value"),
        Input("main-graph-average", "value"),
    ],
    [State({"type": "main-graph-scale-trigger", "index": 0}, "value")],
)(update_filter_df)

# handle creation and removal of search filters
app.callback(
    [
        Output("search-controls-container", "children"),
        Output({"type": "submit-search", "index": 1}, "n_clicks"),
    ],
    [
        Input("add-param", "n_clicks"),
        Input("clear-search", 'n_clicks'),
        Input({"type": "remove-param", "index": ALL}, "n_clicks"),
    ],
    [
        State("search-controls-container", "children"),
        State({"type": "submit-search", "index": 1}, "n_clicks"),
    ],
)(control_search_dropdowns)

# make graph viewer tabs
app.callback(
    [
        Output("tabs", "children"),
        Output("tabs", "value"),
        Output({"type": "load-trigger", "index": 0}, "value"),
    ],
    [
        Input("viewer-open-button", "n_clicks"),
        Input({"type": "tab-close-button", "index": ALL}, "n_clicks"),
        Input("load-search-load-button", "n_clicks"),
    ],
    [
        State("tabs", "children"),
        State("load-search-drop", "value"),
        State({"type": "load-trigger", "index": 0}, "value"),
    ],
)(control_tabs)

# debug printer
# app.callback(
#     Output('fake-output-for-callback-with-only-side-effects-1', 'children'),
#     [Input('load-search-drop', 'value')]
# )(print_callback)


# point-hover functions.
# right now main and view graph hover functions are basically duplicates,
# but i'm reserving the possibility that they'll have different behaviors later
# app.callback(
#     Output({'type': 'main-spec-image-left', 'index': 0}, "children"),
#     [Input('main-graph', "hoverData")]
# )(update_spectrum_images)
#
# app.callback(
#     Output({'type': 'main-spec-image-right', 'index': 0}, "children"),
#     [Input('main-graph', "hoverData")]
# )(update_spectrum_images)

app.callback(
    Output({"type": "main-spec-image", "index": 0}, "children"),
    [Input("main-graph", "hoverData")],
)(update_spectrum_images)

app.callback(
    Output({"type": "main-spec-print", "index": 0}, "children"),
    [Input("main-graph", "hoverData")],
)(graph_point_to_metadata)

app.callback(
    Output({"type": "main-spec-graph", "index": 0}, "figure"),
    [
        Input("main-graph", "hoverData"),
        Input("main-spec-scale", "value"),
        Input("main-spec-average", "value"),
        Input("main-spec-error", "value")
    ],
)(update_spectrum_graph)

app.callback(
    Output({"type": "view-spec-image", "index": MATCH}, "children"),
    [
        Input({"type": "view-graph", "index": MATCH}, "hoverData"),
    ]
)(update_spectrum_images)

app.callback(
    Output({"type": "view-spec-print", "index": MATCH}, "children"),
    [Input({"type": "view-graph", "index": MATCH}, "hoverData")],
)(graph_point_to_metadata)

app.callback(
    Output({"type": "view-spec-graph", "index": MATCH}, "figure"),
    [
        Input({"type": "view-graph", "index": MATCH}, "hoverData"),
        Input("main-spec-scale", "value"),
        Input("main-spec-average", "value"),
        Input("main-spec-error", "value")
    ],
)(update_spectrum_graph)

app.callback(
    Output({'type': "save-trigger", 'index': 0}, 'value'),
    [Input("save-search-save-button", "n_clicks")],
    [
        State("save-search-name-input", "value"),
        State({'type': "save-trigger", 'index': 0}, 'value')
    ]
)(save_search_tab_state)

app.callback(
    Output("load-search-drop", "options"),
    [
        Input({'type': 'save-trigger', 'index': 0}, "value"),
        Input('fire-on-load', 'children')
    ],
)(populate_saved_search_drop)

app.callback(
    Output("fake-output-for-callback-with-only-side-effects-1", "children"),
    [Input("main-export-csv", "n_clicks")],
    [State("main-graph", "selectedData")]
)(export_graph_csv)

app.callback(
    [
        Output({"type": "collapsible-panel", "index": MATCH}, "style"),
        Output({"type": "collapse-arrow", "index": MATCH}, "style"),
        Output({"type": "collapse-text", "index": MATCH}, "style")
    ],
    [Input({'type': "collapse-div", "index": MATCH}, "n_clicks")],
    [
        State({'type': "collapsible-panel", "index": MATCH}, "style"),
        State({"type": "collapse-arrow", "index": MATCH}, "style"),
        State({"type": "collapse-text", "index": MATCH}, "style")
    ]
)(toggle_panel_visibility)

# app.run_server(debug=True, use_reloader=False,
# dev_tools_silence_routes_logging=True)
app.run_server(dev_tools_silence_routes_logging=True, port=10001)
