"""
functions used for selecting, manipulating, and drawing spectral data
within plotly-dash objects. this module is _separate_ from app structure
definition_ and, to the extent possible, components. considering where
exactly...! these functions are partially defined and/or passed to callback
decorators in order to generate flow control within a dash app.
"""

import datetime as dt
from copy import deepcopy
from functools import reduce
from operator import or_
from typing import TYPE_CHECKING, Any, Callable, Optional

import dash
import dash_html_components as html
import pandas as pd
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate

from plotter.components import (
    parse_model_quant_entry,
    search_parameter_div,
    viewer_tab,
    search_tab,
)
from plotter_utils import (
    djget,
    dict_to_paragraphs,
    rows,
    keygrab,
    first,
    multiple_field_search,
    pickctx,
    not_blank,
    not_triggered,
    trigger_index,
    triggered_by,
    filter_null_attributes,
    seconds_since_beginning_of_day,
    arbitrarily_hash_strings,
    none_to_quote_unquote_none,
    field_values, fetch_css_variables,
)

if TYPE_CHECKING:
    from django.db.models import Model
    from plotter.models import MSpec
    from django.db.models.query import QuerySet
    import flask_caching


css_variables = fetch_css_variables()
COLORBAR_SETTINGS = {
    'tickfont': {'family': 'Fira Mono', 'color': css_variables['midnight-ochre']}
}


# ### cache functions ###

# many of the other functions in this module take outputs of these two
# functions as arguments. returning a function that calls a specific cache
# defined in-app allows us to share data between defined clusters of dash
# objects. the specific cache is in some sense a set of pointers that serves
# as a namespace. this, rather than a global variable or variables, is used
# because Flask does not guarantee thread safety of globals.


def cache_set(cache: "flask_caching.Cache") -> Callable[[str, Any], bool]:
    def cset(key: str, value: Any) -> bool:
        return cache.set(key, value)

    return cset


def cache_get(cache: "flask_caching.Cache") -> Callable[[str], Any]:
    def cget(key: str) -> Any:
        return cache.get(key)

    return cget


def truncate_queryset_for_missing_properties(
    settings: dict, queryset: "QuerySet", prefix: str, suffix: str
):
    axis_option = settings[prefix + "-graph-option-" + suffix]
    props = keygrab(
        queryset.model.marker_value_properties, "value", axis_option
    )
    if props["type"] == "method":
        # we assume here that 'methods' all take a spectrum's filter names
        # as arguments, and have arguments in an order corresponding to the
        # inputs.
        filt_args = [
            settings[prefix + "-filter-" + str(ix) + "-" + suffix]
            for ix in range(1, props["arity"] + 1)
        ]
        return filter_null_attributes(queryset, filt_args)
    elif props["type"] == "self_property":
        return filter_null_attributes(queryset, [axis_option])
    elif props["type"] == "parent_property":
        return filter_null_attributes(queryset, [axis_option], "observation")


def make_axis(
    settings: dict, queryset: "QuerySet", prefix: str, suffix: str
) -> Optional[list[float]]:
    """
    make an axis for one of our graphs by looking at a bunch of objects,
    usually Spectrum instances. this expects a queryset that has already
    been processed by truncate_queryset_for_missing_properties().
    """
    # what is requested function or property?
    axis_option = settings[prefix + "-graph-option-" + suffix]
    # what are the characteristics of that function or property?
    props = keygrab(
        queryset.model.marker_value_properties, "value", axis_option
    )
    if props["type"] == "method":
        # we assume here that 'methods' all take a spectrum's filter names
        # as arguments, and have arguments in an order corresponding to the
        # inputs.
        filt_args = [
            settings[prefix + "-filter-" + str(ix) + "-" + suffix]
            for ix in range(1, props["arity"] + 1)
        ]
        return [
            getattr(spectrum, props["value"])(*filt_args)
            for spectrum in queryset
        ]
    if props["type"] == "parent_property":
        vals = [
            getattr(spectrum.observation, props["value"])
            for spectrum in queryset
        ]

    else:
        vals = [getattr(spectrum, props["value"]) for spectrum in queryset]
    # vals.sort()
    return vals


def redorblue(value, container):
    """placeholder color function"""
    if value in container:
        return "red"
    return "blue"


def make_marker_properties(settings, queryset, prefix, suffix):
    """
    this expects a queryset that has already
    been processed by truncate_queryset_for_missing_properties().
    """
    marker_option = settings[prefix + "-graph-option-" + suffix]
    props = keygrab(
        queryset.model.marker_value_properties, "value", marker_option
    )
    """
    TODO: this stuff should be split off into a function that 
    make_marker_properties and make_axis both call
    """
    # it would really be better to do this in components
    # but is difficult because you have to instantiate the colorbar somewhere
    # it would also be better to style with CSS but it seems like plotly
    # really wants to put element-level style declarations on graph ticks!
    colorbar_dict = COLORBAR_SETTINGS
    if props["type"] == "method":
        # we assume here that 'methods' all take a spectrum's filter names
        # as arguments, and have arguments in an order corresponding to
        # the inputs.
        filt_args = [
            settings[prefix + "-filter-" + str(ix) + "-" + suffix]
            for ix in range(1, props["arity"] + 1)
        ]
        property_list = [
            getattr(spectrum, props["value"])(*filt_args)
            for spectrum in queryset
        ]
    elif props["type"] == "parent_property":
        property_list = [
            getattr(spectrum.observation, props["value"])
            for spectrum in queryset
        ]
    else:
        property_list = [
            getattr(spectrum, props["value"]) for spectrum in queryset
        ]
    if props["value_type"] == "qual":
        string_hash, color_indices = arbitrarily_hash_strings(
            none_to_quote_unquote_none(property_list)
        )
        colorbar_dict |= {
            "tickvals": list(string_hash.values()),
            "ticktext": list(string_hash.keys()),
        }
    else:
        if isinstance(property_list[0], dt.time):
            property_list = list(
                map(seconds_since_beginning_of_day, property_list)
            )
        color_indices = property_list
    colormap = settings[prefix + "-color.value"]
    return {
        "marker": {
            "color": color_indices,
            "colorscale": colormap,
            "colorbar": go.scatter.marker.ColorBar(**colorbar_dict),
        }
    }


def sibling_ids(spec_id, cget):
    """wrapper for getting sets of spectra siblings precalculated in
    update_queryset"""
    return first(lambda x: spec_id in x, cget("sibling_set"))


def main_graph_hover_styler(ctx, cget):
    """
    this is drastically inefficient, i would think. perhaps there's a more
    performant way to do it
    in some of the marker controls.
    """

    hovered_point = ctx.triggered[0]["value"]["points"][0]["customdata"]
    figure = go.Figure(ctx.states["main-graph.figure"])
    sibs = sibling_ids(hovered_point, cget)
    figure.update_traces(
        marker={
            "color": [
                redorblue(point, sibs)
                for point in figure["data"][0]["customdata"]
            ]
        }
    )
    return figure


# this is somewhat nasty.
# is there a cleaner way to do this?
# flow control becomes really hard if we break the function up.
# it requires triggers spread across multiple divs or cached globals
# and is much uglier than even this. dash's restriction on callbacks to a
# single output
# makes it even worse. probably the best thing to do
# in the long run is to treat this basically as a dispatch function.
def recalculate_main_graph(
    *args,
    x_inputs,
    y_inputs,
    marker_inputs,
    graph_function,
    cget,
    cset,
    record_settings=True,
):
    ctx = dash.callback_context
    if ctx.triggered[0]["prop_id"] == "main-graph.hoverData":
        raise PreventUpdate
        # fun but maybe slow and so presently disabled
        # return main_graph_hover_styler(ctx, cget)
    queryset = cget("queryset")
    x_settings = pickctx(ctx, x_inputs)
    y_settings = pickctx(ctx, y_inputs)
    marker_settings = pickctx(ctx, marker_inputs)
    # this is for functions like describe_current_graph
    # that display or recall
    # the settings used to generate a graph
    # we can probably get away without any fancy flow control
    # b/c memoization...if it turns out that hitting the cache this way sucks,
    # we will add it.
    truncated_queryset = queryset
    for settings, suffix in zip(
        [x_settings, y_settings, marker_settings],
        ["x.value", "y.value", "marker.value"],
    ):
        truncated_queryset = truncate_queryset_for_missing_properties(
            settings, truncated_queryset, prefix="main", suffix=suffix
        )
    x_axis = make_axis(
        x_settings, truncated_queryset, prefix="main", suffix="x.value"
    )
    y_axis = make_axis(
        y_settings, truncated_queryset, prefix="main", suffix="y.value"
    )
    marker_properties = make_marker_properties(
        marker_settings,
        truncated_queryset,
        prefix="main",
        suffix="marker.value",
    )
    # these text and customdata choices are likely placeholders
    text = [
        spec.observation.seq_id + " " + spec.color
        for spec in truncated_queryset
    ]
    customdata = [spec.id for spec in truncated_queryset]
    # this case is most likely shortly after page load
    # when not everything is filled out
    # if not (x_axis and y_axis):
    #     raise PreventUpdate

    # for functions that (perhaps asynchronously) fetch the state of the graph.
    # this is another perhaps ugly flow control thing!
    if record_settings:
        for parameter in (
            "x_settings",
            "y_settings",
            "marker_settings",
            "x_axis",
            "y_axis",
            "marker_properties",
            "text",
            "customdata",
            "graph_function",
        ):
            cset(parameter, locals()[parameter])
    graph_layout = ctx.states["main-graph.figure"]["layout"]

    # automatically reset graph zoom only if we're loading the page or
    # changing options and therefore scales
    print(ctx.triggered)
    if not_triggered() or (
        ctx.triggered[0]["prop_id"]
        in ["main-graph-option-y.value", "main-graph-option-x.value"]
    ):
        zoom = None
    else:
        zoom = (graph_layout["xaxis"]["range"], graph_layout["yaxis"]["range"])
    return graph_function(
        x_axis, y_axis, marker_properties, text, customdata, zoom
    )


def toggle_search_input_visibility(field, *, spec_model):
    """
    toggle between showing and hiding term drop down / number range boxes.
    right now just returns simple dicts to the dash callback based on the
    field's value type
    (quant or qual) as appropriate.
    """
    if not field:
        raise PreventUpdate

    if (
        keygrab(spec_model.searchable_fields, "label", field)["value_type"]
        == "quant"
    ):
        return [{"display": "none"}, {}]
    return [{}, {"display": "none"}]


def spectrum_values_range(queryset, field, field_type):
    """
    returns minimum and maximum values of property within queryset of spectra.
    for cueing or aiding searches.
    """
    if field_type == "parent_property":
        values_list = sorted(
            [
                getattr(getattr(item, "observation"), field)
                for item in queryset.prefetch_related("observation")
            ]
        )
    else:
        values_list = sorted([item[field] for item in queryset.objects.all()])
    return values_list[0], values_list[-1]


def update_search_options(
    field, _load_trigger_index, current_quant_search, *, cget
):
    """
    populate term values and parameter range as appropriate when different
    fields are selected in the search interface
    currently this does _not_ cascade according to selected terms. this may
    or may not be desirable
    """
    if not_triggered():
        raise PreventUpdate
    if not field:
        raise PreventUpdate
    queryset = cget("queryset")
    is_loading = (
        "load-trigger" in dash.callback_context.triggered[0]["prop_id"]
    )
    props = keygrab(queryset.model.searchable_fields, "label", field)
    # if it's a field we do number interval searches on, reset term
    # interface and show number ranges in the range display. but don't reset
    # the number entries if we're in the middle of a load!
    if props["value_type"] == "quant":
        if is_loading:
            search_text = current_quant_search
        else:
            search_text = ""
        return [
            [{"label": "any", "value": "any"}],
            "minimum/maximum: "
            + str(
                spectrum_values_range(
                    queryset.model.objects.all(), field, props["type"]
                )
            )
            # TODO: this should be a hover-over tooltip eventually
            + """ e.g., '100-200' or '100, 105, 110'""",
            search_text,
        ]

    # otherwise, populate the term interface and reset the range display and
    # searches

    # TODO: probably this should be modified to nicely check
    #  "parent_property" etc

    return [
        field_values(queryset.model.objects.all(), field, "observation"),
        "",
        "",
    ]


def change_calc_input_visibility(calc_type, *, spec_model):
    """
    turn visibility of filter dropdowns (and later other inputs)
    on and off in response to changes in arity / type of
    requested calc
    """
    props = keygrab(spec_model.marker_value_properties, "value", calc_type)
    # 'methods' are specifically those methods of spec_model
    # that take its filters as arguments
    if props["type"] == "method":
        return [
            {"display": "flex", 'flex-direction': 'column'}
            if x < props["arity"]
            else {"display": "none"}
            for x in range(3)
        ]
    return [{"display": "none"} for _ in range(3)]


def non_blank_search_parameters(parameters):
    entry_keys = ["term", "begin", "end", "value_list"]
    return [
        parameter
        for parameter in parameters
        if reduce(or_, [not_blank(parameter.get(key)) for key in entry_keys])
    ]


def handle_graph_search(model, parameters):
    """
    dispatcher / manager for user-issued searches within the graph interface.
    fills fields from model definition and feeds resultant list to a general-
    purpose search function.
    """
    # add value_type and type information to dictionaries (based on
    # search properties defined in the model)
    for parameter in parameters:
        field = parameter.get("field")
        if field:
            props = keygrab(model.searchable_fields, "label", field)
            parameter["value_type"] = props["value_type"]
            # format references to parent observation appropriately for Q
            # objects
            if props["type"] == "parent_property":
                parameter["field"] = "observation__" + field
    # toss out 'any' entries, blank strings, etc.
    # -- they do not restrict the search
    parameters = non_blank_search_parameters(parameters)
    # do we have any actual constraints? if not, return the entire data set
    if not parameters:
        return model.objects.all()
    # otherwise, actually perform a search
    return multiple_field_search(
        model.objects.all().prefetch_related("observation"), parameters
    )


def make_sibling_set(observations):
    """
    returns a set of tuples, each containing the child spectra of one
    observation.
    for pairing spectra with their siblings without hitting the database
    repeatedly.
    """
    return set(
        [tuple([spec.id for spec in obs.spectra()]) for obs in observations]
    )


def spectrum_queryset_siblings(queryset):
    return make_sibling_set(set([spec.observation for spec in queryset]))


def update_queryset(
    _search_n_clicks,
    _load_trigger_index,
    fields,
    terms,
    quant_search_entries,
    search_trigger_dummy_value,
    *,
    cget,
    cset,
    spec_model,
):
    """
    updates the spectra displayed in the graph view.
    """
    # don't do anything on page load
    if not_triggered():
        raise PreventUpdate
    # or if a blank request is issued
    if not (fields and (terms or quant_search_entries)):
        raise PreventUpdate
    # construct a list of search parameters from the filled inputs
    # (ideally totally non-filled inputs would also be rejected by
    # handle_graph_search)
    # search_list = [
    #     {"field": field, "term": term, "begin": begin, "end": end}
    #     for field, term, begin, end in zip(
    #         fields, terms, begin_numbers, end_numbers
    #     )
    #     if (not_blank(field) and (not_blank(term) or not_blank(begin)))
    # ]

    entries = [
        parse_model_quant_entry(entry) for entry in quant_search_entries
    ]
    search_list = [
        {"field": field, "term": term, **entry}
        for field, term, entry in zip(fields, terms, entries)
        if not_blank(field) and (not_blank(term) or not_blank(entry))
    ]
    # if every search parameter is blank, don't do anything
    # if not search_list:
    #     raise PreventUpdate

    # if the search parameters have changed or if it's a new load, make a
    # new queryset and trigger graph update using copy.deepcopy here to
    # avoid passing doubly-ingested input back after the check although on
    # the other hand it should be memoized -- but still but yes seriously it
    # should be memoized
    search = handle_graph_search(spec_model, deepcopy(search_list))
    ctx = dash.callback_context
    if (
        set(search) != set(cget("queryset"))
        or "load-trigger" in ctx.triggered[0]["prop_id"]
    ):
        cset(
            "queryset",
            search.prefetch_related("observation"),
        )
        # save search parameters for graph description
        cset("search_parameters", search_list)
        # precalculate sets of 'sibling' spectra
        cset("sibling_set", spectrum_queryset_siblings(cget("queryset")))
        if not search_trigger_dummy_value:
            return 1
        return search_trigger_dummy_value + 1

    raise PreventUpdate


def add_dropdown(children, spec_model, cget, cset):
    """
    adds another dropdown for search constraints.
    """
    # check with the cache in order to pick an index
    # this is because resetting the layout for tabs destroys n_clicks
    # could parse the page instead but i think this is better
    index = cget("search_parameter_index")
    if not index:
        index = 1
    else:
        index = index + 1

    searchable_fields = spec_model.searchable_fields
    children.append(search_parameter_div(index, searchable_fields, None))
    cset("search_parameter_index", index)
    return children


def remove_dropdown(index, children, cget, cset):
    """
    remove a search constraint dropdown, and its attendant constraints on
    searches.
    """
    param_to_remove = None
    for param in children:
        if param["props"]["id"]["index"] == index:
            param_to_remove = param
    if not param_to_remove:
        raise ValueError("Got the wrong param-to-remove index from somewhere.")

    new_children = deepcopy(children)
    new_children.remove(param_to_remove)
    return new_children


def control_search_dropdowns(
    _add_clicks,
    _remove_clicks,
    children,
    search_trigger_clicks,
    *,
    spec_model,
    cget,
    cset,
):
    """
    dispatch function for building a new list of search dropdown components
    based on requests to add or remove components.
    returns the new list and an incremented value to trigger update_queryset
    """
    if not_triggered():
        raise PreventUpdate
    if search_trigger_clicks is None:
        search_trigger_clicks = 0
    if triggered_by("add-param"):
        drops = add_dropdown(children, spec_model, cget, cset)
    elif triggered_by("remove-param"):
        ctx = dash.callback_context
        index = trigger_index(ctx)
        drops = remove_dropdown(index, children, cget, cset)
    else:
        raise PreventUpdate
    return drops, search_trigger_clicks + 1


# ## individual-spectrum display functions


def spectrum_from_graph_event(
    event_data: dict, spec_model: "Model"
) -> "Model":
    """
    dcc.Graph event data (e.g. hoverData), plotter.Spectrum class ->
    plotter.Spectrum instance
    this function assumes it's getting data from a browser event that
    highlights
    a single graphed point, like clicking it or hovering on it, and returns
    the associated Spectrum object.
    """
    # the graph's customdata property should contain numbers corresponding
    # to database pks of spectra of associated points.
    return djget(
        spec_model, event_data["points"][0]["customdata"], "id", "get"
    )


def graph_point_to_metadata(event_data, *, spec_model, style=None):
    if style is None:
        style = {"margin": 0, "fontSize": 14}
    # parses hoverdata, clickdata, etc from main graph
    # into html <p> elements containing metadata of associated Spectrum
    if not event_data:
        raise PreventUpdate
    return dict_to_paragraphs(
        spectrum_from_graph_event(event_data, spec_model).metadata_dict(),
        style,
    )


def update_spectrum_graph(event_data, *, spec_model, spec_graph_function):
    if not event_data:
        raise PreventUpdate
    spectrum = spectrum_from_graph_event(event_data, spec_model)
    return spec_graph_function(spectrum)


def make_mspec_browse_image_components(
    mspec, image_directory, base_size, static_image_url
):
    """
    MSpec object, size factor (viewport units), image directory ->
    pair of dash html.Img components containing the spectrum-reduced
    images associated with that object, pathed to the static image
    route defined in the live app instance
    """
    file_info = mspec.overlay_browse_file_info(image_directory)
    image_div_children = []
    for eye in ["left", "right"]:
        try:
            # size = file_info[eye + "_size"]
            filename = static_image_url + file_info[eye + "_file"]
        except KeyError:
            # size = (480, 480)
            filename = static_image_url + "missing.jpg"
        # aspect_ratio = size[0] / size[1]
        # width = base_size * aspect_ratio
        # height = base_size / aspect_ratio
        image_div_children.append(
            html.Img(
                src=filename,
                style={"width": "50%", "height": "50%"},
                id="spec-image-" + eye,
            )
        )
        component = html.Div(
            children=image_div_children,
        )

    return component


def update_spectrum_images(
    event_data, *, spec_model, image_directory, base_size, static_image_url
):
    """
    just a callback-responsive wrapper to make_mspec_browse_image_components --
    probably we should actually put that function on the model
    """
    if not event_data:
        raise PreventUpdate
    # type checking just can't handle django class inheritance
    # noinspection PyTypeChecker
    spectrum: "MSpec" = spectrum_from_graph_event(event_data, spec_model)
    return make_mspec_browse_image_components(
        spectrum, image_directory, base_size, static_image_url
    )


class SPlot:
    """
    class that holds a queryset of spectra with search parameters
    and axis relationships. probably also eventually visual settings.
    used for saving and restoring plots.
    """

    def __init__(self, arg_dict):
        # maybe eventually we define this more flexibly but better to be
        # strict for now
        for parameter in self.canonical_parameters:
            setattr(self, parameter, arg_dict[parameter])

    def axes(self):
        return self.x_axis, self.y_axis

    def graph(self):
        return self.graph_function(
            self.x_axis,
            self.y_axis,
            self.marker_properties,
            self.text,
            self.customdata,
        )

    def settings(self):
        return {
            parameter: getattr(self, parameter)
            for parameter in self.setting_parameters
        }

    canonical_parameters = (
        "x_axis",
        "y_axis",
        "marker_properties",
        "text",
        # customdata is typically a list of the pks of the spectra in axis
        # order
        "customdata",
        "queryset",
        "search_parameters",
        "x_settings",
        "y_settings",
        "marker_settings",
        "graph_function",
    )

    setting_parameters = (
        "search_parameters",
        "x_settings",
        "y_settings",
        "marker_settings",
    )


def describe_current_graph(cget):
    """
    note this this relies on cached 'globals' from recalculate_graph
    and update_queryset! if this turns out to be an excessively ugly flow
    control
    solution, we could instead turn it into a callback that dynamically
    monitors state
    of the same objects they monitor the state of...but that parallel
    structure seems
    worse.
    """
    return {
        parameter: cget(parameter) for parameter in SPlot.canonical_parameters
    }


def open_viewer_tab(tabs, cget, cset):
    graph_params = describe_current_graph(cget)
    # don't open a tab on page load or if the graph is deformed
    if not graph_params["x_axis"]:
        raise PreventUpdate
    splot = SPlot(describe_current_graph(cget))
    # what viewers are open? check cache.
    graph_viewers = cget("open_graph_viewers")
    if graph_viewers:
        index = max(graph_viewers) + 1
        graph_viewers.append(index)
    else:
        index = 1
        graph_viewers = [1]
    # make a new tab at the first available index with the current graph
    new_tab = viewer_tab(index, splot)

    new_tabs = deepcopy(tabs)
    new_tabs.append(new_tab)
    # let cache know you opened a new viewer.
    graph_viewers.append(index)
    cset("open_graph_viewers", graph_viewers)
    return new_tabs, "viewer_tab_" + str(index)


def close_viewer_tab(index, tabs, cget, cset):
    # parse the frankly unfortunate dash input structure for the index of
    # the tab whose button this is
    tab_to_close = None
    for tab in tabs:
        if not isinstance(tab["props"]["id"], dict):
            continue
        if tab["props"]["id"]["type"] != "viewer-tab":
            continue
        if tab["props"]["id"]["index"] == index:
            tab_to_close = tab
    if not tab_to_close:
        raise ValueError("Got the wrong tab-to-close index from somewhere.")

    new_tabs = deepcopy(tabs)
    new_tabs.remove(tab_to_close)
    # let cache know you closed this in case anyone asks
    graph_viewers = cget("open_graph_viewers")
    graph_viewers.remove(index)
    cset("open_graph_viewers", graph_viewers)
    return new_tabs, "main_search_tab"


def save_search_tab_state(
    _n_clicks, cget, filename="./saves/saved_searches.csv"
):
    """
    fairly permissive right now. this saves current search-tab state to a
    file as csv,
    which can then be reloaded by make_loaded_search_tab.
    """

    if not_triggered():
        raise PreventUpdate

    try:
        state_variable_names = [
            *cget("x_settings").keys(),
            *cget("y_settings").keys(),
            *cget("marker_settings").keys(),
            "search_parameters",
        ]

        state_variable_values = [
            *cget("x_settings").values(),
            *cget("y_settings").values(),
            *cget("marker_settings").values(),
            # just passing a list here this makes the pd.DataFrame constructor
            # interpret it as two rows with the same x_setting and y_setting
            # values, which is a cool feature, but not the one we're looking
            # for, so we 'serialize' it
            str(cget("search_parameters")),
        ]
    # silently failing on incompletely-filled-out tabs is fine for now
    # but we would probably prefer an error message in the future
    except AttributeError:
        raise PreventUpdate
    try:
        saved_searches = pd.read_csv(filename)
    except FileNotFoundError:
        saved_searches = pd.DataFrame(
            columns=state_variable_names + ["timestamp"]
        )
    state_line = pd.DataFrame(
        {
            parameter: value
            for parameter, value in zip(
                state_variable_names, state_variable_values
            )
        },
        index=[0],
    )
    state_line["timestamp"] = dt.datetime.now().strftime("%D %H:%M:%S")
    appended_df = pd.concat([saved_searches, state_line], axis=0)
    appended_df.to_csv(filename, index=False)
    return 1


def populate_saved_search_drop(_n_clicks, *, search_file):
    try:
        options = [
            {"label": row["timestamp"], "value": row_index}
            for row_index, row in enumerate(rows(pd.read_csv(search_file)))
        ]
    except FileNotFoundError:
        options = []
    return options


def make_loaded_search_tab(row, spec_model, search_file):
    """makes a search tab with preset values from a saved search."""
    saved_searches = pd.read_csv(search_file)
    row_dict = rows(saved_searches)[row].to_dict()
    return search_tab(spec_model, row_dict)


def load_saved_search(tabs, row, spec_model, search_file):
    """loads a search tab and replaces existing search tab with it"""
    new_tab = make_loaded_search_tab(row, spec_model, search_file)
    if len(tabs) > 1:
        old_tabs = deepcopy(tabs[1:])
        new_tabs = [new_tab] + old_tabs
    else:
        new_tabs = [new_tab]
    return new_tabs, "main_search_tab"


def control_tabs(
    _n_clicks_open,
    _n_clicks_close,
    _n_clicks_load,
    tabs,
    load_row,
    load_trigger_index,
    *,
    spec_model,
    search_file,
    cget,
    cset,
):
    """
    control loading, opening, and closing viewer tabs.

    TODO, maybe: add ability to just open selectedData from main-graph
    """
    if not_triggered():
        raise PreventUpdate
    ctx = dash.callback_context
    trigger = ctx.triggered[0]["prop_id"]
    if "viewer-open-button" in trigger:
        return *open_viewer_tab(tabs, cget, cset), load_trigger_index
    if "tab-close-button" in trigger:
        index = trigger_index(ctx)
        new_tabs, active_tab_value = close_viewer_tab(index, tabs, cget, cset)
        return (
            new_tabs,
            active_tab_value,
            load_trigger_index,
        )
    if "load-search-load-button" in trigger:
        # don't try anything if nothing in the saved search dropdown is
        # selected
        if not load_row:
            raise PreventUpdate
        # explicitly trigger graph recalculation call in update_queryset
        if not load_trigger_index:
            load_trigger_index = 0
        load_trigger_index = load_trigger_index + 1
        new_tabs, active_tab_value = load_saved_search(
            tabs, load_row, spec_model, search_file
        )
        # this cache parameter is for semi-asynchronous flow control of the
        # load process
        # without having to literally
        # have one dispatch callback function for the entire app
        cset("load_state", {"update_search_options": True})
        return new_tabs, active_tab_value, load_trigger_index
