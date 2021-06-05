from inspect import getmembers, isfunction, getmodule
from pathlib import Path

import plotter.application.registry
import plotter.callbacks
from multidex_utils import partially_evaluate_from_parameters
from plotter.application.structure import (
    X_INPUTS,
    Y_INPUTS,
    MARKER_INPUTS,
    GRAPH_DISPLAY_INPUTS,
    STATIC_IMAGE_URL,
)
from plotter.components import main_scatter_graph, spectrum_line_graph


def configure_cache(cache_subdirectory):
    """configure cache for a particular instance of multidex."""

    # we are using flask-caching to share state between callbacks
    # because dash refuses to enforce thread safety in python globals.
    # memcached or another backend can be implemented for improved speed
    # if it ever matters.
    return {
        "CACHE_TYPE": "filesystem",
        "CACHE_DIR": ".cache/" + cache_subdirectory,
        "CACHE_DEFAULT_TIMEOUT": 0,
        "CACHE_THRESHOLD": 0,  # important for filesystem backend
    }


def register_everything(app, configured_functions):
    """
    register all callback functions with appropriate components of the app,
    as defined in plotter.application.registry
    """
    for name, func in configured_functions.items():
        register = getattr(plotter.application.registry, "register_" + name)
        # special cases
        if name == "change_calc_input_visibility":
            for value_class in ["x", "y", "marker"]:
                register(app, func, value_class)
            continue
        register(app, func)


def configure_callbacks(cget, cset, spec_model):
    """
    insert 'settings' / 'global' values for this app into callback functions.
    typically our convention is that 'global' variables in callbacks
    are keyword-only and callback inputs / states are positional.
    """

    settings = {
        "x_inputs": X_INPUTS,
        "y_inputs": Y_INPUTS,
        "marker_inputs": MARKER_INPUTS,
        "graph_display_inputs": GRAPH_DISPLAY_INPUTS,
        "cget": cget,
        "cset": cset,
        # factory functions for plotly figures
        "graph_function": main_scatter_graph,
        "spec_graph_function": spectrum_line_graph,
        # django model containing our spectra.
        "spec_model": spec_model,
        # host-side directory containing context images
        "image_directory": "plotter/application/assets/browse/" \
                           + spec_model.instrument.lower(),
        # scale factor, in viewport units, for spectrum images
        "base_size": 20,
        "static_image_url": STATIC_IMAGE_URL,
        # file containing saved searches
        "search_file": "./saves/" + cget("spec_model_name") + "_searches.csv",
    }
    return {
        name: partially_evaluate_from_parameters(func, settings)
        for name, func in [
            (name, func)
            for name, func in getmembers(plotter.callbacks)
            if isfunction(func) and (getmodule(func) == plotter.callbacks)
        ]
    }
