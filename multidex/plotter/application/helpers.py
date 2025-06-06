from inspect import getmembers, isfunction, getmodule

from multidex._pathref import MULTIDEX_ROOT
import multidex.plotter.application.registry
import multidex.plotter.callbacks
from multidex.multidex_utils import partially_evaluate_from_parameters
from multidex.plotter.application.structure import (
    X_INPUTS,
    Y_INPUTS,
    MARKER_INPUTS,
    GRAPH_DISPLAY_INPUTS,
    STATIC_IMAGE_URL, HIGHLIGHT_INPUTS,
)
from multidex.plotter.application.structure import REGRESSION_LINE_INPUTS


def register_everything(app, configured_functions):
    """
    register all callback functions with appropriate components of the app,
    as defined in plotter.application.registry
    """
    for name, func in configured_functions.items():
        register = getattr(
            multidex.plotter.application.registry, "register_" + name
        )
        # special cases
        if name == "change_calc_input_visibility":
            for value_class in ["x", "y", "marker"]:
                register(app, func, value_class)
            continue
        register(app, func)


def configure_flask_cache(cache_subdirectory):
    """configure cache for a particular instance of multidex."""

    # we are using flask-caching to share state between callbacks
    # because dash refuses to enforce thread safety in python globals.
    # memcached or another backend can be implemented for improved speed
    # if it ever matters.
    return {
        "CACHE_DIR": str(
            (MULTIDEX_ROOT.parent / ".cache" / cache_subdirectory).absolute()
        ),
        "CACHE_DEFAULT_TIMEOUT": 0,
        "CACHE_THRESHOLD": 0,  # important for filesystem backend
        "CACHE_IGNORE_ERRORS": False
    }


def register_clientside_callbacks(app):
    # TODO: move this into external scripts?
    js_callbacks = [
        # TODO, probably: remove!!!!!
        # "record_graph_size_and_trigger_save",
        "drag_spec_print",
        'hide_spec_print'
    ]
    for name in js_callbacks:
        register = getattr(
            multidex.plotter.application.registry, "register_" + name
        )
        register(app)


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
        "highlight_inputs": HIGHLIGHT_INPUTS,
        "graph_display_inputs": GRAPH_DISPLAY_INPUTS,
        "regression_line_inputs": REGRESSION_LINE_INPUTS,
        "cget": cget,
        "cset": cset,
        # django model containing our spectra.
        "spec_model": spec_model,
        # host-side directory containing context images
        "image_directory": str(
            MULTIDEX_ROOT
            / "plotter/application/assets/browse"
            / spec_model.instrument.lower()
        ),
        # scale factor, in viewport units, for spectrum images
        "base_size": 20,
        "static_image_url": STATIC_IMAGE_URL,
        # path containing saved searches
        "search_path": str(
            MULTIDEX_ROOT
            / "saves"
            / spec_model.instrument.lower()
        ),
    }
    return {
        name: partially_evaluate_from_parameters(func, settings)
        for name, func in [
            (name, func)
            for name, func in getmembers(multidex.plotter.callbacks)
            if (
                isfunction(func)
                and (getmodule(func) == multidex.plotter.callbacks)
            )
        ]
    }
