import random
import shutil

import flask
import flask.cli
import pandas as pd
from dash import dash
from flask_caching.backends import FileSystemCache

from multidex import MULTIDEX_ROOT
from multidex.multidex_utils import qlist, model_metadata_df, make_tokens
from multidex.notetaking import Notepad, Paper
from multidex.plotter.application.helpers import (
    register_everything,
    configure_callbacks,
    register_clientside_callbacks,
    configure_flask_cache,
)
from multidex.plotter.application.structure import STATIC_IMAGE_URL
from multidex.plotter.config.settings import instrument_settings
from multidex.plotter.spectrum_ops import data_df_from_queryset

from multidex.plotter.layout import multidex_body
from multidex.plotter.graph import cache_set, cache_get
from multidex.plotter.models import INSTRUMENT_MODEL_MAPPING


def run_multidex(instrument_code, debug=False, use_notepad_cache=False):
    # initialize the app itself. HTML / react objects and callbacks from them
    # must be described in this object as dash components.
    app = dash.Dash(__name__)
    # random prefix for memory blocks / files shared within this instance
    cache_prefix = str(random.randint(1000000, 9999999))
    if use_notepad_cache is True:
        paper = Paper(f"multidex_{instrument_code.lower()}_{cache_prefix}")
        cache = Notepad(paper.prefix)
    else:
        cache = FileSystemCache.factory(
            app.server,
            config=configure_flask_cache(cache_prefix),
            args=[],
            kwargs={'default_timeout': 0}
        )
    spec_model = INSTRUMENT_MODEL_MAPPING[instrument_code.upper()]
    # active queryset is explicitly stored in global cache, as are
    # many other app runtime values
    # setter and getter functions for a flask_caching.Cache object. in the
    # context of this app initialization, they basically define a namespace.
    cset = cache_set(cache)
    cget = cache_get(cache)
    initialize_cache_values(cset, spec_model)
    # configure callback functions
    configured_callbacks = configure_callbacks(cget, cset, spec_model)
    # initialize app layout -- later changes are all performed by callbacks
    app.layout = multidex_body(spec_model)
    # register callbacks with app in reference to layout
    register_everything(app, configured_callbacks)
    # TODO: move these references into external scripts
    register_clientside_callbacks(app)
    # special case: serve context images using a flask 'route'
    static_folder = (
        MULTIDEX_ROOT
        / "plotter/application/assets/browse"
        / spec_model.instrument.lower()
    )

    @app.server.route(STATIC_IMAGE_URL + "<path:path>")
    def static_image_link(path):
        return flask.send_from_directory(static_folder, path)

    # silence irrelevant warnings about the dangers of using a dev server in
    # prod; this app only runs locally and woe betide thee if otherwise
    # noinspection PyUnresolvedReferences
    import multidex.plotter.application._suppress_werkzeug_warning

    flask.cli.show_server_banner = lambda *_: None
    port, looking_for_port = 49303, True
    while looking_for_port is True:
        try:
            app.run(
                debug=debug,
                use_reloader=False,
                dev_tools_silence_routes_logging=True,
                port=port,
                host="127.0.0.1"
            )
            looking_for_port = False
        except (OSError, SystemExit):
            # werkzeug calls sys.exit() if a port is in use, because werkzeug
            # is the most important thing in the world, and if a call to its
            # high-level API fails, it should immediately crash the program
            print("... " + str(port) + " is taken, checking next port ...")
            port += 1

    if use_notepad_cache is True:
        cache._update_index()
        for key in cache.index:
            cache.__delitem__(key)
        cache._index_buffer.unlink()
        cache._index_buffer.close()
    else:
        shutil.rmtree(".cache/" + cache_prefix)


def initialize_cache_values(cset, spec_model):
    cset("spec_model_name", spec_model.instrument_brief_name)
    cset("search_ids", qlist(spec_model.objects.all(), "id"))
    cset("highlight_ids", [])
    cset("label_ids", [])
    cset(
        "data_df",
        data_df_from_queryset(spec_model.objects.all()),
    )
    # TODO: this is a hack that should be initialized from some property of
    #  the model
    cset("r_star", True)
    metadata_df = model_metadata_df(spec_model)
    # TODO: this is a hack in place of adding formatted time parsing at
    #  various places within the application
    for c in metadata_df.columns[
        metadata_df.columns.str.match(r"(rc_)?l[mt]st")
    ]:
        if metadata_df[c].dtype.char != "O":
            # this is a defensive measure in case we decide to ingest times
            # already expressed as some sort of decimal, like ZCAM
            # CALTARGET_LTST (which is superfluous because it's just a decimal
            # form of RC_LTST, but some similarly-formatted field might not be)
            continue
        metadata_df.loc[pd.notna(metadata_df[c]), c] = [
            instant.hour * 3600 + instant.minute * 60 + instant.second
            for instant in metadata_df[c].dropna()
        ]
    # TODO: these are hacky and should go on models somewhere
    if "zoom" in metadata_df.columns:
        metadata_df["zoom"] = metadata_df["zoom"].astype(float)
    if {"rc_ltst", "rc_sol", "sol", "ltst"}.issubset(metadata_df.columns):
        for k, v in spec_model.cal_goodness(
            metadata_df[['ltst', 'rc_ltst', 'sol', 'rc_sol']]
        ).items():
            metadata_df[k] = v
    cset(
        "palette_memory",
        instrument_settings(spec_model.instrument)["palette_memory"]
    )
    cset("metadata_df", metadata_df)
    cset("tokens", make_tokens(metadata_df))
    cset("scale_to", "none")
    cset("average_filters", False)

