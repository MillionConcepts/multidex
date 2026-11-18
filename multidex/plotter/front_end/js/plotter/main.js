import { fetch_json } from "js-utilz";
import h from "hscrpt";

import BrowseImageViewer from "plotter/browse-images";
import PlotPane from "plotter/plotpane";
import MainDataSet from "plotter/maindataset";
import ReflDataSet from "plotter/refldataset";

/** Application global state. */
let STATE = {
    // Window panes
    panes: {},

    // The main data set, which acts as a DataSet for the main plot
    // pane but also manages a bunch of other stuff.
    data_set: null,
};

// debugging
window.STATE = STATE;

async function onDOMContentLoaded () {
    let colspecs = await fetch_json("/data/columns");

    STATE.data_set = new MainDataSet(colspecs);
    STATE.panes.browse_image_viewer = new BrowseImageViewer(
        document.getElementById("browse-image-container"),
        colspecs,
    );
    STATE.panes.main_plot = new PlotPane({
        id: "main-plot",
        dataset: STATE.data_set,
        container: document.getElementById("main-plot-container"),
        plot_title: "Observations",
        x_label: STATE.data_set.columns.x,
        y_label: STATE.data_set.columns.y,
    });
    STATE.panes.refl_plot = new PlotPane({
        id: "refl-plot",
        dataset: new ReflDataSet(),
        container: document.getElementById("refl-plot-container"),
        plot_title: "Reflectance of selected ROIs",
        x_label: "Wavelength (nm)",
        y_label: "Reflectance",
        y_side: "right",
        no_data: () => h("p", ["Select observations to see spectra"]),
    });

    await STATE.panes.main_plot.refresh({});
}

if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", onDOMContentLoaded);
} else {
    // already loaded enough
    onDOMContentLoaded();
}
