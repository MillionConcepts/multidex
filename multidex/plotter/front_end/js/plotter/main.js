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
    main_data_set: null,

    // The reflectance data set
    refl_data_set: null,
};

// debugging
window.STATE = STATE;

async function onDOMContentLoaded () {
    let colspecs = await fetch_json("/data/columns");

    STATE.refl_data_set = new ReflDataSet();
    STATE.main_data_set = new MainDataSet(
        colspecs,
        (ids) => {
            if (ids.length > 0) {
                STATE.panes.refl_plot.refresh({ ids });
            } else {
                STATE.panes.refl_plot.clear();
            }
        }
    );

    STATE.panes.browse_image_viewer = new BrowseImageViewer(
        document.getElementById("browse-image-container"),
        colspecs,
    );
    STATE.panes.main_plot = new PlotPane({
        id: "main-plot",
        dataset: STATE.main_data_set,
        container: document.getElementById("main-plot-container"),
        plot_title: "Observations",
        tools: "select-one zoom-box pan | zoom-plus zoom-minus autoscale",
        autoselect_tool: "select-one",
        x_label: STATE.main_data_set.columns.x,
        y_label: STATE.main_data_set.columns.y,
    });
    STATE.panes.refl_plot = new PlotPane({
        id: "refl-plot",
        dataset: STATE.refl_data_set,
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
