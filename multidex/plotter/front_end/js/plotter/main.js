import { fetch_json } from "js-utilz";
import h from "hscrpt";

import BrowseImageViewer from "plotter/browse-images";
import PlotPane from "plotter/plotpane";
import MainDataSet from "plotter/maindataset";
import ReflDataSet from "plotter/refldataset";

/** Application global state. */
class PlotterState {
    constructor() {
        this.panes = {
            main: null,
            reflectance: null,
            browse: null,
        };
        this.data = {
            main: null,
            reflectance: null,
        };
    }

    /** Glue: upon point selection in the main plot, trigger
     *  updates to the reflectance pane.
     */
    on_main_points_selected(ids) {
        if (ids.length > 0) {
            this.panes.reflectance.refresh({ ids });
        } else {
            this.panes.reflectance.clear();
        }
    }

    /** Initialization hook triggered by page load. */
    async on_DOMContentLoaded () {
        let colspecs = await fetch_json("/data/columns");

        this.data.reflectance = new ReflDataSet();
        this.data.main = new MainDataSet(
            colspecs,
            this.on_main_points_selected.bind(this)
        );

        this.panes.browse = new BrowseImageViewer(
            document.getElementById("browse-image-container"),
            colspecs,
        );
        this.panes.main = new PlotPane({
            id: "main-plot",
            dataset: this.data.main,
            container: document.getElementById("main-plot-container"),
            plot_title: "Observations",
            tools: "select-one zoom-box pan | zoom-plus zoom-minus autoscale",
            autoselect_tool: "select-one",
            x_label: this.data.main.columns.x,
            y_label: this.data.main.columns.y,
        });
        this.panes.reflectance = new PlotPane({
            id: "refl-plot",
            dataset: this.data.reflectance,
            container: document.getElementById("refl-plot-container"),
            plot_title: "Reflectance of selected ROIs",
            x_label: "Wavelength (nm)",
            y_label: "Reflectance",
            y_side: "right",
            no_data: () => h("p", ["Select observations to see spectra"]),
        });

        await this.panes.main.refresh({});
    }
}

// for troubleshooting, the global state object is accessible from the
// window
window.STATE = new PlotterState();

if (document.readyState === "loading") {
    let st = window.STATE;
    document.addEventListener("DOMContentLoaded", st.on_DOMContentLoaded.bind(st));
} else {
    // already loaded enough
    window.STATE.on_DOMContentLoaded();
}
