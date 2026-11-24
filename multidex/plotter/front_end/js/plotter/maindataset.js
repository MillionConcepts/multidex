import { h } from "hscrpt";
import { h_circle } from "plotter/plotpane";
import DataSet from "plotter/dataset";
import { set_attrs } from "js-utilz";

const MIN = Math.min;
const MAX = Math.max;
const ISNAN = Number.isNaN;

// FIXME: this is only good for MCAM and belongs in the back end anyway
const SKIP_COLUMNS = [
    "ltst", "seq_id", "name", "modification_time",
];

export class MainDataSet extends DataSet {
    //
    // core DataSet functionality
    //

    constructor(colspecs, on_select_hook, on_axis_change_hook) {
        super();

        let avail_columns = { qual: [], quant: [] };
        let columns = { x: null, y: null, m: null, id: null };
        for (let [col, spec] of Object.entries(colspecs)) {
            // we do this because there might not *be* an "id" column
            // (although there are probably going to be problems if there
            // isn't)
            if (col === "id") {
                columns.id = col;
                continue;
            }

            if (!spec.meta || SKIP_COLUMNS.includes(col)) {
                continue;
            }

            avail_columns[spec.type].push(col);
        }
        this.avail_columns = avail_columns;

        // Use the first available quantitative meta column as the default X
        // axis and the second as the default Y axis.  But don't pick "sol"
        // as the default X or Y axis (it's fine if it gets selected later).
        for (let col of avail_columns.quant) {
            if (col === "sol") {
                continue;
            }
            if (columns.x == null) {
                columns.x = col;
                continue;
            }
            if (columns.y == null) {
                columns.y = col;
                continue;
            }
            break;
        }

        // Use the first available qualitative meta column as the default
        // marker axis.
        columns.m = avail_columns.qual[0] ?? "";

        this.columns = columns;
        this.filtered = { x: [], y: [], m: [], id: [] };
        this.on_select_hook =
            on_select_hook ?? ((ignore) => undefined);
        this.on_axis_change_hook =
            on_axis_change_hook ?? ((...ignore) => undefined);
    }

    clear_data() {
        super.clear_data();
        this.filtered = { x: [], y: [], m: [], id: [] };
    }

    query_for_data(params) {
        if (params == null || typeof params !== "object") {
            throw new Error("bad params to MainDataSet.query_for_data");
        }
        if (Object.keys(params).length > 0) {
            throw new Error("not yet implemented: query for non-default data");
        }

        let query = new URLSearchParams();
        query.append("columns", Object.values(this.columns).join(","));
        return fetch(`/data?${query}`);
    }

    find_data_limits() {
        let x = this.data[this.columns.x];
        let y = this.data[this.columns.y];
        let m = this.data[this.columns.m];
        let id = this.data[this.columns.id];

        let f = { x: [], y: [], m: [], id: [] };

        let xmin = Infinity;
        let ymin = Infinity;
        let xmax = -Infinity;
        let ymax = -Infinity;
        for (let i = 0; i < MAX(x.length, y.length, m.length); i++) {
            if (x[i] == null || y[i] == null || m[i] == null || id[i] == null
                || ISNAN(x[i]) || ISNAN(y[i])) {
                continue;
            }
            xmin = MIN(x[i], xmin);
            xmax = MAX(x[i], xmax);
            ymin = MIN(y[i], ymin);
            ymax = MAX(y[i], ymax);

            f.x.push(x[i]);
            f.y.push(y[i]);
            f.m.push(m[i]);
            f.id.push(id[i]);
        }

        this.filtered = f;
        return {
            x: [xmin, xmax],
            y: [ymin, ymax],
        };
    }

    represent_data() {
        let { x, y, m, id } = this.filtered;
        if (x.length != y.length
            || x.length != m.length
            || x.length != id.length)
            throw new Error([
                "length mismatch",
                `${this.columns.x}[${x.length}]`,
                `${this.columns.y}[${y.length}]`,
                `${this.columns.m}[${m.length}]`,
                `${this.columns.id}[${id.length}]`,
            ].join(", "));

        let points = [];
        for (let i = 0; i < x.length; i++) {
            points.push(h_circle("obs-point", {
                "data-x": x[i],
                "data-y": y[i],
                "data-m": m[i],
                "data-id": id[i],
            }));
        }
        return points;
    }

    place_data(plot) {
        let x_scale = this.scales.x;
        let y_scale = this.scales.y;
        // TODO: marker scale
        for (let point of plot.getElementsByClassName("obs-point")) {
            set_attrs(point, {
                cx: x_scale(point.dataset.x),
                cy: y_scale(point.dataset.y),
            });
        }
    }

    select_points(plot, points) {
        let selected_ids = points.map((point) => point.dataset.id);

        for (let point of plot.getElementsByClassName("obs-point")) {
            if (selected_ids.includes(point.dataset.id)) {
                point.classList.add("selected");
            } else {
                point.classList.remove("selected");
            }
        }

        this.on_select_hook(selected_ids);
    }

    //
    // Axis controls
    //

    // Generate the label for axis AXIS.
    axis_label(axis) {
        // This returns a lambda because columns[axis] can change, so
        // we want PlotPane to check for a change on every repaint.
        return () => this.columns[axis];
    }

    // Generate a menu for choosing what column of the data set is
    // displayed on axis AXIS.
    axis_selector(axis) {
        let select_id = "axis-menu-main-"+axis;
        let select = h(
            "select",
            {
                "class": "axis-menu",
                "id": select_id,
                "onchange": (e) => this.#axis_selector_onchange(axis, e),
            },
            this.avail_columns[axis === "m" ? "qual" : "quant"]
                .map((col) => {
                    let attrs = { value: col };
                    if (col == this.columns[axis]) {
                        attrs["selected"] = "selected";
                    }
                    return h("option", attrs, [col]);
                })
        );
        let label = h(
            "label", { "for": select_id }, [ `${axis} axis: ` ]
        );
        return h("div", { "class": "labeled-menu", }, label, select);
    }

    // Event handler called when the column for AXIS is changed.
    #axis_selector_onchange(axis, event) {
        let target = event.target;
        let value = event.target.value;
        for (let option of target.querySelectorAll(":scope option")) {
            if (option.getAttribute("value") === value) {
                if (option.getAttribute("selected") == null) {
                    option.setAttribute("selected", "selected");
                }
            } else if (option.getAttribute("selected") != null) {
                option.removeAttribute("selected");
            }
        }
        this.columns[axis] = value;
        this.on_axis_change_hook(axis, value);
    }
}

export default MainDataSet;
