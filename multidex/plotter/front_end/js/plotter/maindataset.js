import { h } from "hscrpt";
import { h_circle } from "plotter/plotpane";
import DataSet from "plotter/dataset";
import { set_attrs } from "js-utilz";

const MIN = Math.min;
const MAX = Math.max;
const ISNAN = Number.isNaN;

export class MainDataSet extends DataSet {

    constructor(colspecs, on_select_hook) {
        super();

        let columns = { x: null, y: null, m: null, id: null };
        for (let [col, spec] of Object.entries(colspecs)) {
            // Use the first quantitative column as the x-axis default,
            // the second quantitative column as the y-axis default,
            // and the first qualitative column as the marker default.
            // The columns "id", "name", "sol", "ltst", and "seq_id" are
            // special.
            if (col === "id") {
                columns.id = col;
            } else if (col === "seq_id") {
                // only take seq_id if we don't have id
                if (columns.id == null)
                    columns.id = col;
            } else if (/^(?:name|sol|ltst)$/i.test(col) || !spec.meta) {
                // skip this one
            } else if (spec.type == "quant") {
                if (columns.x == null) {
                    columns.x = col;
                } else if (columns.y == null) {
                    columns.y = col;
                }
            } else if (spec.type == "qual") {
                if (columns.m == null) {
                    columns.m = col;
                }
            } else {
                console.error(
                    `${col}: unrecognized type '${spec.type}'` +
                    ` (meta: ${spec.meta})`
                );
            }

            // stop if we have everything we need
            if (columns.x != null
                && columns.y != null
                && columns.m != null
                && columns.id != null)
                break;
        }

        this.columns = columns;
        this.filtered = { x: [], y: [], m: [], id: [] };
        this.on_select_hook = on_select_hook ?? ((ignore) => {});
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

        let fx = [];
        let fy = [];
        let fm = [];
        let fid = [];

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

            fx.push(x[i]);
            fy.push(y[i]);
            fm.push(m[i]);
            fid.push(id[i]);
        }

        this.filtered.x = fx;
        this.filtered.y = fy;
        this.filtered.m = fm;
        this.filtered.id = fid;
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
}

export default MainDataSet;
