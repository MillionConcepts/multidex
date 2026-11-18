import { h } from "hscrpt";
import { h_circle } from "plotter/plotpane";
import DataSet from "plotter/dataset";
import { set_attrs } from "js-utilz";

const MIN = Math.min;
const MAX = Math.max;
const ISNAN = Number.isNaN;

export class MainDataSet extends DataSet {

    constructor(colspecs) {
        super();

        let columns = { x: null, y: null, m: null, };
        for (let [col, spec] of Object.entries(colspecs)) {
            if (!spec.meta)
                continue;
            // The columns "id", "name", "sol", "ltst", and "seq_id" are
            // special and should not be the default for any axis.
            if (/^(?:id|name|sol|ltst|seq_id)$/i.test(col))
                continue;

            // Use the first quantitative column as the x-axis default,
            // the second quantitative column as the y-axis default,
            // and the first qualitative column as the marker default.
            if (spec.type == "quant") {
                if (columns.x == null) {
                    columns.x = col;
                } else if (columns.y == null) {
                    columns.y = col;
                }
            } else if (spec.type == "qual") {
                if (columns.m == null) {
                    columns.m = col;
                }
            }
            if (columns.x != null && columns.y != null && columns.m != null)
                break;
        }

        this.columns = columns;
        this.filtered = { x: [], y: [], m: [] };
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

        let fx = [];
        let fy = [];
        let fm = [];

        let xmin = Infinity;
        let ymin = Infinity;
        let xmax = -Infinity;
        let ymax = -Infinity;
        for (let i = 0; i < MAX(x.length, y.length, m.length); i++) {
            if (x[i] == null || y[i] == null || m[i] == null
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
        }

        this.filtered.x = fx;
        this.filtered.y = fy;
        this.filtered.m = fm;
        return {
            x: [xmin, xmax],
            y: [ymin, ymax],
        };
    }

    represent_data() {
        let x = this.filtered.x;
        let y = this.filtered.y;
        let m = this.filtered.m;

        if (x.length != y.length || x.length != m.length)
            throw new Error(
                `length mismatch: ${this.columns.x}[${x.length}] !=` +
                    ` ${this.columns.y}[${y.length}]`
            );

        let points = [];
        for (let i = 0; i < x.length; i++) {
            points.push(h_circle("obs-point", {
                "data-x": x[i],
                "data-y": y[i],
                "data-m": m[i],
                "r": 2,
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
}

export default MainDataSet;
