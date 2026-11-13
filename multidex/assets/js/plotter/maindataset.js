import { h } from "hscrpt";
import { h_circle } from "plotter/plotpane";
import DataSet from "plotter/dataset";
import { set_attrs } from "js-utilz";

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

        if (x.length != y.length)
            throw new Error(
                `length mismatch: ${this.columns.x}[${x.length}] !=` +
                    ` ${this.columns.y}[${y.length}]`
            );

        return {
            x: [Math.min(...x), Math.max(...x)],
            y: [Math.min(...y), Math.max(...y)]
        };
    }

    represent_data() {
        let x = this.data[this.columns.x];
        let y = this.data[this.columns.y];

        if (x.length != y.length)
            throw new Error(
                `length mismatch: ${this.columns.x}[${x.length}] !=` +
                    ` ${this.columns.y}[${y.length}]`
            );

        // TODO: marker scale
        let points = [];
        for (let i = 0; i < x.length; i++) {
            points.push(h_circle("obs-point", {
                "data-x": x[i],
                "data-y": y[i],
                "r": 2,
            }));
        }
        return points;
    }

    place_data(plot) {
        let x_scale = this.scales.x;
        let y_scale = this.scales.y;
        for (let point of plot.getElementsByClassName("obs-point")) {
            set_attrs(point, {
                cx: x_scale(point.dataset.x),
                cy: y_scale(point.dataset.y),
            });
        }
    }
}

export default MainDataSet;
