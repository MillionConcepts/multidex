import { h } from "hscrpt";
import { h_g, h_polyline, h_circle, h_path } from "plotter/plotpane";
import DataSet from "plotter/dataset";
import { set_attrs } from "js-utilz";

// shorthand
const MIN = Math.min;
const MAX = Math.max;

// TODO make this come from the view (and then make it adjustable)
const error_bar_delta = 5;

export class ReflDataSet extends DataSet {
    query_for_data(params) {
        if (params == null || typeof params !== "object"
            || !Object.hasOwn(params, "ids")) {
            throw new Error("bad params to ReflDataSet.query_for_data");
        }
        let {ids} = params;
        if (!Array.isArray(ids))
            ids = [ids];

        let query = new URLSearchParams();
        query.append("id", ids.join(","));
        return fetch(`/data/spectrum?${query}`);
    }

    find_data_limits() {
        if (!Array.isArray(this.data)) {
            throw new Error("Plot data isn't an array");
        }

        // the _min variables start out at _positive_ infinity so that
        // the first min(val, _min) will always take val; same logic
        // in the opposite direction for the _max variables
        let xmin = Infinity, xmax = -Infinity;
        let ymin = Infinity, ymax = -Infinity;
        for (let [id, points] of this.data) {
            if (!Array.isArray(points)) {
                throw new Error(`Plot group for ${id} isn't an array`);
            }
            for (let { band, wave, mean, std } of points) {
                xmin = MIN(wave, xmin);
                xmax = MAX(wave, xmax);

                ymin = MIN(mean - std, ymin);
                ymax = MAX(mean + std, ymax);
            }
        }
        // the y-axis should always show zero
        ymin = MIN(ymin, 0);

        return { x: [xmin, xmax], y: [ymin, ymax] };
    }

    represent_data() {
        return this.data.map((group) => this.#h_refl_line(group));
    }

    place_data(plot) {
        for (let rl of plot.getElementsByClassName("refl-line")) {
            this.#place_refl_line(rl);
        }
    }

    #h_refl_line(group) {
        let [id, points] = group;
        // TODO colors
        return h_g(
            `refl-line rl-black`, { "data-id": id },
            h_polyline("refl-connline"),
            points.map((pt) => [
                h_circle("refl-dot", { "data-x": pt.wave,
                                       "data-y": pt.mean }),
                h_path("refl-bar", { "data-x": pt.wave,
                                     "data-ymin": pt.mean - pt.std,
                                     "data-ymax": pt.mean + pt.std }),
            ]));
    }

    #place_refl_line(rl) {
        let x_scale = this.scales.x;
        let y_scale = this.scales.y;

        let connector = rl.getElementsByClassName("refl-connline")[0];
        let circles = Array.from(rl.getElementsByClassName("refl-dot"));
        let bars = Array.from(rl.getElementsByClassName("refl-bar"));

        let points = [];
        for (let c of circles) {
            let cx = x_scale(c.dataset.x);
            let cy = y_scale(c.dataset.y);
            points.push(`${cx},${cy}`);
            set_attrs(c, { cx, cy });
        }
        connector.setAttribute("points", points.join(" "));

        for (let b of bars) {
            let x = x_scale(b.dataset.x);
            let xleft = x - error_bar_delta;
            let xdelta = error_bar_delta * 2;
            let ymax = y_scale(b.dataset.ymax);
            let ymin = y_scale(b.dataset.ymin);
            let ybeg = MIN(ymin, ymax);
            let ydelta = MAX(ymin, ymax) - ybeg;

            let d = (
                `M${xleft},${ymin}h${xdelta}`
              + `M${xleft},${ymax}h${xdelta}`
              + `M${x},${ybeg}v${ydelta}`
            );
            b.setAttribute("d", d);
        }
    }
}

export default ReflDataSet;
