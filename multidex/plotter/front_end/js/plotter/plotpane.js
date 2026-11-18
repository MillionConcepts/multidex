import { deepfreeze, set_attrs } from "js-utilz";
import { h } from "hscrpt";
import { PlotToolbarElement } from "toolbar";
import { SIZE_TRACKER } from "plotter/size-tracker";

// slightly less typing and visual clutter
const ABS = Math.abs;
const MAX = Math.max;
const MIN = Math.min;

/** Conversion factor from CSS points to CSS pixels.  */
const PT = (96./72.);

/** Defaults for plot styling.  These can be overridden with the
 *  "options" object to the PlotPane constructor.
 */
const DEF = deepfreeze({
    // Title for the entire plot
    plot_title: "",

    // Label for the X-axis
    x_label: "",

    // Label for the Y-axis
    y_label: "",

    // In messages about loading data, this can be changed to a more
    // specific word than "data".
    loading_what: "data",

    // Rendering hook; this can be replaced by code that produces a
    // more specific "no data to make a plot out of" message.
    // Whatever you have this return will be put inside a
    // <div class="plot-box no-plot">.
    no_data: () => h("p", ["No data available"]),

    // Number of major ticks on each axis.  This number should always
    // be >= 3.  There will be a minor tick in between each pair of
    // major ticks.
    major_ticks: 5,

    // Scale factors applied by the "zoom in" and "zoom out" controls
    zoom_in_factor: 2.0/3.0,
    zoom_out_factor: 3.0/2.0,

    // All these lengths are in SVG user units, which is equivalent to CSS px
    major_tick_length: 8,
    minor_tick_length: 4,
    tick_to_label_gap: 4,
    tick_to_axis_label_gap: 6,
    data_from_axis_inset: 18,

    // These lengths are in px and must agree with the CSS.
    axis_line_thickness: 1.5,
    dot_radius: 3.5,
    error_bar_delta: 5,
});

/** A `PlotPane` renders a plot as a pane of the larger application.
 *
 * The PlotPane starts out rendering nothing.  To make it load some data,
 * fire a custom event "updateDataSet" at its container element; the `detail`
 * of that event should be a key-value dictionary that makes sense as the
 * `params` argument to the `query_for_data` method of the DataSet attached
 * to this PlotPane.
 */
export class PlotPane {
    #container;
    #dataset;
    #current_tool;
    #current_drag;

    /** Construct a new PlotPane.  OPTIONS must be a key-value dictionary
     *  containing at least these keys:
     *
     *  "id":       string to use as as the HTML id of the <svg> element at
     *              the root of the plot itself.  Several other elements will
     *              be given IDs derived from it.  This must be unique across
     *              the whole application.
     * "dataset":   Concrete `DataSet` instance that will hold the plot's data.
     * "container": DOM element that will contain the plot.  Its children will
     *              be replaced, and it will grow event handlers (see the
     *              class-level documentation for details).
     *
     * OPTIONS may also contain any item seen in the DEF table (above) to
     * override the default for that option.  You will almost certainly want
     * to set the "plot_title", "x_label", and "y_label" keys.
     */
    constructor(options) {
        options ??= {};

        this.base_id    = options.id ?? "<base id unspecified>";
        this.bounds_id  = `${this.base_id}-data-bounds`;
        this.#container = options.container;
        this.#dataset   = options.dataset;

        for (let option of Object.keys(DEF)) {
            this[option] = options[option] ?? DEF[option];
        }

        let plot_box = h("div", {
            "class": "plot-box no-plot",
            "onresize":        this.#on_resize.bind(this),
            "onpointerdown":   this.#on_pointerdown.bind(this),
            "onpointerup":     this.#on_pointerup.bind(this),
            "onpointercancel": this.#on_pointercancel.bind(this),
            // the pointer-move event handler is only present while dragging
        }, this.no_data());
        SIZE_TRACKER.track(plot_box);

        let plot_items = [];
        if (this.plot_title !== "") {
            plot_items.push(h("legend", [this.plot_title]));
        }
        plot_items.push(this.#h_plot_controls());
        plot_items.push(plot_box);

        this.#container.replaceChildren(h("fieldset", {
            "id": `${this.base_id}-controls`,
            "class": "plot-with-controls",
        }, plot_items));

        this.#container.addEventListener(
            "updateDataSet",
            this.#update_data_set.bind(this)
        );
    }

    //
    // Plot rendering.
    //

    /** Construct an "axis size" record like #size_axis would, for an axis that
     * isn't there at all (not even the container element).  See #size_axis
     * for the meanings of the fields.
     */
    #size_null_axis(edge) {
        return {
            edge,
            axis_depth:   0,
            axis_line_o:  0,
            tick_label_o: 0,
            axis_label_o: 0,
            axis_inset:   0,
            ...EDGE_PROPS[edge],
        };
    }

    /** Determine the space needed in the margins of the plot for axis AXIS.
     *
     * The bottom axis of a plot typically looks like this:
     *
     *  ┊ ┌──────────┬──────────┬──────────┬──────────┬──────────┐ ┊
     *  ┊                                                          ┊
     *  ┊25         50         75         100        125        150┊
     *  ┊                                                          ┊
     *  ┊                         Axis Label                       ┊
     *   ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄
     *
     * where solid lines are visible and dotted lines indicate the boundaries
     * of the whole plot.  Note how the axis line is inset a bit from the
     * plot boundaries so the outermost tick labels aren't cut off.
     *
     * The height of the whole display is the sum of:
     *    1. the thickness of the axis line itself, or the maximum length
     *       of a tick line, whichever is more
     *    3. the gap between the ticks and the tick labels
     *    4. the maximum height of any tick label
     *    5. the gap between the tick labels and the axis labels
     *    6. the height of the axis label
     *
     * In both (4) and (6) "height" means _bounding box_ height, not height
     * above text baseline.  If there are no tick labels, then (3) is reduced
     * to zero; if there is no axis label, then (5) is reduced to zero.
     *
     * The space left beyond the ends of the axis line is half the width of
     * the widest tick label, plus 1pt. (This is an approximation; the widest
     * tick label might not be at an end, and the tick labels that actually
     * are at the ends will almost never be the same width.  But greater
     * accuracy would require us to pay attention to which labels go where,
     * and might wind up looking unpleasantly asymmetric.)
     *
     * We calculate and return:
     *
     *  { edge,               // "top", "left", "bottom", or "right"
     *    broad, deep,        // "height" or "width": the dimensions parallel
     *                        // and perpendicular to the axis line
     *    axis_depth,         // space required in the dimension perpendicular
     *                        // to the axis line
     *    outward: +1/-1      // unit vector perpendicular to the axis line
     *                        // pointing toward the plot edge
     *    axis_line_o,        // depth offset of the axis line
     *    tick_label_o,       // depth offset of the anchor points of the
     *                        // tick labels
     *    axis_label_o,       // depth offset of the anchor point of the
     *                        // axis label
     *    axis_inset,         // distance from the edges of the axis box to the
     *                        // endpoints of the axis line
     *  }
     *
     * The "depth offset" of a point is its coordinate in the "deep" dimension,
     * relative to the side of the axis furniture's bounding box that has the
     * axis line on it.  (Thus, depth offsets always increase in the same
     * direction as "outward".)
     *
     * The anchor point of a label is always the center of its bounding box,
     * because that lets us treat all the labels uniformly.
     */
    #size_axis(axis) {
        let edge  = axis_edge(axis);
        let sz    = this.#size_null_axis(edge);
        let broad = sz.broad;
        let deep  = sz.deep;

        let n_ticks                = 0;
        let ticks_depth            = 0;
        let axis_label_depth       = 0;
        let tick_label_max_breadth = 0;
        let tick_label_max_depth   = 0;

        for (let child of axis.children) {
            let classes = child.classList;
            if (classes.contains("axis-line")) {
                ticks_depth = MAX(ticks_depth, this.axis_line_thickness);
            } else if (classes.contains("tick-major")) {
                ticks_depth = MAX(ticks_depth, this.major_tick_length);
            } else if (classes.contains("tick-minor")) {
                ticks_depth = MAX(ticks_depth, this.minor_tick_length);

            } else if (classes.contains("axis-label")) {
                axis_label_depth = child.getBoundingClientRect()[deep];

            } else if (classes.contains("tick-label")) {
                let box = child.getBoundingClientRect();
                tick_label_max_breadth = MAX(tick_label_max_breadth, box[broad]);
                tick_label_max_depth = MAX(tick_label_max_depth, box[deep]);
                n_ticks += 1;
            }
        }

        let axis_label_gap =
            (axis_label_depth > 0) ? this.tick_to_axis_label_gap : 0;
        let tick_label_gap =
            (tick_label_max_depth > 0) ? this.tick_to_label_gap : 0;

        sz.axis_depth = axis_label_depth + axis_label_gap
            + tick_label_max_depth + tick_label_gap
            + ticks_depth;
        sz.axis_line_o = this.axis_line_thickness / 2;
        sz.tick_label_o = ticks_depth + tick_label_gap + tick_label_max_depth/2;
        sz.axis_label_o = ticks_depth + tick_label_gap + tick_label_max_depth
            + axis_label_gap + axis_label_depth / 2;
        sz.axis_inset = tick_label_max_breadth / 2. + 1*PT;

        return sz;
    }

    /** Complete layout calculations for axis furniture; this part of
     *  the calculation cannot be done until we know the amount of space
     *  reserved for margins on all four sides of the plot area.
     */
    #place_axis_box(axis, sz, axis_margin, width, height) {
        let box = {
            horizontal: sz.broad === "width",
            x: 0, y: 0,
            width: 0, height: 0,
            axis_line_d: 0,
            axis_label_d: 0,
            tick_label_d: 0,
        };

        if (box.horizontal) {
            box.x = axis_margin.left;
            box.y = sz.edge === "top" ? 0 : height - sz.axis_depth;
            box.width = width - (axis_margin.left + axis_margin.right);
            box.height = sz.axis_depth;
        } else {
            box.x = sz.edge === "left" ? 0 : width - sz.axis_depth;
            box.y = axis_margin.top;
            box.width = sz.axis_depth;
            box.height = height - (axis_margin.top + axis_margin.bottom);
        }

        // `sz.{axis_line,tick_label,axis_label}_o` are relative to
        // "the side of the axis furniture's bounding box that has the
        // axis line on it"; convert them to coordinates relative to the
        // top left corner of `box`.
        let line_side, outward;
        if (sz.edge === "top") {
            line_side = box.height;
        } else if (sz.edge === "right") {
            line_side = box.x;
        } else if (sz.edge === "bottom") {
            line_side = box.y;
        } else /* sz.edge === "left" */ {
            line_side = box.width;
        }
        box.axis_line_d  = line_side + sz.outward * sz.axis_line_o;
        box.axis_label_d = line_side + sz.outward * sz.axis_label_o;
        box.tick_label_d = line_side + sz.outward * sz.tick_label_o;

        return box;
    }

    /** Set the final position of the background box for an axis.  */
    #place_axis_bg(axis, box) {
        let bg = axis.getElementsByClassName("axis-bg")[0];
        if (bg == null) return;
        set_attrs(bg, {
            x: box.x,
            y: box.y,
            width: box.width,
            height: box.height,
        });
    }

    /** Set the final position of the axis line for an axis (this is
     *  the line drawn right along the edge of the plot area).
     */
    #place_axis_line(axis, box) {
        let axis_line = axis.getElementsByClassName("axis-line")[0];
        if (axis_line == null) return;
        let scale = this.#dataset.scales[box.horizontal ? "x" : "y"];
        let [b1, b2] = scale.range();

        set_attrs(axis_line, {
            x1: box.horizontal ? b1 : box.axis_line_d,
            x2: box.horizontal ? b2 : box.axis_line_d,
            y1: box.horizontal ? box.axis_line_d : b1,
            y2: box.horizontal ? box.axis_line_d : b2,
        });
    }

    /** Set the final position of the overall label for an axis. */
    #place_axis_label(axis, box) {
        let axis_label = axis.getElementsByClassName("axis-label")[0];
        if (axis_label == null) return;
        set_attrs(axis_label, {
            x: box.horizontal ? box.x + box.width/2 : box.axis_label_d,
            y: box.horizontal ? box.axis_label_d : box.y + box.height/2,
        });
    }

    /** Set the final position of one tick mark along an axis.
     *
     *  D0 and D1 are the plot-space coordinates of the tick's
     *  endpoints on the "deep" dimension for this axis; SCALE is the
     *  transformation from data values to plot-space coordinates
     *  along the "broad" dimension for this axis.
     */
    #place_tick(tick, scale, horizontal, d0, d1) {
        let b = scale(+tick.dataset.value);
        set_attrs(tick, {
            x1: horizontal ? b  : d0,
            x2: horizontal ? b  : d1,
            y1: horizontal ? d0 : b,
            y2: horizontal ? d1 : b,
        });
    }

    /** Set the final position of one tick label along an axis.
     *
     *  D is the plot-space coordinate for all tick labels on the
     *  "deep" dimension for this axis; SCALE is the transformation
     *  from data values to plot-space coordinates along the "broad"
     *  dimension for this axis.
     */
    #place_tick_label(label, scale, horizontal, d) {
        let bounds = label.getBoundingClientRect();
        let b = scale(+label.dataset.value);
        set_attrs(label, {
            x: horizontal ? b : d,
            y: horizontal ? d : b,
        });
    }

    /** Set the final position of all ticks and tick labels on this axis.  */
    #place_ticks(axis, box, sz) {
        let scale = this.#dataset.scales[box.horizontal ? "x" : "y"];
        let d0     = box.axis_line_d;
        let maj_d1 = box.axis_line_d + sz.outward * this.major_tick_length;
        let min_d1 = box.axis_line_d + sz.outward * this.minor_tick_length;

        for (let tick of axis.getElementsByClassName("tick-major")) {
            this.#place_tick(tick, scale, box.horizontal, d0, maj_d1);
        }
        for (let tick of axis.getElementsByClassName("tick-minor")) {
            this.#place_tick(tick, scale, box.horizontal, d0, min_d1);
        }
        for (let label of axis.getElementsByClassName("tick-label")) {
            this.#place_tick_label(label, scale, box.horizontal,
                                   box.tick_label_d);
        }
    }

    /** Set the final position of all the plot furniture on this axis.  */
    #place_axis(axis, sz, axis_margin, data_margin, width, height) {
        // the depth might have been increased to make room for adjacent axes
        sz.axis_depth = axis_margin[sz.edge];
        let box = this.#place_axis_box(axis, sz, axis_margin, width, height);

        this.#place_axis_bg(axis, box);
        this.#place_axis_line(axis, box);
        this.#place_axis_label(axis, box);
        this.#place_ticks(axis, box, sz);
    }

    /** Subroutine of place_grid: determine whether a set of grid lines
     *  are associated with the X or the Y axis.
     */
    #grid_is_x(grid) {
        if (grid.classList.contains("grid-x"))
            return true;
        else if (grid.classList.contains("grid-y"))
            return false;
        else
            throw new Error(
                "grid direction unknown"
              + ` (id: ${grid.id} classes: ${grid.className})`
            );
    }

    /** Set the final position of all the grid lines associated with a
     *  particular axis (if a plot has axes on both sides of the box
     *  either horizontally or vertically, it may have a set of grid
     *  lines for each).  WIDTH and HEIGHT give the total space
     *  allocated to the plot and its furniture, not the space
     *  allocated to the data area.
     */
    #place_grid(grid, axis_margin, width, height) {
        let x_axis = this.#grid_is_x(grid);
        let scale = this.#dataset.scales[x_axis ? "x" : "y"];
        const line_coords = x_axis
              ? ((val) => ({
                  x1: val,
                  x2: val,
                  y1: axis_margin.top,
                  y2: height - axis_margin.bottom }))
              : ((val) => ({
                  x1: axis_margin.left,
                  x2: width - axis_margin.right,
                  y1: val,
                  y2: val }));

        for (let line of grid.children) {
            if (!(line instanceof SVGLineElement))
                continue;
            let val = line.dataset.value;
            if (val == null)
                continue;
            let coords = line_coords(scale(+val));
            set_attrs(line, coords);
        }
    }

    /** Set the final position of the clipping box for the data area
     *  of the plot.  WIDTH and HEIGHT give the total space allocated
     *  to the plot and its furniture, not the space allocated to the
     *  data area.
     */
    #place_data_clip(clip_path, data_margin, width, height) {
        let clip_rect = clip_path.firstElementChild;
        if (clip_path.childElementCount !== 1
            || clip_rect.tagName !== "rect") {
            console.error("plot clip not as expected, clip not set: ",
                          clip_path);
            return;
        }
        set_attrs(clip_rect, {
            x:      data_margin.left,
            y:      data_margin.top,
            width:  width - (data_margin.right + data_margin.left),
            height: height - (data_margin.bottom + data_margin.top),
        });
    }

    /** Event handler: called whenever the plot is resized. */
    #on_resize(event) {
        let plot_box = event.currentTarget;
        if (plot_box.classList.contains("no-plot")
            || plot_box.classList.contains("load-error")
            || plot_box.classList.contains("loading"))
            return;

        let width = plot_box.clientWidth;
        let height = plot_box.clientHeight;
        let plot = plot_box.firstElementChild;
        if (plot === null || plot.tagName !== "svg"
            || plot.id !== this.base_id) {
            console.error("plot box not as expected", plot_box);
            return;
        }

        // Determine the space required for plot furniture.
        //
        // On each side where there's an axis, we need to reserve space
        // perpendicular to the data area for that axis.  We also need to
        // reserve space at the *ends* of that axis (so, parallel to the data
        // area on that side) to ensure that the outermost tick labels, which
        // overhang the ends of the axis line, won't overflow the plot
        // container.  Finally, to ensure that data does not collide with the
        // axis lines, and also to prevent vertical and horizontal tick labels
        // from colliding at any corner where two axes meet, we insert an
        // additional gap between the axis line and the data area, on each side
        // where there's an axis.
        let axes        = {};
        let axis_size   = {};
        let axis_inset  = { width: 0, height: 0 };
        let axis_margin = new Margin();
        let data_margin = new Margin();

        for (let axis of plot.getElementsByClassName("axis")) {
            let sz = this.#size_axis(axis);
            let edge = sz.edge;

            axes[edge] = axis;
            axis_size[edge] = sz;
            axis_margin[edge] = sz.axis_depth;
            data_margin[edge] = (sz.axis_depth > 0)
                ? sz.axis_depth + this.data_from_axis_inset : 0;

            axis_inset[sz.broad] = MAX(axis_inset[sz.broad], sz.axis_inset);
        }
        for (let edge of EDGES) {
            let inset = axis_inset[EDGE_PROPS[edge].broad];
            axis_margin[edge] = MAX(axis_margin[edge], inset);
            data_margin[edge] = MAX(data_margin[edge], inset);
        }

        // Adjust the declared size of the plot itself; this restores the
        // 1px == 1 SVG unit equivalence.
        set_attrs(plot, { width, height, viewBox: `0 0 ${width} ${height}` });

        // The X and Y scales run between the data edges.  The Y scale is
        // inverted so that larger domain values are plotted above smaller.
        this.#dataset.scales.x.range(
            [data_margin.left, width - data_margin.right]
        );
        this.#dataset.scales.y.range(
            [height - data_margin.bottom, data_margin.top]
        );

        // The placement of all these elements depends on the scale ranges
        // and/or the margins calculated above.
        for (let edge of EDGES) {
            if (Object.hasOwn(axes, edge))
                this.#place_axis(axes[edge], axis_size[edge],
                                 axis_margin, data_margin, width, height);
        }
        for (let grid of plot.getElementsByClassName("grid")) {
            this.#place_grid(grid, axis_margin, width, height);
        }
        this.#place_data_clip(plot.getElementById(this.bounds_id),
                              data_margin, width, height);
        this.#dataset.place_data(plot);
    }

    /** Have the DataSet load a new view of the data and redraw the plot.
     *  This may be called directly or by firing an updateDataSet event
     *  at the plot container (see #update_data_set below).
     */
    async refresh(view_params) {
        let controls = this.#container.querySelector(":scope .controls-box");
        let plot_box = this.#container.querySelector(":scope .plot-box");

        controls.disabled = true;
        this.#on_deselect_tools();

        plot_box.classList.add("loading");
        plot_box.classList.remove("load-error");
        plot_box.classList.remove("no-plot");
        plot_box.replaceChildren(
            h("img", { src: "/s/loading.svg" }),
            h("p", `Loading ${this.loading_what}…`)
        );

        try {
            await this.#dataset.load_data(view_params);
        } catch (err) {
            plot_box.classList.remove("loading");
            plot_box.classList.add("load-error");
            plot_box.replaceChildren(h("p", [
                `Error loading ${this.loading_what}:`,
                h("br"),
                `${err}`.replace(/^Error: /, "")
            ]));
            return;
        }

        if (!this.#dataset.data) {
            plot_box.classList.remove("loading");
            plot_box.classList.add("no-plot");
            plot_box.replaceChildren(this.no_data());
            return;
        }

        plot_box.classList.remove("loading");
        controls.disabled = false;
        this.autoscale();
    }

    /** Clear the plot pane.
     *  This may be called directly or by firing an updateDataSet event
     *  at the plot container (see #update_data_set below).
     */
    clear() {
        let controls = this.#container.querySelector(":scope .controls-box");
        controls.disabled = true;
        this.#on_deselect_tools();

        this.#dataset.clear_data();

        let plot_box = this.#container.querySelector(":scope .plot-box");
        plot_box.classList.remove("loading");
        plot_box.classList.remove("load-error");
        plot_box.classList.add("no-plot");
        plot_box.replaceChildren(this.no_data());
    }

    /** Event handler for "updateDataSet".
     *  `event.detail == null` means clear the plot.
     *  `event.detail == {}` means load and plot the default data set,
     *  assuming "the default data set" is a thing that makes sense for
     *  this pane.  To load a non-default data set, put something in the
     *  object (exactly what is defined by the plot's DataSet subclass).
     */
    #update_data_set(event) {
        if (event.detail == null) {
            this.clear();
        } else if (typeof event === "object") {
            // I *think* it's fine to just drop the promise this returns;
            // it should get added to the task queue and executed anyway.
            this.refresh(event.detail);
        } else {
            throw new Error(`bad detail for updateDataSet: ${event.detail}`);
        }
    }

    /** Construct the HTML structure for the plot's toolbar. */
    #h_plot_controls() {
        // TODO options to control which controls you get
        return h("plot-toolbar", {
            "class": "controls-box",
            "disabled": "disabled",
            "buttons": "pan zoom-box | zoom-plus zoom-minus autoscale",
            "ontoolTriggered": (e) => {
                let action = e.detail.triggered;
                switch (action) {
                case "zoom-plus":  this.#on_trigger_zoom_plus();  return;
                case "zoom-minus": this.#on_trigger_zoom_minus(); return;
                case "autoscale":  this.autoscale();  return;
                default:
                    console.error(
                        `trigger event for unregistered action ${action}`
                    );
                }
            },
            "ontoolSelected": (e) => {
                let tool = e.detail.selected;
                switch(tool) {
                case null:
                case undefined:
                    this.#on_deselect_tools();
                    return;
                case "pan":
                    this.#on_select_pan();
                    return;
                case "zoom-box":
                    this.#on_select_zoom_box();
                    return;
                default:
                    console.error(
                        `select event for unregistered tool`, tool
                    );
                }
            },
        });
    }

    /** Construct the SVG structure of an actual plot.  Should only be
     *  called when data is available (i.e. this.#dataset.data != null).
     *
     *  TODO: Options to control which edges get axes.
     */
    #h_plot_svg() {
        let x_tick_vals = this.#dataset.scales.x.ticks(this.major_x_ticks);
        let y_tick_vals = this.#dataset.scales.y.ticks(this.major_y_ticks);

        return h("svg", {
            width: 0,
            height: 0,
            viewBox: "0 0 0 0",
            id: this.base_id,
            preserveAspectRatio: "none",
        }, [
            h("defs", [h("clipPath", { "id": this.bounds_id }, [h_rect()])]),
            h("g", { "class": "main-layer" }, [
                h_axis("bottom",
                       { tick_vals: x_tick_vals, tick_labels: true,
                         label: this.x_label }),
                h_axis("right",
                       { tick_vals: y_tick_vals, tick_labels: true,
                         label: this.y_label }),
                h_grid("x", x_tick_vals),
                h_grid("y", y_tick_vals),
                h("g", { "attr-clip-path": `url(#${this.bounds_id})` },
                  this.#dataset.represent_data()),
            ]),
            h_rect("drag-box"),
        ]);
    }

    //
    // Interactivity
    //

    /** Redraw the plot after a user action which altered the axis
     *  limits or the data set.  Regrettably, this involves recreating
     *  the entire SVG structure from scratch.
     */
    #after_rescale() {
        let controls_box = this.#container.querySelector(":scope .controls-box");
        if (this.#dataset.can_zoom_out()) {
            controls_box.dispatchEvent(new CustomEvent(
                "enableButton", { detail: ["pan", "zoom-minus", "autoscale"] }
            ));
        } else {
            controls_box.dispatchEvent(new CustomEvent(
                "disableButton", { detail: ["pan", "zoom-minus", "autoscale"] }
            ));
        }

        let plot_box = this.#container.querySelector(":scope .plot-box");
        plot_box.replaceChildren(this.#h_plot_svg());

        // No matter what I try, the first resize event's handler
        // observes axis and tick labels to have bounding boxes with
        // height and width zero.  But if we go ahead and place the
        // plot furniture incorrectly (according to those garbage text
        // bounding boxes), then fire a second resize event from an
        // animation frame, that second event sees accurate box sizes
        // and will place everything correctly, and it all happens
        // fast enough to not have a visible flash of nonsense.
        plot_box.dispatchEvent(new Event("resize"));
        window.requestAnimationFrame(() => {
            plot_box.dispatchEvent(new Event("resize"));
        });
    }

    /** Change the axis limits to exactly LIMITS.  LIMITS is expected
     *  to be a structure of the form
     *
     *      { x: [x_min, x_max], y: [y_min, y_max] }
     *
     *  with all values in data coordinates.
     */
    #zoom_to(limits) {
        this.#dataset.zoom_to(limits);
        this.#after_rescale();
    }

    /** Event handler: The "autoscale" button was clicked, change the
     *  axis limits so all available data is visible and fits neatly
     *  in the plot area.  This function may also be called directly.
     */
    autoscale() {
        this.#dataset.autoscale({ x: this.major_ticks, y: this.major_ticks });
        this.#after_rescale();
    }

    /** Event handler: The "zoom in" button was clicked, change the
     *  axis limits to enlarge the central part of the plot.
     */
    #on_trigger_zoom_plus() {
        let cur_limits = {
            x: this.#dataset.scales.x.domain(),
            y: this.#dataset.scales.y.domain(),
        };
        let step_factor = (1 - this.zoom_in_factor) / 2;
        let step_width = step_factor * (cur_limits.x[1] - cur_limits.x[0]);
        let step_height = step_factor * (cur_limits.y[1] - cur_limits.y[0]);

        let new_limits = {
            x: [ cur_limits.x[0] + step_width,
                 cur_limits.x[1] - step_width ],
            y: [ cur_limits.y[0] + step_height,
                 cur_limits.y[1] - step_height ],
        };
        this.#zoom_to(new_limits);
    }

    /** Event handler: The "zoom out" button was clicked, change the
     *  axis limits to shrink what's currently visible and show more
     *  of the data.
     */
    #on_trigger_zoom_minus() {
        let cur_limits = {
            x: this.#dataset.scales.x.domain(),
            y: this.#dataset.scales.y.domain(),
        };
        let step_factor = (this.zoom_out_factor - 1) / 2;
        let step_width = step_factor * (cur_limits.x[1] - cur_limits.x[0]);
        let step_height = step_factor * (cur_limits.y[1] - cur_limits.y[0]);

        let new_limits = {
            x: [ cur_limits.x[0] - step_width,
                 cur_limits.x[1] + step_width ],
            y: [ cur_limits.y[0] - step_height,
                 cur_limits.y[1] + step_height ],
        };
        this.#zoom_to(new_limits);
    }

    /** Event handler: The "pan tool" button was clicked, make the
     *  plot area grabbable and draggable.
     */
    #on_select_pan() {
        let plot = this.#container.querySelector(":scope .plot-box");

        plot.classList.remove("zoombox-active");
        plot.classList.add("pan-active");
        this.#current_tool = "pan";
    }

    /** Event handler: The "zoom to rectangle" button was clicked,
     *  prepare to draw a selection rectangle on top of the plot.
     */
    #on_select_zoom_box() {
        let plot = this.#container.querySelector(":scope .plot-box");

        plot.classList.add("zoombox-active");
        plot.classList.remove("pan-active");
        this.#current_tool = "zoombox";
    }

    /** Event handler: The currently active tool was clicked again
     *  (or something else happened that causes the currently active
     *  tool to be deselected), clear any transient state associated
     *  with a tool being active.
     */
    #on_deselect_tools() {
        let plot = this.#container.querySelector(":scope .plot-box");
        if (plot) {
            plot.classList.remove("zoombox-active");
            plot.classList.remove("pan-active");
        }
        this.#current_tool = undefined;
        this.#end_drag();
    }

    /** Event handler subroutine: Start a drag action within the plot area.
     *  Returns an object with these properties:
     *
     *  {  plot:    DOM object for the entire plot SVG
     *     plot_l:  Viewport X-coordinate of the left edge of the plot
     *     plot_t:  Viewport Y-coordinate of the top edge of the plot
     *     down_x:  Viewport X-coordinate of the mouse position where the drag began
     *     down_y:  Viewport Y-coordinate of the mouse position where the drag began
     *     pointer_id: ID code for the pointing device that was used
     *  }
     *
     * It may have other properties as well; these should be treated as
     * private and not to be altered.
     *
     * Event handlers for specific types of drag action may add their
     * own private data to this object when a drag begins; that data
     * will be available to all the event handlers called over the
     * course of that drag.
     */
    #begin_drag(event) {
        let plot = event.currentTarget;
        let plot_area = plot.getBoundingClientRect();

        let drag = {
            plot,
            plot_l: plot_area.left,
            plot_t: plot_area.top,
            down_x: event.clientX,
            down_y: event.clientY,
            pointer_id: event.pointerId,
            onmove: this.#on_pointermove.bind(this),
        };

        plot.addEventListener("pointermove", drag.onmove, {
            capture: true, passive: true,
        });
        plot.setPointerCapture(drag.pointerId);

        this.#current_drag = drag;
        return drag;
    }

    /** Event handler subroutine: End a drag action within the plot area.
     *  Returns an object with these properties:
     *
     *  {  plot:    DOM object for the entire plot SVG
     *     plot_l:  Viewport X-coordinate of the left edge of the plot
     *     plot_t:  Viewport Y-coordinate of the top edge of the plot
     *     down_x:  Viewport X-coordinate of the mouse position where the drag began
     *     down_y:  Viewport Y-coordinate of the mouse position where the drag began
     *     up_x:    Viewport X-coordinate of the mouse position where the drag ended
     *     up_y:    Viewport Y-coordinate of the mouse position where the drag ended
     *     pointer_id: ID code for the pointing device that was used
     *  }
     *
     * It may have other properties as well; these should be treated as
     * private and not to be altered.
     */
    #end_drag(event) {
        if (this.#current_drag == null)
            return undefined;
        let drag = this.#current_drag;
        this.#current_drag = undefined;

        let plot = drag.plot;
        if (plot != null) {
            plot.removeEventListener("pointermove", drag.onmove, {
                capture: true, passive: true,
            });
            plot.releasePointerCapture(drag.pointerId);
        }

        if (event !== undefined) {
            drag.up_x = event.clientX;
            drag.up_y = event.clientY;
        }
        delete drag.onmove;
        return drag;
    }

    // for a zoom, we drag out a rectangle from the current point
    // and the plot area only changes on mouseup.

    /** Event handler subroutine: Begin a drag action with the
     *  zoom tool active.
     */
    #begin_zoom(drag) {
        let box = drag.plot.querySelector(":scope .drag-box");
        box.classList.add("placed");
        box.setAttribute("x", drag.down_x - drag.plot_l);
        box.setAttribute("y", drag.down_y - drag.plot_t);
        box.setAttribute("width", 0);
        box.setAttribute("height", 0);
        drag.box = box;
    }

    /** Event handler subroutine: Update the screen for mouse motion
     *  during a drag with the zoom tool active.
     */
    #move_zoom(event, drag) {
        // SVG "width" and "height" properties cannot be negative
        set_attrs(drag.box, {
            x:      MIN(drag.down_x, event.clientX) - drag.plot_l,
            y:      MIN(drag.down_y, event.clientY) - drag.plot_t,
            width:  ABS(drag.down_x - event.clientX),
            height: ABS(drag.down_y - event.clientY),
        });
    }

    /** Event handler subroutine: Complete a zoom-to-rectangle
     *  drag action.
     */
    #end_zoom(drag) {
        this.#cancel_zoom(drag);

        let scales = this.#dataset.scales;
        let x0 = MIN(drag.up_x, drag.down_x) - drag.plot_l;
        let x1 = MAX(drag.up_x, drag.down_x) - drag.plot_l;
        let y0 = MIN(drag.up_y, drag.down_y) - drag.plot_t;
        let y1 = MAX(drag.up_y, drag.down_y) - drag.plot_t;
        this.#zoom_to({
            x: [ scales.x.invert(x0), scales.x.invert(x1) ],
            y: [ scales.y.invert(y0), scales.y.invert(y1) ],
        });
    }

    /** Event handler subroutine: Cancel a zoom-to-rectangle
     *  drag action.
     */
    #cancel_zoom(drag) {
        let box = drag.box;
        if (box != null) {
            box.classList.remove("placed");
            box.setAttribute("x", 0);
            box.setAttribute("y", 0);
            box.setAttribute("width", 0);
            box.setAttribute("height", 0);
        }
    }

    // For a pan, the plot area moves continuously during the drag and we
    // don't show a box, but we do alter the cursor as an extra visual cue.
    // Because the plot area moves during the drag, we need to save the
    // original scales in case the drag is cancelled.

    /** Event handler subroutine: Begin a drag action with the
     *  pan tool active.
     */
    #begin_pan(drag) {
        drag.plot.classList.add("panning");
        drag.original_limits = {
            x: this.#dataset.scales.x.domain(),
            y: this.#dataset.scales.y.domain(),
        };
    }

    /** Event handler subroutine: Update the screen for mouse motion
     *  during a drag with the pan tool active.
     */
    #move_pan(event, drag) {
        let scales = this.#dataset.scales;
        let dx = scales.x.invert(event.movementX)
            - scales.x.invert(0);
        let dy = scales.y.invert(event.movementY)
            - scales.y.invert(0);

        this.#zoom_to({
            x: scales.x.domain().map((x) => x - dx),
            y: scales.y.domain().map((y) => y - dy),
        });
    }

    /** Event handler subroutine: Complete a panning drag action. */
    #end_pan(drag) {
        drag.plot.classList.remove("panning");
        // no further action is required
    }

    /** Event handler subroutine: Cancel a panning drag action. */
    #cancel_pan(drag) {
        drag.plot.classList.remove("panning");
        this.#zoom_to(drag.original_limits);
    }

    /** Event handler: Primary mouse button pressed and held within the
     *  plot area.
     */
    #on_pointerdown(event) {
        let plot_box = event.currentTarget;
        if (plot_box.classList.contains("no-plot")
            || plot_box.classList.contains("load-error")
            || plot_box.classList.contains("loading"))
            return;

        // only start a drag if we have an active tool
        // only start a drag if it's specifically the primary button of
        // the primary pointer that was depressed
        if (this.#current_tool == null
            || !event.isPrimary
            || event.buttons !== 1)
            return;

        if (this.#current_tool === "zoombox")
            this.#begin_zoom(this.#begin_drag(event));
        else if (this.#current_tool === "pan")
            this.#begin_pan(this.#begin_drag(event));
        else if (!this.#current_tool)
            /* do nothing */;
        else {
            console.error("pointerup: unknown tool", this.#current_tool);
        }
    }

    /** Event handler: Mouse moved with the primary button held down,
     *  having initially been pressed and held with the pointer inside
     *  the plot area.  (This event handler is only registered while
     *  the button is held down.)
     */
    #on_pointermove(event) {
        let plot_box = event.currentTarget;
        if (plot_box.classList.contains("no-plot")
            || plot_box.classList.contains("load-error")
            || plot_box.classList.contains("loading")
            || this.#current_tool == null
            || this.#current_drag == null) {
            console.error("pointermove: should not have been called");
            return;
        }

        if (this.#current_tool === "zoombox") {
            this.#move_zoom(event, this.#current_drag);
        } else if (this.#current_tool === "pan") {
            this.#move_pan(event, this.#current_drag);
        } else {
            console.error("pointermove: unknown tool", this.#current_tool);
        }
    }

    /** Event handler: Primary mouse button released, completing a drag
     *  action begun within the plot area.
     */
    #on_pointerup(event) {
        let drag = this.#end_drag(event);
        if (this.#current_tool === "zoombox")
            this.#end_zoom(drag);
        else if (this.#current_tool === "pan")
            this.#end_pan(drag);
        else if (!this.#current_tool)
            /* do nothing */;
        else {
            console.error("pointerup: unknown tool", this.#current_tool);
        }
    }

    /** Event handler: User has done something that cancels a drag action
     *  that was in progress.
     */
    #on_pointercancel(event) {
        let drag = this.#end_drag(event);
        if (this.#current_tool === "zoombox")
            this.#cancel_zoom(drag);
        else if (this.#current_tool === "pan")
            this.#cancel_pan(drag);
        else if (!this.#current_tool)
            /* do nothing */;
        else {
            console.error("pointercancel: unknown tool", this.#current_tool);
        }
    }
}

//
// only implementation details below
//

/** Dimensional properties of plot "edges", which have the same names
 *  and the same spatial relationships with the plot data area that
 *  CSS margins have with a CSS box.  (Note: the plot data area is not
 *  a CSS box.)  The names are "top", "right", "bottom", and "left",
 *  and the properties are:
 *
 *  "deep"     the box dimension ("height" or "width") perpendicular to the
 *             plot area on this edge's side
 *  "broad"    the box dimension parallel to the plot area on this edge's side
 *  "outward"  1-dimensional unit vector pointing away from the plot area
 *             (in the 2D plane, this is always along the "deep" dimension,
 *             which is why we don't need it to be a 2D unit vector).
 */
const EDGE_PROPS = deepfreeze({
    top:    { deep: "height", broad: "width",  outward: -1, },
    right:  { deep: "width",  broad: "height", outward: +1, },
    bottom: { deep: "height", broad: "width",  outward: +1, },
    left:   { deep: "width",  broad: "height", outward: -1, },
});
const EDGES = Object.freeze(Object.keys(EDGE_PROPS));

/** Data object holding a length (in SVG user units = CSS pixels)
 *  for each edge of a plot.
 */
class Margin {
    constructor() {
        for (let edge of EDGES) {
            this[edge] = 0;
        }
    }
}

/** ELT is a SVG group containing axis furniture.  Determine from
 *  its class properties which edge of the plot it belongs to.
 */
function axis_edge(elt) {
    for (let c of elt.classList.values()) {
        if (!c.startsWith("axis-"))
            continue;

        let edge = c.slice(5);
        if (Object.hasOwn(EDGE_PROPS, edge))
            return edge;

        throw new Error(
            `axis has bogus edge class ${edge}`
          + ` (id: ${elt.id} classes: ${elt.className})`
        );
    }
    throw new Error(
        "axis has no edge class"
      + ` (id: ${elt.id} classes: ${elt.className})`
    );
}

/** Given an array of major tick positions along an axis, place a
 *  minor tick halfway between each pair of major ticks and return a
 *  new array of all the ticks' positions, sorted into ascending
 *  numerical order.
 */
function add_minor_ticks(ticks) {
    let minor_ticks = [];
    for (let i = 0; i < ticks.length - 1; i++) {
        minor_ticks.push((ticks[i] + ticks[i+1]) / 2);
    }
    ticks = [...ticks, ...minor_ticks];
    // the compareFn might look pointless, but without it the array is
    // sorted in ascending *string collation* order!
    ticks.sort((a, b) => a - b);
    return ticks;
}

// SVG structure molds
// these are all fairly tightly coupled to the PlotPane class and its
// two-phase algorithm for rendering a plot, but are not part of it
// because they have no need for the `this` object, and making them
// free functions reduces visual clutter at point of use
// the exported functions are also useful to PlotDataSet subclasses

/** Fabricate an SVG `<text>` element with text content TEXT and
 *  attributes ...ATTRS.  By default it will be positioned at (0,0).
 */
export function h_text(classes, text, ...attrs) {
    return h("text", classes, { x: 0, y: 0 }, ...attrs, [text]);
}

/** Fabricate an SVG `<circle>` element with attributes ...ATTRS.
 *  By default it will be positioned at (0,0) and have radius zero.
 */
export function h_circle(classes, ...attrs) {
    // r is set either by CSS or by attrs
    return h("circle", classes, { x: 0, y: 0, r: 0 }, ...attrs);
}

/** Fabricate an SVG `<rect>` element with attributes ...ATTRS.
 *  By default it will be positioned at (0,0) and have width and height zero.
 */
export function h_rect(classes, ...attrs) {
    return h("rect", classes, { x: 0, y: 0, width: 0, height: 0 }, ...attrs);
}

/** Fabricate an SVG `<line>` element with attributes ...ATTRS.
 *  By default both endpoints will be (0,0).
 */
export function h_line(classes, ...attrs) {
    return h("line", classes, { x1: 0, y1: 0, x2: 0, y2: 0 }, ...attrs);
}

/** Fabricate an SVG `<polyline>` element with attributes ...ATTRS.
 *  By default its point sequence will be empty.
 */
export function h_polyline(classes, ...attrs) {
    return h("polyline", classes, { points: "" }, ...attrs);
}

/** Fabricate an SVG `<path>` element with attributes ...ATTRS.
 *  By default its "d" attribute (path definition) will be empty.
 */
export function h_path(classes, ...attrs) {
    return h("path", classes, { d: "" }, ...attrs);
}

/** Fabricate an SVG `<g>` element with children and properties ...ARGS.
 *  Arguments follow the same rules as the arguments to h().
 */
export function h_g(...args) {
    return h("g", ...args);
}

/** Fabricate all the furniture for a plot axis.
 *  The DOM structure produced has the form
 *
 *    <g class="axis axis-{side}">
 *      <rect class="axis-bg"/>
 *      <line class="axis-line"/>
 *      <text class="axis-label">{label}</text>
 *      [<line class="tick-[major/minor]"/><text class="tick-label">n.nn</text>
 *       ...]
 *    </g>
 */
function h_axis(side, { label, tick_vals, tick_labels }) {
    let want_ticks = false;
    let want_tick_labels = false;
    let tick_labeler = null;
    if (tick_vals == null || tick_vals === false) {
        tick_vals = [];
    } else if (!Array.isArray(tick_vals)) {
        tick_vals = Array.from(tick_vals);
    }
    if (tick_vals.length > 0) {
        want_ticks = true;
        tick_vals = add_minor_ticks(tick_vals);
        if (tick_labels == null || tick_labels === false) {
            // use above defaults
        } else if (tick_labels === true) {
            want_tick_labels = true;
            tick_labeler = (tick, ix) => tick;
        } else if (Array.isArray(tick_labels)) {
            want_tick_labels = true;
            tick_labeler = (tick, ix) => tick_labels[ix];
        } else if (typeof tick_labels === "function") {
            want_tick_labels = true;
            tick_labeler = tick_labels;
        } else {
            throw new Error(
                `bad "tick_labels" argument: ${tick_labels}`
            );
        }
    }

    let furniture = [h_rect("axis-bg"), h_line("axis-line")];
    if (label != null && label !== "") {
        furniture.push(h_text("axis-label", `${label}`));
    }
    if (want_ticks) {
        let ix = 0;
        for (let tick of tick_vals) {
            // ix starts from 0, so minor ticks have odd indices
            let major = ix % 2 === 0;
            furniture.push(h_line(
                major ? "tick-major" : "tick-minor",
                { "data-value": tick }
            ));
            if (want_tick_labels && major) {
                furniture.push(h_text(
                    "tick-label", tick_labeler(tick, ix/2),
                    { "data-value": tick }
                ));
            }
            ix += 1;
        }
    }
    return h_g("axis axis-" + side, furniture);
}

/** Fabricate plot grid lines going in one direction (i.e. vertical or
 *  horizontal, not both).  The DOM structure produced has the form
 *
 *    <g class="grid grid-{dir}">
 *      [<line class="gridline-[major/minor]"/>
 *       ...]
 *    </g>
 */
function h_grid(dir, tick_vals) {
    if (tick_vals == null || tick_vals === false) {
        tick_vals = [];
    } else if (!Array.isArray(tick_vals)) {
        tick_vals = Array.from(tick_vals);
    }
    if (tick_vals.length === 0) {
        return [];
    }
    tick_vals = add_minor_ticks(tick_vals);

    return h_g(
        "grid grid-"+dir,
        tick_vals.map((val, ix) => h_line(
            "gridline-" + (ix % 2 === 0 ? "major" : "minor"),
            { "data-value": val })));
}

export default PlotPane;
