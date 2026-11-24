/** Abstract base class for plot data sets.
 *
 *  A plot data set is responsible for loading data from the back end,
 *  determining the data limits and the initial plot scales, and drawing
 *  the actual data points.
 */
import { scaleLinear } from "d3-scale";
import { DataLimits } from "data-limits";
import { abstract_method } from "js-utilz";

/** The plot pane is constructed with a DataSet *subclass* as an argument.
 *  It creates a data set object when it needs one.  Events fired at
 *  the plot pane may cause the DataSet object to be destroyed and
 *  re-created, or asked to load in a new set of data.
 *
 *  DataSet.data holds the actual data, when available; its structure
 *  is up to your subclass.
 *
 *  DataSet.limits_data, DataSet.limits_nice, and DataSet.scales are
 *  all objects with two properties, { x: ..., y: ... }; each describes
 *  something about the data's projection onto the X and Y axes of the
 *  plot.  Take care not to use these at the wrong time; the .limits_*
 *  properties are only valid when data is available, and .scales is only
 *  fully valid during calls to .place_data.
 *
 *  DataSet.limits_data describes the range of the actual data along
 *  both the X and Y axes.
 *
 *  DataSet.limits_nice is a "nicer" version of DataSet.limits_data;
 *  the limits are enlarged to allow the outermost tick labels of the
 *  default plot scale to be "nice round numbers".
 *
 *  DataSet.scales gives the transformation from data coordinates to
 *  plot canvas coordinates.
 */
export class DataSet {
    // Methods subclasses must implement

    /** Called whenever new data is to be loaded into the DataSet from
     *  the back end.  In this method you're responsible for
     *  formulating the query to the back end and calling fetch();
     *  PARAMS are a dictionary object holding whatever query
     *  parameters make sense for your kind of data (how this is
     *  defined is TBD).
     *
     *  Notice that this is *not* an async method.  You should just
     *  return the promise that fetch() returns.  Your `find_data_limits`
     *  method will be called when the back end delivers the data.
     */
    query_for_data(params) {
        abstract_method("DataSet.load_data");
    }

    /** Called after new data has been fetched and assigned as this.data.
     *  (We presume it to be transmitted from the back end as JSON.)
     *  Determine the range of the data along both axes and return an
     *  object like { x: [xmin, xmax], y: [ymin, ymax] }.
     */
    find_data_limits() {
        abstract_method("DataSet.find_data_limits");
    }

    /** Called every time the set of data points that are currently visible
     *  changes, either because the visible area of the plot has been changed,
     *  or because the data set itself has been changed (in a way that doesn't
     *  require the entire DataSet object to be re-created).  Return a freshly
     *  created list of SVG elements that represent the data.  DO NOT attempt
     *  to position those elements in canvas space; the mapping between data
     *  and canvas coordinates may not have been established yet.  Instead,
     *  annotate each element with its position in data coordinates, e.g.
     *  using data-xxx attributes.  Also tag these elements so you can easily
     *  find them in your place_data method (e.g. using HTML classes).  The
     *  place_data method is responsible for actually setting positions in
     *  canvas coordinates.
     */
    represent_data() {
        abstract_method("DataSet.represent_data");
    }

    /** Called every time the mapping from data coordinates to canvas
     *  coordinates changes, and immediately after the data set changes.
     *  PLOT is the DOM node for the entire plot (the <svg> element).
     *  Find all the DOM nodes below PLOT that you created in your
     *  represent_data method, and place them in canvas coordinates,
     *  using the current coordinate mapping (this.scales).  Do not
     *  destroy and recreate those elements.
     */
    place_data(plot) {
        abstract_method("DataSet.place_data");
    }

    /** Called when one or more points of your data set become
     *  selected.  (If your pane never selects anything, you don't
     *  need to implement this method.)  PLOT is the DOM node for
     *  the entire plot (the <svg> element), and POINTS is a list of
     *  SVG elements representing the selected points.  Take whatever
     *  action is appropriate to mark those points as selected and
     *  unmark all other points.
     */
    select_points(plot, points) {
        abstract_method("DataSet.select_points");
    }

    /** Construct a new DataSet object.
     *  You may override this method if you need to, but make sure
     *  to call super() first thing in your override.
     */
    constructor() {
        if (new.target === DataSet)
            throw new TypeError("cannot create an abstract DataSet");

        this.data = null;
        this.limits_data = { x: new DataLimits(), y: new DataLimits() };
        this.limits_nice = { x: new DataLimits(), y: new DataLimits() };
        this.scales      = { x: scaleLinear(),    y: scaleLinear() };
    }

    /** Reset this data set to its initial state of having no actual
     *  data loaded.
     *  You may override this method if you need to, but make sure
     *  to call super.clear_data() first thing in your override.
     */
    clear_data() {
        for (let axis of ["x", "y"]) {
            this.limits_data[axis].reset();
            this.limits_nice[axis].reset();
            this.scales[axis].domain([0, 0]).range([0, 0]);
        }
        this.data = null;
    }

    // Methods subclasses should not need to override

    /** Orchestrate loading new data into this data set.
     *  PARAMS are TBD and subclass-dependent.
     */
    async load_data(params) {
        this.clear_data();

        let resp = await this.query_for_data(params);
        if (!resp.ok)
            throw new Error(`${resp.url}: ${resp.status} ${resp.statusText}`);

        this.data = await resp.json();
        let limits = this.find_data_limits();
        this.limits_data.x.set(...limits.x);
        this.limits_data.y.set(...limits.y);
        // can't set limits_nice until we know the number of major ticks
    }

    /** Reset the domains of the X and Y scales to "nice round numbers"
     *  that encompass the entire data set, and update this.limits_nice.
     */
    autoscale(major_ticks) {
        for (let axis of ["x", "y"]) {
            this.limits_data[axis].apply_to_domain(this.scales[axis]);
            this.scales[axis].nice(major_ticks[axis]);
            this.limits_nice[axis].set_from_domain(this.scales[axis]);
        }
    }

    /** Change the domains of the X and Y scales to reflect new user-supplied
     *  limits LIMITS.
     */
    zoom_to(limits) {
        for (let axis of ["x", "y"]) {
            this.scales[axis].domain(
                this.limits_nice[axis].clamp_range(limits[axis])
            );
        }
    }

    /** Returns true if either the current X or Y scale is smaller than
     *  the nice limits for that axis.
     */
    can_zoom_out() {
        let dom = { x: this.scales.x.domain(),
                    y: this.scales.y.domain() };
        return (   dom.x[0] > this.limits_nice.x.lo
                || dom.x[1] < this.limits_nice.x.hi
                || dom.y[0] > this.limits_nice.y.lo
                || dom.y[1] < this.limits_nice.y.hi);
    }
}

export default DataSet;
