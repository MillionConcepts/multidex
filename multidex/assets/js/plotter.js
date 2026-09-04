"use strict";

// Utilities

/// If ITEM is neither null nor undefined, append it to ARRAY.
/// Otherwise do nothing.
function push_nonnull(array, item) {
    if (item != null)
        array.push(item);
}

/// Construct an SVG element whose contents will be populated from the
/// Mithril "oncreate" hook, presumably (but not necessarily) by D3.
/// Embeds certain expectations about how we style and lay out SVG
/// elements (see main.css).
function m_svg(class_, oncreate, onupdate) {
    const m = window.m;

    // Ensure preserveAspectRatio is set properly from birth to
    // minimize the chance of a flash of bad layout.  Also set
    // viewBox to a harmless value at birth; the real value cannot
    // be set until clientHeight/clientWidth are available.
    return m("svg", {
        "class": class_,
        preserveAspectRatio: "none",
        viewBox: "0 0 1 1",
        oncreate,
        onupdate,
    });
}

// Set up an SVG element's attributes and overall transform matrix
// the way we need them to be.  The 'svg' argument should be a bare
// DOM node which is an SVG element.
//
// Returns a D3 selection for either the original SVG element, or for
// a <g transform="..."> node which has been inserted as a child of
// the original SVG node; in either case, it's the selection into
// which graphical elements should be inserted.
//
// Because it might need to insert that transform node, the
// original node must be empty when this function is called.
// (This is why m_svg does not take a children argument.)
function adjust_svg(dom, width, height, t_width, t_height) {
    const d3 = window.d3;
    let svg = d3.select(dom);

    if (dom.tagName !== "svg") {
        console.error(`adjust_svg called on a <${dom.tagName}>:`, dom);
        return svg;
    }

    // Reiterate the setting of preserveAspectRatio here just to be sure.
    svg.attr("preserveAspectRatio", "none")
        .attr("viewBox", `0 0 ${width} ${height}`);
    if (t_width == 0 && t_height == 0)
        return svg;
    return svg.append("g")
        .attr("transform", `translate(${t_width},${t_height})`);
}

// Return a D3 selection for DOM node 'dom', assumed to be an SVG
// element, _or_, if 'dom' has exactly one child which is an
// anonymous 'g' element with a transform attribute, return a
// D3 selection for that child.  (This mirrors the return value of
// adjust_svg.  You call that one from an oncreate hook, and this
// one from the matching onupdate hook.)
function select_svg(dom) {
    const d3 = window.d3;
    let svg = d3.select(dom);

    if (dom.tagName !== "svg") {
        console.error(`select_svg called on a <${dom.tagName}>:`, dom);
        return svg;
    }

    let g = svg.selectChildren();
    return (g.size() === 1
            && g.node().tagName === "g"
            && g.attr("transform") != null)
        ? g
        : svg;
}

// Tweak the result of applying a D3 axis to a SVG element or
// a group within an SVG element.  Intended to be used as e.g.
// `svg.call(d3.axisLeft(...)).call(adjust_axis("left"))`.
// This does only the tweaking that has to be done *every time*
// an axis is applied to a particular element, not the tweaking that
// has to be done *only the first time*.  (Do the first-time tweaks
// directly in the oncreate hook, *after* calling this function.)
function adjust_axis(cls) {
    return function adjust_axis_curried(axis) {
        axis.select(".domain").remove();
        axis.classed(`axis axis-${cls}`, true)
            .attr("font-size", null)
            .attr("font-family", null);
    };
}


/// Application global state
let STATE = {
    // Whether browse images can be displayed.
    browse_images_available: null,

    // Browse image currently being displayed.
    browse_image: null,

    // Data row currently selected in the main view.
    selected_main_row: null,
    selected_main_obs_id: null,

    // Specifications for all the columns the back end can give us.
    columns_available: {},

    // Names of the columns currently being used as x axis, y axis,
    // and markers in the main plot.
    x_col: null,
    y_col: null,
    m_col: null,

    // might become switchable in the future
    image_col: "images_left",

    // Data backing the main plot.
    main_plot_data: {},

    // Options for the reflectance plot.
    refl_options: {
        avg: true,
        bayer: true,
        scale: true,
    },

    // Data backing the reflectance plot.
    refl_plot_data: null,
    refl_plot_error: null,

    loading: function() {
        return (!STATE.x_col || !STATE.y_col || !STATE.m_col
                || !STATE.main_plot_data);
    },

    refresh_main_plot: function() {
        const m = window.m;

        let prev_selected_id = STATE.selected_main_obs_id;
        STATE.select_observation(null);

        let columns = [];
        push_nonnull(columns, STATE.x_col);
        push_nonnull(columns, STATE.y_col);
        push_nonnull(columns, STATE.m_col);
        if (STATE.browse_images_available) {
            columns.push("images_left");
            columns.push("images_right");
        }

        m.request({
            url: "/data",
            params: { columns: columns.join(",") }
        }).then((columns) => {
            STATE.main_plot_data = columns;
            if (prev_selected_id != null) {
                let new_index = STATE.main_plot_data["id"].findIndex(
                    (id) => id === prev_selected_id
                );
                if (new_index !== -1)
                    STATE.select_observation(new_index);
                // if it's -1, we already cleared the active observation
            }
        });
    },

    refresh_refl_plot: function() {
        const m = window.m;
        if (STATE.selected_main_obs_id == null) {
            STATE.refl_plot_data = null;
            STATE.refl_plot_error = null;
            return;
        }

        m.request({
            url: "/data/spectrum",
            params: {
                id: STATE.selected_main_obs_id,
                avg: STATE.refl_options.avg,
                bayer: STATE.refl_options.bayer,
                scale: STATE.refl_options.scale
            },
        }).then((spectrum) => {
            // D3 wants this as an array of records, not an object.
            // FIXME Maybe produce the right thing on the back end?
            let s_array = Object.entries(spectrum).map(
                ([band, { wave, mean, std }]) => ({ band, wave, mean, std })
            );
            s_array.sort((a, b) => a.wave - b.wave);

            STATE.refl_plot_error = null;
            STATE.refl_plot_data = s_array;
        }).catch((err) => {
            STATE.refl_plot_error = err;
            STATE.refl_plot_data = null;
        });
    },

    refresh_browse_image: function() {
        if (!STATE.browse_images_available
            || STATE.selected_main_row == null) {
            STATE.browse_image = null;
        } else {
            STATE.browse_image =
                STATE.main_plot_data[STATE.image_col][STATE.selected_main_row];
        }
    },

    select_observation: function(row_index) {
        if (row_index == null) {
            STATE.selected_main_row = null;
            STATE.selected_main_obs_id = null;
        } else {
            let obs_id = STATE.main_plot_data["id"][row_index];
            STATE.selected_main_row = row_index;
            STATE.selected_main_obs_id = obs_id;
        }
        STATE.refresh_refl_plot();
        STATE.refresh_browse_image();
    },
};

//
// The main plot
//

/// View rendering the main plot and its controls.
/// No arguments; the data comes from STATE (see above).
///
/// As a temporary measure, this renders a *table* of the main plot
/// data; clicking on any row causes the browse image for that row to
/// be displayed.  (This might become an alternative presentation tab
/// in the future.)
function MainPlot() {
    const m = window.m;

    function onclick(event) {
        let target = event.target;
        // the initial target of the event is probably a <td>
        while (target
               && (target.nodeType !== Node.ELEMENT_NODE
                   || target.tagName !== "TR")) {
            target = target.parentElement;
        }
        if (!target) {
            console.error("no <tr> found as parent of %o", event.target);
            return;
        }
        // clicking again on the same observation clears the selection
        if (target.sectionRowIndex == STATE.selected_main_row) {
            STATE.select_observation(null);
        } else {
            STATE.select_observation(target.sectionRowIndex);
        }
    }

    function still_loading() {
        // continue to show loading spinner until data is available
        // AFAICT there's no way to not duplicate the structure, see
        // https://github.com/MithrilJS/mithril.js/discussions/3079
        return m("div.pinwheel-holder", [
            m("img", { src: "/s/loading.svg" }),
            m("p", ["Loading…"])
        ]);
    }

    function view() {
        if (STATE.loading()) {
            return still_loading();
        }

        let ids   = STATE.main_plot_data["id"]        ?? [];
        let x_col = STATE.main_plot_data[STATE.x_col] ?? [];
        let y_col = STATE.main_plot_data[STATE.y_col] ?? [];
        let m_col = STATE.main_plot_data[STATE.m_col] ?? [];
        let images = (
            STATE.browse_images_available
                ? STATE.main_plot_data[STATE.image_col]
                : null
        ) ?? [];

        if (ids.length == 0
            || x_col.length == 0
            || y_col.length == 0
            || m_col.length == 0) {
            return still_loading();
        }

        let base_key = `${STATE.x_col},${STATE.y_col},${STATE.m_col}`;
        if (images.length > 0) {
            base_key = `${base_key},${STATE.image_col}`;
        }

        let nrows = Math.max(
            x_col.length,
            y_col.length,
            m_col.length,
            images.length,
        );

        let rows = Array.from(
            // an object with a "length" property is "array-like"
            // enough to get the mapFn called for all i in
            // 0 .. length.  Doncha love javascript?
            { length: nrows },
            (_, i) => {
                let cells = [
                    // `${expr}` is the current recommended way to
                    // coerce expr to a string.
                    m("td", [`${ids[i] ?? ""}`]),
                    m("td", [`${x_col[i] ?? ""}`]),
                    m("td", [`${y_col[i] ?? ""}`]),
                    m("td", [`${m_col[i] ?? ""}`]),
                ];
                if (images.length > 0) {
                    cells.push(m("td", [`${images[i] ?? ""}`]));
                }
                let attrs = {
                    key: `${base_key};${i}`,
                };
                if (i === STATE.selected_main_row) {
                    attrs["class"] = "row-selected";
                }
                return m("tr", attrs, cells);
            }
        );

        let colscope = {scope: "col"};
        let colheads = [
            m("th", colscope, ["obs. id"]),
            m("th", colscope, [STATE.x_col]),
            m("th", colscope, [STATE.y_col]),
            m("th", colscope, [STATE.m_col]),
        ];
        if (images.length > 0) {
            colheads.push(m("th", colscope, ["left image"]));
        }

        return m("table#main-data", [
            m("thead", [m("tr", colheads)]),
            m("tbody", { onclick }, rows)
        ]);
    }
    return { view };
}

//
// Reflectance plot
//

/// Active view for the reflectance pane.  No arguments; all data comes
/// from STATE (see above).
function ReflPlot() {
    const m = window.m;
    const d3 = window.d3;

    // We render the plot with fixed dimensions and then use SVG
    // viewboxing to make the browser scale it to its container.
    // This is not ideal, but it's the path of least resistance
    // with D3.  I don't know yet if it will be necessary to make
    // these parameters adjustable.
    const plot_width = 1000;
    const plot_height = 1000;
    const plot_padding = 20;
    const x_grain = 50;
    const y_grain = 0.05;

    // Internal state describing the axis scales; shared among all plot
    // components, written only by 'update_scales'.
    let x_scale = null;
    let y_scale = null;
    let wavelengths = null;
    let wave_to_band = null;

    function update_scales() {
        wavelengths = STATE.refl_plot_data.map((rec) => rec.wave);
        wave_to_band = Object.fromEntries(
            STATE.refl_plot_data.map((rec) => [rec.wave, rec.band])
        );

        let xmin = Math.floor(Math.min(...wavelengths) / x_grain) * x_grain;
        let xmax = Math.ceil(Math.max(...wavelengths) / x_grain) * x_grain;

        // Geologists working with spectra want longer wavelengths on the right.
        x_scale = d3.scaleLinear()
            .domain([xmin, xmax])
            .range([0 + plot_padding, plot_width - plot_padding]);

        let ymin = Math.min(...STATE.refl_plot_data.map(
            (rec) => rec.mean - rec.std
        ));
        let ymax = Math.max(...STATE.refl_plot_data.map(
            (rec) => rec.mean + rec.std
        ));

        if (ymin < 0) {
            ymin = Math.floor(ymin / y_grain) * y_grain;
        } else {
            ymin = 0;
        }
        ymax = Math.ceil(ymax / y_grain) * y_grain;

        // The Y-axis range is reversed because the SVG coordinate
        // system maps larger numbers to positions lower on the screen,
        // but we want larger numbers in the data to be mapped to
        // positions *higher* on the scren.
        y_scale = d3.scaleLinear()
            .domain([ymin, ymax])
            .range([plot_height - plot_padding, 0 + plot_padding]);
    }

    // The _structure_ of the plot DOM is built by Mithril rather than
    // D3, because Mithril's "hyperscript" notation is more congenial
    // than D3 selector goo.  Then, D3 renders the data- and scale-
    // dependent parts of the plot DOM into each of the sub-areas,
    // from Mithril's "lifecycle hooks".

    // I'm not happy with the positioning logic for the axis titles,
    // but as far as I can tell there's no way to position an SVG
    // <text> element relative to a point on its _bounding box_,
    // and without that, I don't see how to do better.

    function wave_axis() {
        function wave_axis_refresh(svg) {
            let axis = d3.axisBottom(x_scale)
                .tickValues(wavelengths)
                .tickFormat((wave) => `${wave}`);

            return svg
                .call(axis)
                .call(adjust_axis("bot"));
        }
        function wave_axis_oncreate(vnode) {
            let height = vnode.dom.clientHeight;
            adjust_svg(vnode.dom, 1000, height, 0, 0)
                .call(wave_axis_refresh)
                .append("text")
                .classed("axis-title", true)
                .attr("x", 500)
                .attr("y", height - 4)
                .attr("text-anchor", "middle")
                .attr("fill", "currentColor")
                .text("Wavelength (nm)");
        }
        function wave_axis_onupdate(vnode) {
            select_svg(vnode.dom).call(wave_axis_refresh);
        }
        return m_svg("p-ax-bot", wave_axis_oncreate, wave_axis_onupdate);
    }

    function band_axis() {
        function band_axis_refresh(svg) {
            let axis = d3.axisTop(x_scale)
                .tickValues(wavelengths)
                .tickFormat((wave) => wave_to_band[wave]);

            return svg
                .call(axis)
                .call(adjust_axis("top"));
        }
        function band_axis_oncreate(vnode) {
            let height = vnode.dom.clientHeight;
            adjust_svg(vnode.dom, 1000, height, 0, height)
                .call(band_axis_refresh);
            // The band axis doesn't have an axis title.
        }
        function band_axis_onupdate(vnode) {
            select_svg(vnode.dom).call(band_axis_refresh);
        }
        return m_svg("p-ax-top", band_axis_oncreate, band_axis_onupdate);
    }

    function refl_axis() {
        function refl_axis_refresh(svg) {
            return svg
                .call(d3.axisLeft(y_scale))
                .call(adjust_axis("lft"));
        }
        function refl_axis_oncreate(vnode) {
            let width = vnode.dom.clientWidth;
            adjust_svg(vnode.dom, width, 1000, width, 0)
                .call(refl_axis_refresh)
                .append("text")
                .classed("axis-title", true)
                .attr("x", -500)
                .attr("y", -42)
                .attr("text-anchor", "middle")
                .attr("transform", "rotate(-90)")
                .attr("fill", "currentColor")
                .text("Reflectance");
        }
        function refl_axis_onupdate(vnode) {
            select_svg(vnode.dom).call(refl_axis_refresh);
        }

        return m_svg("p-ax-lft", refl_axis_oncreate, refl_axis_onupdate);
    }

    function grid() {
        function grid_refresh(svg) {
            svg.select(".grid-x")
                .selectAll("line")
                .data(wavelengths)
                .join("line")
                .attr("x1", (d) => 0.5 + x_scale(d))
                .attr("x2", (d) => 0.5 + x_scale(d))
                .attr("y1", 0)
                .attr("y2", plot_height);
            svg.select(".grid-y")
                .selectChildren()
                .data(y_scale.ticks())
                .join("line")
                .attr("y1", (d) => 0.5 + y_scale(d))
                .attr("y2", (d) => 0.5 + y_scale(d))
                .attr("x1", 0)
                .attr("x2", plot_width);
        }
        function grid_oncreate(vnode) {
            let svg = adjust_svg(vnode.dom, 1000, 1000, 0, 0);
            svg.append("g").classed("grid-x", true);
            svg.append("g").classed("grid-y", true);
            svg.call(grid_refresh);
        }
        function grid_onupdate(vnode) {
            select_svg(vnode.dom).call(grid_refresh);
        }
        return m_svg("p-grid", grid_oncreate, grid_onupdate);
    }

    function data() {
        function point_refresh(datum) {
            let prev_datum = this.__oldData__;
            if (prev_datum != null
                && datum.band === prev_datum.band
                && datum.wave === prev_datum.wave
                && datum.mean === prev_datum.mean
                && datum.std  === prev_datum.std) {
                // this point does not need to be updated at all
                return;
            }

            // something about the datum represented by this point has
            // changed, or we're creating a new point from scratch.
            let x = x_scale(datum.wave);
            let xmin = x_scale(datum.wave - 5);
            let xmax = x_scale(datum.wave + 5);
            let xdelta = xmax - xmin;

            let y = y_scale(datum.mean);
            // y-axis is reversed
            let ymin = y_scale(datum.mean + datum.std);
            let ymax = y_scale(datum.mean - datum.std);
            let ydelta = ymax - ymin;

            // Setting the entire class attribute ensures that the
            // new-or-altered point is not marked selected.
            // 'this.className = "point"' won't work because this is
            // an SVGElement; className is read-only on SVGElements.
            this.setAttribute("class", "point");

            // have Mithril optimize the DOM update within the point
            m.render(this, [
                // the rect defines the point's hitbox, so you don't
                // have to click exactly on the lines; see overlay_onclick
                m("rect", {
                    "class": "hitbox",
                    x: xmin, y: ymin, width: xdelta, height: ydelta,
                }),
                m("line", {
                    "class": "h",
                    x1: xmin, x2: xmax, y1: y, y2: y
                }),
                m("line", {
                    "class": "v",
                    x1: x, x2: x, y1: ymin, y2: ymax
                }),
            ]);
        }
        function data_refresh(svg) {
            svg.selectAll("g.point")
                .property("__oldData__", (d) => d)
                .data(STATE.refl_plot_data, (d) => `${d.wave}`)
                .join("g")
                .each(point_refresh);
        }
        function data_oncreate(vnode) {
            adjust_svg(vnode.dom, 1000, 1000, 0, 0)
                .call(data_refresh);
        }

        function data_onupdate(vnode) {
            select_svg(vnode.dom).call(data_refresh);
        }

        return m_svg("p-data", data_oncreate, data_onupdate);
    }

    function overlay() {
        function overlay_onclick(event) {
            let overlay = event.currentTarget;
            let popover = overlay.querySelector(":scope > .data-pop");
            let data_box = d3.select("#refl-chart > .p-data");
            if (popover == null) {
                console.error("popover missing from overlay", overlay);
            }

            let hits = document.elementsFromPoint(event.clientX, event.clientY)
                .filter((el) => (el.classList.contains("hitbox")
                                 || el.classList.contains("data-pop")));

            if (hits.length === 0) {
                // a click on the plot area, not within any point or
                // the popover, clears the selection
                data_box.selectAll(".point.selected")
                    .classed("selected", false);
                popover.classList.remove("visible");
                return;
            }

            let hit = hits[0];
            if (hit.classList.contains("data-pop")) {
                // click within the popover does not affect the selection
                return;
            }

            // The D3 datum is attached to the hitbox's parent, and the
            // parent is also the node that needs to be marked selected
            hit = hit.parentNode;
            if (hit.classList.contains("selected")) {
                // clicking the last selected point again deselects it
                hit.classList.remove("selected");
                popover.classList.remove("visible");
                return;
            }

            // selecting a new point; clear any previous selection
            data_box.selectAll(".point.selected").classed("selected", false);
            hit.classList.add("selected");

            let datum = d3.select(hit).datum();
            if (datum == null) {
                // no data to show in the popover
                popover.classList.remove("visible");
                return;
            }
            let label = `${datum.band}: ${datum.wave} nm<br>`
                + `Reflectance: ${datum.mean.toFixed(4)}`
                + ` ± ${datum.std.toFixed(4)} (1 σ)`;

            // The overlay box has the *actual* dimensions of the plot
            // area, after viewbox scaling.  We need to manually apply
            // that scaling to the abstract plot coordinates that come
            // out of x_scale and y_scale.
            let cws = event.currentTarget.clientWidth / 1000;
            let chs = event.currentTarget.clientHeight / 1000;

            let px = x_scale(datum.wave);
            let py = y_scale(datum.mean);

            // Initially try to position the popover below and to the
            // right of the selected element.
            let left = (px + 10) * cws;
            let top = (py + 10) * chs;

            let pop = d3.select(popover);
            pop.html(label)
                .style("top", `${top}px`)
                .style("left", `${left}px`)
                .style("bottom", null)
                .style("right", null)
                .classed("visible", true);

            // If that caused overflow, move the popover to the
            // opposite side of the selected element in each affected
            // dimension. Note: the overlay box is known to have
            // overflow:hidden.
            if (overlay.clientWidth < overlay.scrollWidth) {
                let right = (1000 - (px - 10)) * cws;
                pop.style("left", null).style("right", `${right}px`);
            }
            if (overlay.clientHeight < overlay.scrollHeight) {
                let bottom = (1000 - (py - 10)) * chs;
                pop.style("top", null).style("bottom", `${bottom}px`);
            }
        }

        function overlay_onupdate(vnode) {
            let overlay = vnode.dom;
            let popover = overlay.querySelector(":scope > .data-pop");
            let data_box = d3.select("#refl-chart > .p-data");
            if (popover == null) {
                console.error("popover missing from overlay", overlay);
            }

            // if the point that used to be selected has just been removed
            // from the data set, hide the popover
            if (data_box.selectAll(".point.selected").empty()) {
                popover.classList.remove("visible");
            }
        }

        return m("div",
                 { "class": "p-overlay",
                   "onclick": overlay_onclick,
                   "onupdate": overlay_onupdate },
                 [ m("div", { "class": "data-pop" }) ]);
    }

    function plot_skeleton() {
        return m("div#refl-chart", [
            wave_axis(),
            band_axis(),
            refl_axis(),
            grid(),
            data(),
            overlay(),
        ]);
    }

    // Controls
    function bool_control(id, label, property) {
        let iattrs = {
            type: "checkbox",
            switch: "switch",
            id,
            name: id,
            onchange: (e) => {
                STATE.refl_options[property] = e.target.checked;
                STATE.refresh_refl_plot();
            }
        };
        if (STATE.refl_options[property]) {
            iattrs["checked"] = "checked";
        }
        return m("div", [
            m("input", iattrs),
            m("label", { "for": id }, [ label ])
        ]);
    }

    function controls() {
        return m("fieldset.controls", [
            m("legend", ["Plot options:"]),
            bool_control("refl-avg", "Averaged bands", "avg"),
            bool_control("refl-bayer", "Bayer bands", "bayer"),
        ]);
    }

    function plot_container(data, err) {
        if (err == null && data == null) {
            return m("p#refl-no-selection", [
                STATE.loading()
                    ? "" : "Select an observation to see its spectrum."
            ]);
        }

        if (err != null && data != null) {
            return m("p#refl-error", [
                "impossible: data and err both non-null"
            ]);
        }
        if (err != null && data == null) {
            // FIXME The actual back-end error message gets eaten
            // somewhere in the guts of m.request.  It's really hard
            // to trigger this case from inside the GUI, so not urgent.
            return m("p#refl-error", [`${err.code ?? err}`]);
        }

        // data != null, err == null
        // we need to be sure the scales are set before any of the oncreate
        // hooks fire
        update_scales();
        return plot_skeleton();
    }

    function view() {
        return [
            controls(),
            plot_container(STATE.refl_plot_data, STATE.refl_plot_error),
        ];
    }

    return { view };
}


//
// Browse-image display
//

/// Active view for the browse-image pane.  No arguments; the src= for
/// the image comes from STATE (see above).
function BrowseImage() {
    const m = window.m;

    // Internal state: have we tried to load the image yet, and if so,
    // did that succeed? Possible values are "unloaded", "ok", and "fail".
    let status = "unloaded";

    // Internal state: the last image we tried to load.  Needed so we
    // notice when STATE.browse_image changes.
    let last_image = null;

    // Auto-redraw handles view updates after these events fire.
    function onload() {
        status = "ok";
    }
    function onerror() {
        // Disappointingly, the event object doesn't record the
        // HTTP error code, or any other details worth reporting.
        // In context, this _should_ only happen when the browse
        // image collection is incomplete, so we just say the image
        // is "missing" in the visible error message.
        status = "error";
    }

    function view() {
        if (STATE.browse_image !== last_image) {
            last_image = STATE.browse_image;
            status = "unloaded";
        }

        let contents = [];
        if (status == "error") {
            contents.push(m("div.no-image", [
                "Browse image missing:",
                m("br"),
                m("code", [last_image])
            ]));
        } else if (last_image == null) {
            // don't display an image in this case
            status = "ok";
        } else {
            let attrs = { src: `/browse/${last_image}` };
            if (status == "unloaded") {
                attrs.onload = onload;
                attrs.onerror = onerror;
            }
            contents.push(m("img", attrs));
        }
        return contents;
    }

    return { view };
}

/// Static view for the browse-image pane, used when browse images are
/// not available
function NoBrowseImages() {
    const m = window.m;
    return {
        view: function () {
            return m("div.no-image", [
                "Browse images", m("br"), "not available"
            ]);
        }
    };
}


//
// Menus
//

/// Subroutine of the menu view methods; create a vnode tree for
/// a <select> element.
///
/// ID       HTML id and name to use for the element; it needs to be
///              globally unique.
/// CLS      space-separated list of class tags to apply to the element.
/// LABEL    human visible name of the <select>, or null to omit one.
///              (This will populate a <label for=ID> next to the <select>.)
/// CHOICES  list of options.  Each element of the list must be a 2-tuple
///              whose first element is the internal "value" for that choice,
///              and whose second element is the human-visible text.
/// ISEL     Internal "value" of the choice that should initially be selected.
///              Null means use browser's default (option 0).
/// ONCHANGE event handler function for the "change" event for this <select>.
///              Null means don't establish an event handler.
function make_dropdown({ id, cls, label, choices, isel, onchange }) {
    const m = window.m;

    let sel_attrs = {
        name: id,
        id: id,
    };
    if (cls)
        sel_attrs["class"] = cls;
    if (onchange != null) // also excludes undefined
        sel_attrs["onchange"] = onchange;

    let select = m("select", sel_attrs, choices.map((opt) => {
        let opt_attrs = {"value": opt[0]};
        if (isel == opt[0])
            opt_attrs["selected"] = "selected";
        return m("option", opt_attrs, [opt[1]]);
    }));

    if (!label) {
        return select;
    }
    return m("div.label-select-wrapper", [
        m("label", { "for": id }, [label]),
        select
    ]);
}

/// This is just a wrapper because m.mount expects to mount a single
/// component on a single element.
///
/// MENUS  list of PrimaryMenu components.
function MenuBar(menus) {
    const m = window.m;
    return {
        view: function() {
            return menus.map(m);
        }
    };
}

/// A "primary" menu is one of the top-level inhabitants of the menu
/// bar.  It may have "secondary" submenus.
///
/// ID           HTML id of the menu; it needs to be globally unique.
///                This also determines which field of STATE controls
///                the selected value and is poked upon a change event.
/// LABEL        human visible name of the menu.
/// CHOICES      list of menu options (see make_dropdown for specifics)
/// SECONDARIES  list of SecondaryMenu components forming the submenus
function PrimaryMenu({ id, label, choices, secondaries }) {
    const m = window.m;
    secondaries = secondaries ?? [];

    function onchange(event) {
        STATE[id] = event.target.value;
        STATE.refresh_main_plot();
    }

    return {
        view: () => {
            return m("details.menu", { open: "open" }, [
                m("summary", [m("span.vcenter-menu-name", [label])]),
                make_dropdown({
                    id, choices, onchange,
                    isel: STATE[id],
                    cls: "menu-primary"
                }),
                m("div.submenus", secondaries.map(
                    (submenu) => m(submenu, { "primary_selection": STATE[id] })
                )),
            ]);
        }
    };
}

// A "secondary" menu is a submenu of a primary menu, which is visible,
// or not, depending on the currently selected option in the primary menu.
// There are multiple kinds of secondary menus, presenting different UIs.

/// `SingleSecondaryMenu` presents a single drop-down menu when visible.
///
/// PRIMARIES  list of choice values *in this menu's primary menu*
///                for which this menu should be visible.
/// ID         HTML id of the menu; it needs to be globally unique.
/// LABEL      human visible name of the menu.
/// CHOICES    list of menu options (see make_dropdown for specifics)
function SingleSecondaryMenu({ primaries, id, label, choices }) {
    const m = window.m;

    function onchange(event) {
        // placeholder
    }

    return {
        view: (vnode) => {
            let active = primaries.includes(vnode.attrs.primary_selection);
            let classes = active ? "submenu active" : "submenu";
            return m("div", { "class": classes }, [
                make_dropdown({
                    id, label, choices, onchange, cls: "menu-secondary"
                })
            ]);
        }
    };
}

/// `LRSecondaryMenu` presents a pair of drop-down menus, described as
/// "left" and "right", when active.
///
/// PRIMARIES  list of choice values *in this menu's primary menu*
///                for which this menu should be visible.
/// L_ID       HTML id of the left menu; it needs to be globally unique.
/// L_LABEL    human visible name of the left menu.
/// L_CHOICES  list of options for the left menu.
/// R_ID       HTML id of the right menu; it needs to be globally unique.
/// R_LABEL    human visible name of the right menu.
/// R_CHOICES  list of options for the right menu.
function LRSecondaryMenu({
    primaries,
    l_label, l_id, l_choices,
    r_label, r_id, r_choices
}) {
    const m = window.m;

    function l_onchange(event) {
        // placeholder
    }
    function r_onchange(event) {
        // placeholder
    }

    return {
        view: (vnode) => {
            let active = primaries.includes(vnode.attrs.primary_selection);
            let classes = active ? "submenu active" : "submenu";
            return m("div", { "class": classes }, [
                make_dropdown({
                    id: l_id, label: l_label, choices: l_choices,
                    onchange: l_onchange, cls: "menu-secondary"
                }),
                make_dropdown({
                    id: r_id, label: r_label, choices: r_choices,
                    onchange: r_onchange, cls: "menu-secondary",
                }),
            ]);
        }
    };
}

/// `LCRSecondaryMenu` presents a triplet of drop-down menus, described as
/// "left", "center", and "right", when active.
///
/// PRIMARIES  list of choice values *in this menu's primary menu*
///                for which this menu should be visible.
/// L_ID       HTML id of the left menu; it needs to be globally unique.
/// L_LABEL    human visible name of the left menu.
/// L_CHOICES  list of options for the left menu.
/// C_ID       HTML id of the center menu; it needs to be globally unique.
/// C_LABEL    human visible name of the center menu.
/// C_CHOICES  list of options for the center menu.
/// R_ID       HTML id of the right menu; it needs to be globally unique.
/// R_LABEL    human visible name of the right menu.
/// R_CHOICES  list of options for the right menu.
function LCRSecondaryMenu({
    primaries,
    l_label, l_id, l_choices,
    c_label, c_id, c_choices,
    r_label, r_id, r_choices
}) {
    const m = window.m;

    function l_onchange(event) {
        // placeholder
    }
    function c_onchange(event) {
        // placeholder
    }
    function r_onchange(event) {
        // placeholder
    }

    return {
        view: (vnode) => {
            let active = primaries.includes(vnode.attrs.primary_selection);
            let classes = active ? "submenu active" : "submenu";
            return m("div", { "class": classes }, [
                make_dropdown({
                    id: l_id, label: l_label, choices: l_choices,
                    onchange: l_onchange, cls: "menu-secondary",
                }),
                make_dropdown({
                    id: c_id, label: c_label, choices: c_choices,
                    onchange: c_onchange, cls: "menu-secondary",
                }),
                make_dropdown({
                    id: r_id, label: r_label, choices: r_choices,
                    onchange: r_onchange, cls: "menu-secondary",
                }),
            ]);
        }
    };
}

function make_menus(colspecs) {
    let choices = [];
    for (let [col, spec] of Object.entries(colspecs)) {
        if (spec.meta) {
            choices.push([col, col]);

            // Use the first quantitative column as the x-axis default,
            // the second quantitative column as the y-axis default,
            // and the first qualitative column as the marker default.
            if (spec.type == "quant") {
                if (STATE.x_col == null) {
                    STATE.x_col = col;
                } else if (STATE.y_col == null) {
                    STATE.y_col = col;
                }
            } else if (spec.type == "qual") {
                if (STATE.m_col == null) {
                    STATE.m_col = col;
                }
            }
        }
    }
    return [
        PrimaryMenu({
            id: "x_col", label: "x axis", choices
        }),
        PrimaryMenu({
            id: "y_col", label: "y axis", choices
        }),
        PrimaryMenu({
            id: "m_col", label: "markers", choices
        }),
    ];
}

function onDOMContentLoaded () {
    const m = window.m;
    m.request({ url: "/data/columns" })
        .then((colspecs) => {
            STATE.columns_available = colspecs;

            if ("images_left" in colspecs
                || "images_right" in colspecs) {
                m.mount(document.getElementById("browse-image"),
                        BrowseImage());
                STATE.browse_images_available = true;
            } else {
                // use m.render here so Mithril knows it won't change
                m.render(document.getElementById("browse-image"),
                         m(NoBrowseImages()));
                STATE.browse_images_available = false;
            }

            m.mount(document.getElementById("main-plot"), MainPlot());
            m.mount(document.getElementById("refl-plot"), ReflPlot());

            // this has the side effect of working out which columns
            // should be selected by default:
            m.mount(document.getElementById("menubar"),
                    MenuBar(make_menus(colspecs)));

            // so now we can call this:
            STATE.refresh_main_plot();
        })
        .catch((error) => {
            console.error(error);
        });
}

if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", onDOMContentLoaded);
} else {
    // already loaded enough
    onDOMContentLoaded();
}
