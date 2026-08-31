"use strict";

// Utilities

/// If ITEM is neither null nor undefined, append it to ARRAY.
/// Otherwise do nothing.
function push_nonnull(array, item) {
    if (item != null)
        array.push(item);
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
            STATE.refl_plot_data = spectrum;
            STATE.refl_plot_error = null;
        }).catch((err) => {
            STATE.refl_plot_data = null;
            STATE.refl_plot_error = err;
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

    function bool_control(id, label, property) {
        let iattrs = {
            type: "checkbox",
            switch: "switch",
            id,
            name: id,
            onchange: (e) => {
                console.log(e);
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

    function render_controls() {
        return m("fieldset.controls", [
            m("legend", ["Plot options:"]),
            bool_control("refl-avg", "Averaged bands", "avg"),
            bool_control("refl-bayer", "Bayer bands", "bayer"),
        ]);
    }

    function render_blank() {
        return m("p#refl-no-selection", [
            STATE.loading() ? "" : "Select an observation to see its spectrum."
        ]);
    }
    function render_err(err) {
        return m("p#refl-error", [err]);
    }
    function render_spectrum(spec) {
        let heads = [
            "band", "wave", "mean", "std.dev."
        ].map((label) => m("th", [label]));
        let rows = Object.entries(spec).map(
            ([band, { wave, mean, std }], _) => [band, wave, mean, std]
        );
        rows.sort((a, b) => a[0] - b[0]);

        let id = STATE.selected_main_obs_id;

        return m("table#refl-data", [
            m("thead", [m("tr", heads)]),
            m("tbody", rows.map(
                (row) => m("tr", { key: `${id},${row[0]}` },
                           row.map((cell) => m("td", cell)))
            )),
        ]);
    }
    function view() {
        let data = STATE.refl_plot_data;
        let err = STATE.refl_plot_error;
        let fragment = [ render_controls() ];

        if (data == null && err == null) {
            fragment.push(render_blank());
        } else if (data != null && err == null) {
            fragment.push(render_spectrum(data));
        } else if (data == null && err != null) {
            fragment.push(render_err(`${err}`));
        } else {
            fragment.push(render_err("impossible: data and err both non-null"));
        }
        return fragment;
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
