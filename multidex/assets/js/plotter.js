"use strict";

/// Application global state
let STATE = {
    // Browse image currently being displayed.
    browse_image: null,
};

//
// Browse-image handling
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
            let last_image_base = last_image.substring(
                last_image.lastIndexOf("/") + 1
            );
            contents.push(m("div.no-image", [
                "Browse image missing:",
                m("br"),
                m("code", [last_image_base])
            ]));
        } else if (last_image == null) {
            // don't display an image in this case
            status = "ok";
        } else {
            let attrs = { src: last_image };
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
/// LABEL        human visible name of the menu.
/// CHOICES      list of menu options (see make_dropdown for specifics)
/// ISEL         initially selected menu option; null means use choice 0
/// SECONDARIES  list of SecondaryMenu components forming the submenus
function PrimaryMenu({ id, label, choices, isel, secondaries }) {
    const m = window.m;
    secondaries = secondaries ?? [];

    // if we don't have an explicit initial option, the browser will
    // default to showing the first option
    let selected = isel ?? choices[0][0];

    function onchange(event) {
        selected = event.target.value;
    }

    return {
        view: () => {
            return m("details.menu", [
                m("summary", [m("span.vcenter-menu-name", [label])]),
                make_dropdown({
                    id, choices, isel, onchange, cls: "menu-primary"
                }),
                m("div.submenus", secondaries.map(
                    (submenu) => m(submenu, { "primary_selection": selected })
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
    let x_isel = null;
    let y_isel = null;
    let m_isel = null;
    for (let [col, spec] of Object.entries(colspecs)) {
        if (spec.meta) {
            choices.push([col, col]);

            // Use the first quantitative column as the x-axis default,
            // the second quantitative column as the y-axis default,
            // and the first qualitative column as the marker default.
            if (spec.type == "quant") {
                if (x_isel == null) {
                    x_isel = col;
                } else if (y_isel == null) {
                    y_isel = col;
                }
            } else if (spec.type == "qual") {
                if (m_isel == null) {
                    m_isel = col;
                }
            }
        }
    }
    return [
        PrimaryMenu({
            id: "x-primary", label: "x axis", isel: x_isel, choices
        }),
        PrimaryMenu({
            id: "y-primary", label: "y axis", isel: y_isel, choices
        }),
        PrimaryMenu({
            id: "m-primary", label: "markers", isel: m_isel, choices
        }),
    ];
}

function onDOMContentLoaded () {
    const m = window.m;
    m.request({ url: "/data/columns" })
        .then((colspecs) => {
            m.mount(document.getElementById("menubar"),
                    MenuBar(make_menus(colspecs)));
            if ("images_left" in colspecs
                || "images_right" in colspecs) {
                m.mount(document.getElementById("browse-image"),
                        BrowseImage());
            } else {
                // use m.render here so Mithril knows it won't change
                m.render(document.getElementById("browse-image"),
                         m(NoBrowseImages()));
            }
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
