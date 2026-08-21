"use strict";

/// Subroutine of the menu view methods; create a vnode tree for
/// a <select> element.
///
/// ID       HTML id and name to use for the element; it needs to be
///              globally unique.
/// KLASS    space-separated list of class tags to apply to the element.
/// LABEL    human visible name of the <select>, or null to omit one.
///              (This will populate a <label for=ID> next to the <select>.)
/// CHOICES  list of options.  Each element of the list must be a 2-tuple
///              whose first element is the internal "value" for that choice,
///              and whose second element is the human-visible text.
/// ONCHANGE event handler function for the "change" event for this <select>.
function _makeDropdown(id, klass, label, choices, onchange) {
    const m = window.m;

    let select = m(
        "select",
        { "class": klass, "name": id, "id": id, "onchange": onchange },
        choices.map((opt) => m("option", {"value": opt[0]}, [opt[1]]))
    );

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
            return m("nav#menubar", menus.map(m));
        }
    };
}

/// A "primary" menu is one of the top-level inhabitants of the menu
/// bar.  It may have "secondary" submenus.
///
/// ID           HTML id of the menu; it needs to be globally unique.
/// LABEL        human visible name of the menu.
/// CHOICES      list of menu options (see _makeDropdown for specifics)
/// SECONDARIES  list of SecondaryMenu components forming the submenus
function PrimaryMenu(id, label, choices, secondaries) {
    const m = window.m;

    // since we don't mark any of the choices as explicitly selected,
    // the browser will default the <select> to showing the first option
    let selected = choices[0][0];

    function onchange(event) {
        selected = event.target.value;
    }

    return {
        view: () => {
            return m("details.menu", [
                m("summary", [m("span.vcenter-menu-name", [label])]),
                _makeDropdown(id, "menu-primary", null, choices, onchange),
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
/// CHOICES    list of menu options (see _makeDropdown for specifics)
function SingleSecondaryMenu(primaries, id, label, choices) {
    const m = window.m;

    function onchange(event) {
        // placeholder
    }

    return {
        view: (vnode) => {
            let active = primaries.includes(vnode.attrs.primary_selection);
            let classes = active ? "submenu active" : "submenu";
            return m("div", { "class": classes }, [
                _makeDropdown(id, "menu-secondary", label, choices, onchange)
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
function LRSecondaryMenu(
    primaries,
    l_label, l_id, l_choices,
    r_label, r_id, r_choices
) {
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
                _makeDropdown(l_id, "menu-secondary",
                              l_label, l_choices, l_onchange),
                _makeDropdown(r_id, "menu-secondary",
                              r_label, r_choices, r_onchange),
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
function LCRSecondaryMenu(
    primaries,
    l_label, l_id, l_choices,
    c_label, c_id, c_choices,
    r_label, r_id, r_choices
) {
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
                _makeDropdown(l_id, "menu-secondary",
                              l_label, l_choices, l_onchange),
                _makeDropdown(c_id, "menu-secondary",
                              c_label, c_choices, c_onchange),
                _makeDropdown(r_id, "menu-secondary",
                              r_label, r_choices, r_onchange),
            ]);
        }
    };
}

function make_menus(colspecs) {
    let axis_choices = [];
    for (let [col, spec] of Object.entries(colspecs)) {
        if (spec.meta) {
            axis_choices.push([col, col]);
        }
    }
    return [
        PrimaryMenu("x-primary", "x axis", axis_choices, []),
        PrimaryMenu("y-primary", "y axis", axis_choices, []),
        PrimaryMenu("m-primary", "markers", axis_choices, []),
    ];
}

function onDOMContentLoaded () {
    const m = window.m;
    m.request({ url: "/data/columns" })
        .then((colspecs) => {
            m.mount(document.getElementById("menubar"),
                    MenuBar(make_menus(colspecs)));
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
