import h from "hscrpt";


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

    let sel_attrs = {
        name: id,
        id: id,
    };
    if (cls)
        sel_attrs["class"] = cls;
    if (onchange != null) // also excludes undefined
        sel_attrs["onchange"] = onchange;

    let select = h("select", sel_attrs, choices.map((opt) => {
        let opt_attrs = {"value": opt[0]};
        if (isel == opt[0])
            opt_attrs["selected"] = "selected";
        return h("option", opt_attrs, [opt[1]]);
    }));

    if (!label) {
        return select;
    }
    return h("div", {"class": "label-select-wrapper"}, [
        h("label", { "for": id }, [label]),
        select
    ]);
}

/// This is just a wrapper because m.mount expects to mount a single
/// component on a single element.
///
/// MENUS  list of PrimaryMenu components.
function MenuBar(menus) {
    return menus.map(h);
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
    secondaries = secondaries ?? [];

    function onchange(event) {
        STATE[id] = event.target.value;
        STATE.refresh_main_plot();
    }

    return {
        view: () => {
            return h("detail", { "class": "menu", "open": "open" }, [
                h("summary", [h("span", {"class":"vcenter-menu-name"},
                                [label])]),
                make_dropdown({
                    id, choices, onchange,
                    isel: STATE[id],
                    cls: "menu-primary"
                }),
                h("div", {"class": "submenus"}, secondaries.map(
                    (submenu) => h(submenu, { "primary_selection": STATE[id] })
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
    function onchange(event) {
        // placeholder
    }

    return {
        view: (vnode) => {
            let active = primaries.includes(vnode.attrs.primary_selection);
            let classes = active ? "submenu active" : "submenu";
            return h("div", { "class": classes }, [
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
            return h("div", { "class": classes }, [
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
            return h("div", { "class": classes }, [
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
