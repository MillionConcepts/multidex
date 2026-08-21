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

// The data set from which menus are generated is temporarily hardcoded.
const MENUS = [
    {
        "id": "x-primary",
        "label": "x axis",
        "choices": [
            ["emission_angle", "emission_angle"],
            ["incidence_angle", "incidence_angle"],
            ["lat", "lat"],
            ["lon", "lon"],
            ["l_s", "l_s"],
            ["ltst", "ltst"],
            ["min_count", "min_count"],
            ["odometry", "odometry"],
            ["phase_angle", "phase_angle"],
            ["rover_elevation", "rover_elevation"],
            ["sclk", "sclk"],
            ["sol", "sol"],
            ["feature", "feature"],
            ["feature_subtype", "feature_subtype"],
            ["float", "float"],
            ["formation", "formation"],
            ["group", "group"],
            ["lab_spectrum_type", "lab_spectrum_type"],
            ["member", "member"],
            ["rock_class", "rock_class"],
            ["soil_class", "soil_class"],
            ["ref", "ref"],
            ["slope", "slope"],
            ["band_avg", "band_avg"],
            ["band_max", "band_max"],
            ["band_min", "band_min"],
            ["ratio", "ratio"],
            ["band_depth", "band_depth"],
            ["PCA", "PCA"],
            ["filter_avg", "filter_avg"],
            ["std_avg", "std_avg"],
            ["rel_std_avg", "rel_std_avg"],
            ["l_rmad", "l_rmad"],
            ["r_rmad", "r_rmad"],
            ["l_rstd", "l_rstd"],
            ["r_rstd", "r_rstd"],
            ["mean_wrasd", "mean_wrasd"],
            ["max_wrasd", "max_wrasd"],
            ["mean_wasd", "mean_wasd"],
            ["max_wasd", "max_wasd"],
            ["p2p", "p2p"],
        ],
        "secondaries": [
            {
                "primaries": ["ref"],
                "type": "single",
                "id": "x-one-band",
                "label": "band",
                "choices": [
                    ["L2", " 445 nm (L2)"],
                    ["R2", " 447 nm (R2)"],
                    ["L0B", " 495 nm (L0B)"],
                    ["R0B", " 495 nm (R0B)"],
                    ["L1", " 527 nm (L1)"],
                    ["R1", " 527 nm (R1)"],
                    ["R0G", " 551 nm (R0G)"],
                    ["L0G", " 554 nm (L0G)"],
                    ["R0R", " 638 nm (R0R)"],
                    ["L0R", " 640 nm (L0R)"],
                    ["L4", " 676 nm (L4)"],
                    ["L3", " 751 nm (L3)"],
                    ["R3", " 805 nm (R3)"],
                    ["L5", " 867 nm (L5)"],
                    ["R4", " 908 nm (R4)"],
                    ["R5", " 937 nm (R5)"],
                    ["L6", "1012 nm (L6)"],
                    ["R6", "1013 nm (R6)"],
                ]
            },
            {
                "primaries": [
                    "slope", "band_avg", "band_max", "band_min", "ratio"
                ],
                "type": "lr",
                "l_id": "x-two-bands-left",
                "l_label": "left",
                "l_choices": [
                    ["L2", " 445 nm (L2)"],
                    ["R2", " 447 nm (R2)"],
                    ["L0B", " 495 nm (L0B)"],
                    ["R0B", " 495 nm (R0B)"],
                    ["L1", " 527 nm (L1)"],
                    ["R1", " 527 nm (R1)"],
                    ["R0G", " 551 nm (R0G)"],
                    ["L0G", " 554 nm (L0G)"],
                    ["R0R", " 638 nm (R0R)"],
                    ["L0R", " 640 nm (L0R)"],
                    ["L4", " 676 nm (L4)"],
                    ["L3", " 751 nm (L3)"],
                    ["R3", " 805 nm (R3)"],
                    ["L5", " 867 nm (L5)"],
                    ["R4", " 908 nm (R4)"],
                    ["R5", " 937 nm (R5)"],
                    ["L6", "1012 nm (L6)"],
                    ["R6", "1013 nm (R6)"],
                ],
                "r_id": "x-two-bands-right",
                "r_label": "right",
                "r_choices": [
                    ["L2", " 445 nm (L2)"],
                    ["R2", " 447 nm (R2)"],
                    ["L0B", " 495 nm (L0B)"],
                    ["R0B", " 495 nm (R0B)"],
                    ["L1", " 527 nm (L1)"],
                    ["R1", " 527 nm (R1)"],
                    ["R0G", " 551 nm (R0G)"],
                    ["L0G", " 554 nm (L0G)"],
                    ["R0R", " 638 nm (R0R)"],
                    ["L0R", " 640 nm (L0R)"],
                    ["L4", " 676 nm (L4)"],
                    ["L3", " 751 nm (L3)"],
                    ["R3", " 805 nm (R3)"],
                    ["L5", " 867 nm (L5)"],
                    ["R4", " 908 nm (R4)"],
                    ["R5", " 937 nm (R5)"],
                    ["L6", "1012 nm (L6)"],
                    ["R6", "1013 nm (R6)"],
                ]
            },
            {
                "primaries": ["band_depth"],
                "type": "lcr",
                "l_id": "x-three-bands-left",
                "l_label": "left",
                "l_choices": [
                    ["L2", " 445 nm (L2)"],
                    ["R2", " 447 nm (R2)"],
                    ["L0B", " 495 nm (L0B)"],
                    ["R0B", " 495 nm (R0B)"],
                    ["L1", " 527 nm (L1)"],
                    ["R1", " 527 nm (R1)"],
                    ["R0G", " 551 nm (R0G)"],
                    ["L0G", " 554 nm (L0G)"],
                    ["R0R", " 638 nm (R0R)"],
                    ["L0R", " 640 nm (L0R)"],
                    ["L4", " 676 nm (L4)"],
                    ["L3", " 751 nm (L3)"],
                    ["R3", " 805 nm (R3)"],
                    ["L5", " 867 nm (L5)"],
                    ["R4", " 908 nm (R4)"],
                    ["R5", " 937 nm (R5)"],
                    ["L6", "1012 nm (L6)"],
                    ["R6", "1013 nm (R6)"],
                ],
                "c_id": "x-three-bands-center",
                "c_label": "center",
                "c_choices": [
                    ["L2", " 445 nm (L2)"],
                    ["R2", " 447 nm (R2)"],
                    ["L0B", " 495 nm (L0B)"],
                    ["R0B", " 495 nm (R0B)"],
                    ["L1", " 527 nm (L1)"],
                    ["R1", " 527 nm (R1)"],
                    ["R0G", " 551 nm (R0G)"],
                    ["L0G", " 554 nm (L0G)"],
                    ["R0R", " 638 nm (R0R)"],
                    ["L0R", " 640 nm (L0R)"],
                    ["L4", " 676 nm (L4)"],
                    ["L3", " 751 nm (L3)"],
                    ["R3", " 805 nm (R3)"],
                    ["L5", " 867 nm (L5)"],
                    ["R4", " 908 nm (R4)"],
                    ["R5", " 937 nm (R5)"],
                    ["L6", "1012 nm (L6)"],
                    ["R6", "1013 nm (R6)"],
                ],
                "r_id": "x-three-bands-right",
                "r_label": "right",
                "r_choices": [
                    ["L2", " 445 nm (L2)"],
                    ["R2", " 447 nm (R2)"],
                    ["L0B", " 495 nm (L0B)"],
                    ["R0B", " 495 nm (R0B)"],
                    ["L1", " 527 nm (L1)"],
                    ["R1", " 527 nm (R1)"],
                    ["R0G", " 551 nm (R0G)"],
                    ["L0G", " 554 nm (L0G)"],
                    ["R0R", " 638 nm (R0R)"],
                    ["L0R", " 640 nm (L0R)"],
                    ["L4", " 676 nm (L4)"],
                    ["L3", " 751 nm (L3)"],
                    ["R3", " 805 nm (R3)"],
                    ["L5", " 867 nm (L5)"],
                    ["R4", " 908 nm (R4)"],
                    ["R5", " 937 nm (R5)"],
                    ["L6", "1012 nm (L6)"],
                    ["R6", "1013 nm (R6)"],
                ],
            },
            {
                "primaries": ["PCA"],
                "type": "single",
                "id": "x-pca-component",
                "label": "component #",
                "choices": [
                    ["1", "1"],
                    ["2", "2"],
                    ["3", "3"],
                    ["4", "4"],
                    ["5", "5"],
                    ["6", "6"],
                ]
            }
        ],
    },
    {
        "label": "y axis",
        "id": "y-primary",
        "choices": [
            ["emission_angle", "emission_angle"],
            ["incidence_angle", "incidence_angle"],
            ["lat", "lat"],
            ["lon", "lon"],
            ["l_s", "l_s"],
            ["ltst", "ltst"],
            ["min_count", "min_count"],
            ["odometry", "odometry"],
            ["phase_angle", "phase_angle"],
            ["rover_elevation", "rover_elevation"],
            ["sclk", "sclk"],
            ["sol", "sol"],
            ["feature", "feature"],
            ["feature_subtype", "feature_subtype"],
            ["float", "float"],
            ["formation", "formation"],
            ["group", "group"],
            ["lab_spectrum_type", "lab_spectrum_type"],
            ["member", "member"],
            ["rock_class", "rock_class"],
            ["soil_class", "soil_class"],
            ["ref", "ref"],
            ["slope", "slope"],
            ["band_avg", "band_avg"],
            ["band_max", "band_max"],
            ["band_min", "band_min"],
            ["ratio", "ratio"],
            ["band_depth", "band_depth"],
            ["PCA", "PCA"],
            ["filter_avg", "filter_avg"],
            ["std_avg", "std_avg"],
            ["rel_std_avg", "rel_std_avg"],
            ["l_rmad", "l_rmad"],
            ["r_rmad", "r_rmad"],
            ["l_rstd", "l_rstd"],
            ["r_rstd", "r_rstd"],
            ["mean_wrasd", "mean_wrasd"],
            ["max_wrasd", "max_wrasd"],
            ["mean_wasd", "mean_wasd"],
            ["max_wasd", "max_wasd"],
            ["p2p", "p2p"],
        ],
        "secondaries": [
            {
                "primaries": ["ref"],
                "type": "single",
                "id": "y-one-band",
                "label": "band",
                "choices": [
                    ["L2", " 445 nm (L2)"],
                    ["R2", " 447 nm (R2)"],
                    ["L0B", " 495 nm (L0B)"],
                    ["R0B", " 495 nm (R0B)"],
                    ["L1", " 527 nm (L1)"],
                    ["R1", " 527 nm (R1)"],
                    ["R0G", " 551 nm (R0G)"],
                    ["L0G", " 554 nm (L0G)"],
                    ["R0R", " 638 nm (R0R)"],
                    ["L0R", " 640 nm (L0R)"],
                    ["L4", " 676 nm (L4)"],
                    ["L3", " 751 nm (L3)"],
                    ["R3", " 805 nm (R3)"],
                    ["L5", " 867 nm (L5)"],
                    ["R4", " 908 nm (R4)"],
                    ["R5", " 937 nm (R5)"],
                    ["L6", "1012 nm (L6)"],
                    ["R6", "1013 nm (R6)"],
                ]
            },
            {
                "primaries": [
                    "slope", "band_avg", "band_max", "band_min", "ratio"
                ],
                "type": "lr",
                "l_id": "y-two-bands-left",
                "l_label": "left",
                "l_choices": [
                    ["L2", " 445 nm (L2)"],
                    ["R2", " 447 nm (R2)"],
                    ["L0B", " 495 nm (L0B)"],
                    ["R0B", " 495 nm (R0B)"],
                    ["L1", " 527 nm (L1)"],
                    ["R1", " 527 nm (R1)"],
                    ["R0G", " 551 nm (R0G)"],
                    ["L0G", " 554 nm (L0G)"],
                    ["R0R", " 638 nm (R0R)"],
                    ["L0R", " 640 nm (L0R)"],
                    ["L4", " 676 nm (L4)"],
                    ["L3", " 751 nm (L3)"],
                    ["R3", " 805 nm (R3)"],
                    ["L5", " 867 nm (L5)"],
                    ["R4", " 908 nm (R4)"],
                    ["R5", " 937 nm (R5)"],
                    ["L6", "1012 nm (L6)"],
                    ["R6", "1013 nm (R6)"],
                ],
                "r_id": "y-two-bands-right",
                "r_label": "right",
                "r_choices": [
                    ["L2", " 445 nm (L2)"],
                    ["R2", " 447 nm (R2)"],
                    ["L0B", " 495 nm (L0B)"],
                    ["R0B", " 495 nm (R0B)"],
                    ["L1", " 527 nm (L1)"],
                    ["R1", " 527 nm (R1)"],
                    ["R0G", " 551 nm (R0G)"],
                    ["L0G", " 554 nm (L0G)"],
                    ["R0R", " 638 nm (R0R)"],
                    ["L0R", " 640 nm (L0R)"],
                    ["L4", " 676 nm (L4)"],
                    ["L3", " 751 nm (L3)"],
                    ["R3", " 805 nm (R3)"],
                    ["L5", " 867 nm (L5)"],
                    ["R4", " 908 nm (R4)"],
                    ["R5", " 937 nm (R5)"],
                    ["L6", "1012 nm (L6)"],
                    ["R6", "1013 nm (R6)"],
                ]
            },
            {
                "primaries": ["band_depth"],
                "type": "lcr",
                "l_id": "y-three-bands-left",
                "l_label": "left",
                "l_choices": [
                    ["L2", " 445 nm (L2)"],
                    ["R2", " 447 nm (R2)"],
                    ["L0B", " 495 nm (L0B)"],
                    ["R0B", " 495 nm (R0B)"],
                    ["L1", " 527 nm (L1)"],
                    ["R1", " 527 nm (R1)"],
                    ["R0G", " 551 nm (R0G)"],
                    ["L0G", " 554 nm (L0G)"],
                    ["R0R", " 638 nm (R0R)"],
                    ["L0R", " 640 nm (L0R)"],
                    ["L4", " 676 nm (L4)"],
                    ["L3", " 751 nm (L3)"],
                    ["R3", " 805 nm (R3)"],
                    ["L5", " 867 nm (L5)"],
                    ["R4", " 908 nm (R4)"],
                    ["R5", " 937 nm (R5)"],
                    ["L6", "1012 nm (L6)"],
                    ["R6", "1013 nm (R6)"],
                ],
                "c_id": "y-three-bands-center",
                "c_label": "center",
                "c_choices": [
                    ["L2", " 445 nm (L2)"],
                    ["R2", " 447 nm (R2)"],
                    ["L0B", " 495 nm (L0B)"],
                    ["R0B", " 495 nm (R0B)"],
                    ["L1", " 527 nm (L1)"],
                    ["R1", " 527 nm (R1)"],
                    ["R0G", " 551 nm (R0G)"],
                    ["L0G", " 554 nm (L0G)"],
                    ["R0R", " 638 nm (R0R)"],
                    ["L0R", " 640 nm (L0R)"],
                    ["L4", " 676 nm (L4)"],
                    ["L3", " 751 nm (L3)"],
                    ["R3", " 805 nm (R3)"],
                    ["L5", " 867 nm (L5)"],
                    ["R4", " 908 nm (R4)"],
                    ["R5", " 937 nm (R5)"],
                    ["L6", "1012 nm (L6)"],
                    ["R6", "1013 nm (R6)"],
                ],
                "r_id": "y-three-bands-right",
                "r_label": "right",
                "r_choices": [
                    ["L2", " 445 nm (L2)"],
                    ["R2", " 447 nm (R2)"],
                    ["L0B", " 495 nm (L0B)"],
                    ["R0B", " 495 nm (R0B)"],
                    ["L1", " 527 nm (L1)"],
                    ["R1", " 527 nm (R1)"],
                    ["R0G", " 551 nm (R0G)"],
                    ["L0G", " 554 nm (L0G)"],
                    ["R0R", " 638 nm (R0R)"],
                    ["L0R", " 640 nm (L0R)"],
                    ["L4", " 676 nm (L4)"],
                    ["L3", " 751 nm (L3)"],
                    ["R3", " 805 nm (R3)"],
                    ["L5", " 867 nm (L5)"],
                    ["R4", " 908 nm (R4)"],
                    ["R5", " 937 nm (R5)"],
                    ["L6", "1012 nm (L6)"],
                    ["R6", "1013 nm (R6)"],
                ],
            },
            {
                "primaries": ["PCA"],
                "type": "single",
                "id": "y-pca-component",
                "label": "component #",
                "choices": [
                    ["1", "1"],
                    ["2", "2"],
                    ["3", "3"],
                    ["4", "4"],
                    ["5", "5"],
                    ["6", "6"],
                ]
            }
        ],
    },
    {
        "label": "markers",
        "id": "m-primary",
        "choices": [
            ["emission_angle", "emission_angle"],
            ["incidence_angle", "incidence_angle"],
            ["lat", "lat"],
            ["lon", "lon"],
            ["l_s", "l_s"],
            ["ltst", "ltst"],
            ["min_count", "min_count"],
            ["odometry", "odometry"],
            ["phase_angle", "phase_angle"],
            ["rover_elevation", "rover_elevation"],
            ["sclk", "sclk"],
            ["sol", "sol"],
            ["feature", "feature"],
            ["feature_subtype", "feature_subtype"],
            ["float", "float"],
            ["formation", "formation"],
            ["group", "group"],
            ["lab_spectrum_type", "lab_spectrum_type"],
            ["member", "member"],
            ["rock_class", "rock_class"],
            ["soil_class", "soil_class"],
            ["ref", "ref"],
            ["slope", "slope"],
            ["band_avg", "band_avg"],
            ["band_max", "band_max"],
            ["band_min", "band_min"],
            ["ratio", "ratio"],
            ["band_depth", "band_depth"],
            ["PCA", "PCA"],
            ["filter_avg", "filter_avg"],
            ["std_avg", "std_avg"],
            ["rel_std_avg", "rel_std_avg"],
            ["l_rmad", "l_rmad"],
            ["r_rmad", "r_rmad"],
            ["l_rstd", "l_rstd"],
            ["r_rstd", "r_rstd"],
            ["mean_wrasd", "mean_wrasd"],
            ["max_wrasd", "max_wrasd"],
            ["mean_wasd", "mean_wasd"],
            ["max_wasd", "max_wasd"],
            ["p2p", "p2p"],
        ],
        "secondaries": [
            {
                "primaries": ["ref"],
                "type": "single",
                "id": "m-one-band",
                "label": "band",
                "choices": [
                    ["L2", " 445 nm (L2)"],
                    ["R2", " 447 nm (R2)"],
                    ["L0B", " 495 nm (L0B)"],
                    ["R0B", " 495 nm (R0B)"],
                    ["L1", " 527 nm (L1)"],
                    ["R1", " 527 nm (R1)"],
                    ["R0G", " 551 nm (R0G)"],
                    ["L0G", " 554 nm (L0G)"],
                    ["R0R", " 638 nm (R0R)"],
                    ["L0R", " 640 nm (L0R)"],
                    ["L4", " 676 nm (L4)"],
                    ["L3", " 751 nm (L3)"],
                    ["R3", " 805 nm (R3)"],
                    ["L5", " 867 nm (L5)"],
                    ["R4", " 908 nm (R4)"],
                    ["R5", " 937 nm (R5)"],
                    ["L6", "1012 nm (L6)"],
                    ["R6", "1013 nm (R6)"],
                ]
            },
            {
                "primaries": [
                    "slope", "band_avg", "band_max", "band_min", "ratio"
                ],
                "type": "lr",
                "l_id": "m-two-bands-left",
                "l_label": "left",
                "l_choices": [
                    ["L2", " 445 nm (L2)"],
                    ["R2", " 447 nm (R2)"],
                    ["L0B", " 495 nm (L0B)"],
                    ["R0B", " 495 nm (R0B)"],
                    ["L1", " 527 nm (L1)"],
                    ["R1", " 527 nm (R1)"],
                    ["R0G", " 551 nm (R0G)"],
                    ["L0G", " 554 nm (L0G)"],
                    ["R0R", " 638 nm (R0R)"],
                    ["L0R", " 640 nm (L0R)"],
                    ["L4", " 676 nm (L4)"],
                    ["L3", " 751 nm (L3)"],
                    ["R3", " 805 nm (R3)"],
                    ["L5", " 867 nm (L5)"],
                    ["R4", " 908 nm (R4)"],
                    ["R5", " 937 nm (R5)"],
                    ["L6", "1012 nm (L6)"],
                    ["R6", "1013 nm (R6)"],
                ],
                "r_id": "m-two-bands-right",
                "r_label": "right",
                "r_choices": [
                    ["L2", " 445 nm (L2)"],
                    ["R2", " 447 nm (R2)"],
                    ["L0B", " 495 nm (L0B)"],
                    ["R0B", " 495 nm (R0B)"],
                    ["L1", " 527 nm (L1)"],
                    ["R1", " 527 nm (R1)"],
                    ["R0G", " 551 nm (R0G)"],
                    ["L0G", " 554 nm (L0G)"],
                    ["R0R", " 638 nm (R0R)"],
                    ["L0R", " 640 nm (L0R)"],
                    ["L4", " 676 nm (L4)"],
                    ["L3", " 751 nm (L3)"],
                    ["R3", " 805 nm (R3)"],
                    ["L5", " 867 nm (L5)"],
                    ["R4", " 908 nm (R4)"],
                    ["R5", " 937 nm (R5)"],
                    ["L6", "1012 nm (L6)"],
                    ["R6", "1013 nm (R6)"],
                ]
            },
            {
                "primaries": ["band_depth"],
                "type": "lcr",
                "l_id": "m-three-bands-left",
                "l_label": "left",
                "l_choices": [
                    ["L2", " 445 nm (L2)"],
                    ["R2", " 447 nm (R2)"],
                    ["L0B", " 495 nm (L0B)"],
                    ["R0B", " 495 nm (R0B)"],
                    ["L1", " 527 nm (L1)"],
                    ["R1", " 527 nm (R1)"],
                    ["R0G", " 551 nm (R0G)"],
                    ["L0G", " 554 nm (L0G)"],
                    ["R0R", " 638 nm (R0R)"],
                    ["L0R", " 640 nm (L0R)"],
                    ["L4", " 676 nm (L4)"],
                    ["L3", " 751 nm (L3)"],
                    ["R3", " 805 nm (R3)"],
                    ["L5", " 867 nm (L5)"],
                    ["R4", " 908 nm (R4)"],
                    ["R5", " 937 nm (R5)"],
                    ["L6", "1012 nm (L6)"],
                    ["R6", "1013 nm (R6)"],
                ],
                "c_id": "m-three-bands-center",
                "c_label": "center",
                "c_choices": [
                    ["L2", " 445 nm (L2)"],
                    ["R2", " 447 nm (R2)"],
                    ["L0B", " 495 nm (L0B)"],
                    ["R0B", " 495 nm (R0B)"],
                    ["L1", " 527 nm (L1)"],
                    ["R1", " 527 nm (R1)"],
                    ["R0G", " 551 nm (R0G)"],
                    ["L0G", " 554 nm (L0G)"],
                    ["R0R", " 638 nm (R0R)"],
                    ["L0R", " 640 nm (L0R)"],
                    ["L4", " 676 nm (L4)"],
                    ["L3", " 751 nm (L3)"],
                    ["R3", " 805 nm (R3)"],
                    ["L5", " 867 nm (L5)"],
                    ["R4", " 908 nm (R4)"],
                    ["R5", " 937 nm (R5)"],
                    ["L6", "1012 nm (L6)"],
                    ["R6", "1013 nm (R6)"],
                ],
                "r_id": "m-three-bands-right",
                "r_label": "right",
                "r_choices": [
                    ["L2", " 445 nm (L2)"],
                    ["R2", " 447 nm (R2)"],
                    ["L0B", " 495 nm (L0B)"],
                    ["R0B", " 495 nm (R0B)"],
                    ["L1", " 527 nm (L1)"],
                    ["R1", " 527 nm (R1)"],
                    ["R0G", " 551 nm (R0G)"],
                    ["L0G", " 554 nm (L0G)"],
                    ["R0R", " 638 nm (R0R)"],
                    ["L0R", " 640 nm (L0R)"],
                    ["L4", " 676 nm (L4)"],
                    ["L3", " 751 nm (L3)"],
                    ["R3", " 805 nm (R3)"],
                    ["L5", " 867 nm (L5)"],
                    ["R4", " 908 nm (R4)"],
                    ["R5", " 937 nm (R5)"],
                    ["L6", "1012 nm (L6)"],
                    ["R6", "1013 nm (R6)"],
                ],
            },
            {
                "primaries": ["PCA"],
                "type": "single",
                "id": "m-pca-component",
                "label": "component #",
                "choices": [
                    ["1", "1"],
                    ["2", "2"],
                    ["3", "3"],
                    ["4", "4"],
                    ["5", "5"],
                    ["6", "6"],
                ]
            }
        ],
    },
];

function make_menu(spec) {
    const SECONDARY_COMPONENTS = {
        "single": (s) => SingleSecondaryMenu(
            s.primaries, s.id, s.label, s.choices,
        ),
        "lr": (s) => LRSecondaryMenu(
            s.primaries,
            s.l_label, s.l_id, s.l_choices,
            s.r_label, s.r_id, s.r_choices,
        ),
        "lcr": (s) => LCRSecondaryMenu(
            s.primaries,
            s.l_label, s.l_id, s.l_choices,
            s.c_label, s.c_id, s.c_choices,
            s.r_label, s.r_id, s.r_choices,
        )
    };

    return PrimaryMenu(
        spec.id,
        spec.label,
        spec.choices,
        spec.secondaries.map(
            (s_spec) => SECONDARY_COMPONENTS[s_spec.type](s_spec)
        )
    );
}

function onDOMContentLoaded () {
    const m = window.m;
    let menus = MENUS.map(make_menu);
    m.mount(document.getElementById("menubar"), MenuBar(menus));
}

if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", onDOMContentLoaded);
} else {
    // already loaded enough
    onDOMContentLoaded();
}
