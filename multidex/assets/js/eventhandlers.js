window.onload = function onload () {
    "use strict";

    function menu_change(primary, secondaries) {
        // this should only ever have one element
        const selectedValues =
              Array.from(primary.selectedOptions)
              .map((o) => new RegExp(`\\b${o.value}\\b`));
        for (const submenu of secondaries) {
            // this can have more th
            let primaries = submenu.dataset.primaries ?? "";
            let active = false;
            for (const v of selectedValues) {
                if (v.test(primaries)) {
                    active = true;
                }
            }
            if (active) {
                submenu.classList.add("active");
            } else {
                submenu.classList.remove("active");
            }
        }
    }

    function menu_hydrate(elt) {
        let primary = elt.querySelector(":scope > .menu-primary");
        if (!!primary) {
            let secondaries = Array.from(elt.querySelectorAll(".submenu-for"));
            primary.addEventListener("change", (_) => menu_change(primary, secondaries));
            menu_change(primary, secondaries);
        }
    }


    for (const elt of document.getElementsByClassName("menu")) {
        menu_hydrate(elt);
    }
};
