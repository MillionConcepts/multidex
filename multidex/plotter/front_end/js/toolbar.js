import h from "hscrpt";
import ICON_SPECS from "toolbar-icons";

// Toolbars with a currently selected tool.

// Isolated buttons aren't supported, but you can define a toolbar with
// just one button, which looks and behaves exactly the same as an
// isolated button would.

const TOOLBAR_STYLES = `
:host {
    display: flex;
    flex-direction: row;
    justify-content: start;
    align-items: center;
    margin-bottom: .5rem;
}
.tool-button {
    display: flex;
    padding: .3rem;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
    background: transparent;
    border-color: light-dark(#aaa, #555);
    border-style: solid;
    border-width: 1px 0 1px 1px;
    border-radius: 0;
    position: relative;  /* make this an abspos container */
}
.tool-button.gap-after {
    border-right-width: 1px;
    margin-right: 0.8rem;
}
.tool-button:last-child {
    border-right-width: 1px;
}
.tool-button.gap-after:last-child {
    margin-right: 0px;
}
.tool-button:hover {
    background: light-dark(rgb(0,0,0,.2), rgb(255, 255, 255, .2))
}
.tool-button:active {
    background: light-dark(rgb(0,0,0,.4), rgb(255, 255, 255, .4));
}
.tool-button:disabled:hover, .tool-button:disabled:active {
    background: transparent;
}

.tool-icon {
    width: 1.4rem;
    height: 1.4rem;
    fill: light-dark(#444, #bbb);
}

.tool-button.selected {
    background: light-dark(#444, #bbb);
}
.tool-button.selected > .tool-icon {
    fill: light-dark(#ddd, #222);
}
.tool-button:disabled > .tool-icon {
    fill: light-dark(#aaa, #555);
}

/* tooltips are TODO */
.tool-label { display: none }
`;

/** DOM component for a toolbar.
 *
 * Declarative usage is `<plot-toolbar buttons="..."></plot-toolbar>`.
 * There should never be any child elements.
 *
 * The `buttons` attribute should be a space-separated list of icon names
 * from `ploticon-icons`; each produces one button in the toolbar.  Buttons
 * are normally drawn right next to each other but you can insert a gap with
 * "|", e.g. buttons="zoom-box pan | zoom-minus zoom-plus autoscale" will
 * have a gap between the "pan" and "zoom-minus" buttons.  Note that there
 * must be spaces on either side of the "|", "pan|zoom-minus" will try to
 * look up an icon named "pan|zoom-minus" and crash.  There may not be any
 * duplicate buttons in the toolbar. You can change the set of buttons at
 * any time by altering the attribute's value.
 *
 * The toolbar can host two kinds of buttons; each icon definition specifies
 * which kind of button it is.  "Trigger" buttons represent actions; they
 * have an immediate effect when clicked.  "Select" buttons represent tools;
 * they become selected when clicked (which is visible in the toolbar's
 * appearance) but don't have an effect until the tool is applied to some
 * other object within the application.  At all times, there is *at most*
 * one selected tool per toolbar; clicking a "select" button causes the
 * previously selected tool to be deselected, but clicking a "trigger"
 * button does not change which tool, if any, is selected.
 *
 * When buttons are clicked, they cause custom events to be fired at the
 * toolbar; you're expected to listen for these events and use them to
 * actually do what the button does.  The event name will be one of:
 *
 *  - `toolTriggered`: click on a button that has a direct, immediate effect
 *  - `toolSelected`:  click on a button that merely selects a tool (it will
 *                     have an effect when applied to something else).
 *
 * In both cases the event's `detail` will be an object.  For `toolTriggered`,
 * `detail.triggered` will be the icon name of the button that was clicked.
 * For `toolSelected`, `detail.selected` will be the icon name of the tool
 * that is now selected, and `details.previous` will be the icon name of the
 * tool that *was* selected before the click happened.  It is possible for
 * either of these to be null, but not both.
 *
 * You can programmatically fake a click on any button by firing a custom
 * event at the toolbar, named "activateButton", whose `detail` is the icon
 * name of the button you want to trigger or select, or null if you want to
 * deselect whatever the currently selected tool is.
 *
 * You can enable or disable the entire toolbar by clearing or setting the
 * standard HTML "disabled" attribute on the toolbar.  If the toolbar is
 * enabled, you can enable or disable individual buttons by firing custom
 * events at the toolbar, named "enableButton" or "disableButton"
 * respectively, whose `detail` is the icon name of the button you want to
 * enable or disable (or a list of such icon names).  Disabling the currently
 * selected tool also causes it to be deselected.
 *
 * (Note that when the entire toolbar is disabled, all its buttons have to be
 * given the "disabled" attribute so they behave correctly; for this reason,
 * disabling the entire toolbar causes it to forget which buttons were
 * individually disabled.  When the toolbar is enabled, all the buttons will
 * be active again.  "enableButton" and "disableButton" events are ignored
 * while the toolbar is disabled as a whole.)
 *
 * TODO: tooltip handling.
 *
 * This is *called* PlotToolbarElement and registered as <plot-toolbar>
 * for two reasons: because the *expected* usage is as a toolbar associated
 * with a plot, and to avoid colliding with a hypothetical future <toolbar>
 * element defined by the HTML spec.  That seems like the sort of thing
 * they might decide to add.  There's no actual requirement to use this
 * element that way.
 */
export class PlotToolbarElement extends HTMLElement {
    static observedAttributes = Object.freeze(["buttons", "disabled"]);

    constructor() {
        super();

        let shadow = this.attachShadow({ mode: "open" });
        shadow.adoptedStyleSheets = [];

        (new CSSStyleSheet()).replace(TOOLBAR_STYLES).then((sheet) => {
            shadow.adoptedStyleSheets.push(sheet);
        });

        for (let [e_name, e_handler] of [
            ["enableButton", this.#enableButtonEvent],
            ["disableButton", this.#disableButtonEvent],
            ["activateButton", this.#activateButtonEvent],
            ["click", this.#clickEvent],
        ]) {
            this.addEventListener(e_name, e_handler.bind(this));
        }
    }

    attributeChangedCallback(name, oldValue, newValue) {
        switch (name) {
        case "buttons":
            this.#refreshButtons(oldValue, newValue);
            break;
        case "disabled":
            // *remove* the "disabled" attribute to *enable* the bar; thus,
            // the bar is to be enabled if the new value of the attr is null
            this.#ableBar(newValue == null);
            break;
        default:
            /* no op */
        }
    }

    // make the DOM properties "disabled" and "buttons" reflect the
    // "disabled" and "buttons" properties
    get disabled() {
        return this.hasAttribute("disabled");
    }
    set disabled(val) {
        if (!!val) {
            this.setAttribute("disabled", "");
        } else {
            this.removeAttribute("disabled");
        }
    }

    get buttons() {
        return this.getAttribute("buttons");
    }
    set buttons(val) {
        this.setAttribute("buttons", val);
    }

    #refreshButtons(old_buttons, new_buttons) {
        if (new_buttons === old_buttons) {
            return;
        }

        let new_button_names = new_buttons.split(/\s+/);
        let new_button_els = [];
        let already = {};
        for (let btn of new_button_names) {
            if (btn === "|") {
                if (new_button_els.length > 0) {
                    new_button_els.at(-1).classList.add("gap-after");
                }
            } else {
                let spec = ICON_SPECS[btn];
                if (spec == null)
                    throw new Error(`unknown toolbar button icon: ${btn}`);
                if (Object.hasOwn(already, btn))
                    throw new Error(`button used twice in a toolbar: ${btn}`);
                already[btn] = true;
                new_button_els.push(h_button(btn, spec));
            }
        }

        // gaps only go between buttons, not at the ends
        if (new_button_els.length > 0) {
            new_button_els.at(-1).classList.remove("gap-after");
        }

        this.shadowRoot.replaceChildren(...new_button_els);
        this.#ableBar(!this.hasAttribute("disabled"));
    }

    #enableButtonEvent(event) {
        if (this.disabled)
            return;
        let tools = event.detail;
        if (!Array.isArray(tools))
            tools = [tools];

        for (let tool of tools) {
            this.#ableButton(this.#buttonForTool(tool), true);
        }
    }

    #disableButtonEvent(event) {
        if (this.disabled)
            return;
        let tools = event.detail;
        if (!Array.isArray(tools))
            tools = [tools];

        for (let tool of tools) {
            this.#ableButton(this.#buttonForTool(tool), false);
        }
    }

    #activateButtonEvent(event) {
        if (this.disabled)
            return;
        if (event.detail === null) {
            this.#selectButton(this.#selectedButton(), false);
        } else {
            this.#activateButton(this.#buttonForTool(event.detail), true);
        }
    }

    #clickEvent(event) {
        if (this.disabled)
            return;
        // because the click was on something inside the shadow DOM,
        // event.target will be the toolbar; we have to scan the "composed
        // path" to find the button.
        for (let maybe_target of event.composedPath()) {
            if (maybe_target instanceof HTMLButtonElement) {
                this.#activateButton(maybe_target, null);
                return;
            }
        }
        console.log("stray click", event, event.composedPath());
    }

    #ableBar(enable) {
        for (let btn of this.#getButtons()) {
            this.#ableButton(btn, enable);
        }
    }

    #ableButton(btn, enable) {
        if (enable) {
            btn.disabled = false;
        } else {
            btn.disabled = true;
            this.#selectButton(btn, false);
        }
    }

    #activateButton(btn, sense) {
        if (btn == null)
            return;

        lose_interest(btn);
        if (btn.dataset.behavior === "trigger")
            this.#queueTriggerEvent(btn);
        else
            this.#selectButton(btn, sense);
    }

    #selectButton(btn, sense) {
        let prev = this.#selectedButton();

        if (sense == null) {
            // this is because of a click event; if the currently selected
            // button was clicked (prev === btn), deselect it (sense = false);
            // otherwise select 'btn' and deselect 'prev' (sense = true).
            sense = !(prev === btn);
        }

        if (sense) {
            // force-selecting a button that's already selected is a no-op
            if (prev === btn)
                return;

            btn.classList.add("selected");
            if (prev != null)
                prev.classList.remove("selected");
            this.#queueSelectEvent(btn, prev);

        } else {
            // deselecting a button that's not selected is a no-op
            if (!btn.classList.contains("selected")) {
                return;
            }
            btn.classList.remove("selected");
            this.#queueSelectEvent(null, btn);
        }
    }

    #getButtons() {
        return Array.from(this.shadowRoot.querySelectorAll("button"));
    }

    #selectedButton() {
        for (let btn of this.#getButtons()) {
            if (btn.classList.contains("selected")) {
                return btn;
            }
        }
        return null;
    }

    #buttonForTool(tool) {
        for (let btn of this.#getButtons()) {
            if (tool === btn.dataset.tool) {
                return btn;
            }
        }
        throw new Error(`tool not present: ${tool}`);
    }

    #queueTriggerEvent(triggered) {
        window.queueMicrotask(() => this.dispatchEvent(
            new CustomEvent("toolTriggered", {
                bubbles: true,
                cancelable: true,
                composed: true,
                detail: {
                    triggered: triggered?.dataset?.tool,
                }
            })));
    }

    #queueSelectEvent(selected, previous) {
        window.queueMicrotask(() => this.dispatchEvent(
            new CustomEvent("toolSelected", {
                bubbles: true,
                cancelable: true,
                composed: true,
                detail: {
                    selected: selected?.dataset?.tool,
                    previous: previous?.dataset?.tool,
                }
            })));
    }
}
window.customElements.define("plot-toolbar", PlotToolbarElement);
export default PlotToolbarElement;

//
// Implementation
//

function h_button(name, spec) {
    return h("button", {
        "type": "button",
        "class": `tool-button tool-${name}`,
        "title": spec.label,
        "data-tool": name,
        "data-behavior": spec.behavior,
    }, [
        h("svg", {
            "class": "tool-icon",
            "attr-width":  "100",
            "attr-height": "100",
            "attr-viewBox": "0 0 100 100",
        }, [
            h("path", { "attr-d": spec.d }),
        ]),
        h("p",
          { "class": "tool-label" },
          title_to_label_contents(spec.label)),
    ]);
}

function find_label(button) {
    return button.getElementsByClassName("tool-label")[0];
}

function title_to_label_contents(title) {
    let contents = [];
    for (let line of title.split(/\n+/)) {
        let tline = line.trim();
        if (tline === "")
            continue;
        if (contents.length > 0)
            contents.push(document.createElement("br"));
        contents.push(document.createTextNode(tline));
    }
    return contents;
}

// TODO: tooltips
// We can't use native popover=hint/interestfor for the tooltips
// because that's not yet fully baked, requiring a polyfill in firefox
// and having positioning issues in all browsers.

function gain_interest(button) {
    // TODO
}

function lose_interest(button) {
    // TODO
}

function clear_interests() {
    // TODO
}

// Local Variables:
// js2-additional-externs: ("CustomEvent")
// End:
