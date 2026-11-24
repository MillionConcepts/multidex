import h from "hscrpt";
import { fetch_json } from "js-utilz";

/** The lower right hand pane of the application, which, if we can, shows
 *  browse images of the selected observation.
 */
export class BrowseImageViewer {
    #container;
    #image_files;
    #pending_image_files;
    #pending_selection;

    constructor(container, colspecs) {
        this.#image_files = [];
        this.#container = container;
        this.#container.addEventListener(
            "selectObs",
            (event) => this.select_obs(event.detail)
        );

        let image_cols = Object.entries(colspecs).flatMap(([name, spec]) =>
            (!spec.meta && spec.type === "image") ? [name] : []
        );
        if (image_cols.length === 0) {
            this.#pending_image_files = false;
            this.#container.replaceChildren(
                h("p", {"class": "no-image", }, [
                    "Browse images", h("br"), "not available"
                ])
            );
            return;
        }

        this.#pending_image_files = true;
        let query = new URLSearchParams();
        query.append("columns", Object.values(image_cols).join(","));
        fetch_json(`/data?${query}`)
            .then(this.#on_image_files_available.bind(this))
            .catch((err) => {
                this.#pending_image_files = false;
                this.#pending_selection = null;
                this.#container.replaceChildren(
                    h("p", {"class": "no-image", }, [
                        "Browse images", h("br"),
                        "not available", h("br"),
                        `(${err})`
                    ]));
            });
    }

    #on_image_files_available(blob) {
        // the blob will have the form
        // { "image_XXX": [...], "image_YYY": [...], "id": [...] }
        // turn this into { id1: [ ["XXX", "..."], ["YYY", "..."] ], ... }
        // for efficient lookup later
        let image_files = [];
        let eyes = Object.keys(blob)
            .filter((k) => k != "id")
            .map((k) => [k.replace(/^image_/, ""), k]);

        for (let [i, id] of blob.id.entries()) {
            image_files[id] =
                eyes.map(([eye, col]) => [eye, blob[col]?.[i] ?? ""]);
        }
        this.#image_files = image_files;
        this.#pending_image_files = false;

        // create a "holder" div for each eye
        let base_id = this.#container.id;
        let ordered_eyes = eyes.map(([eye, col]) => eye);
        // this happens to do the right thing for ["left", "right"]
        ordered_eyes.sort();
        this.#container.replaceChildren(...ordered_eyes.map((eye) => h(
            "div", { "class": "browse-image-holder",
                     "id": `${base_id}-${eye}` }
        )));

        if (this.#pending_selection != null) {
            let ids = this.#pending_selection;
            this.#pending_selection = null;
            this.select_obs(ids);
        }
    }

    /** The observations with ids IDS have been selected; display
     *  their browse images, if we can.  This can be called directly,
     *  or by firing a custom event "selectObs" at the container
     *  element with detail equal to the desired IDS.  IDS can be a
     *  list or a single value.
     *
     *  To clear the browse images again, call this function with an
     *  empty array for IDS.
     */
    select_obs(ids) {
        if (Array.isArray(ids) && ids.length === 0) {
            // display nothing
            this.#pending_selection = null;
            for (let holder of this.#container.children) {
                if (holder.classList.contains("browse-image-holder"))
                    holder.replaceChildren();
            }
            return;
        }

        // If we're currently waiting for the file list, just stash
        // this request and return; we'll be called back when the file
        // list becomes available
        if (this.#pending_image_files) {
            this.#pending_selection = ids;
            return;
        }

        // If we know for sure we can't display browse images,
        // drop this request.
        if ((this.#image_files?.length ?? 0) == 0) {
            return;
        }

        // TODO: some sort of carousel for multiple selection
        if (Array.isArray(ids)) {
            if (ids.length !== 1) {
                console.error(`BrowseImageViewer: simultaneous viewing of ${ids.length} observations is not implemented`);
                return;
            }
            ids = ids[0];
        }

        for (let [eye, file] of this.#image_files[+ids] ?? []) {
            let holder = document.getElementById(
                `${this.#container.id}-${eye}`
            );
            if (holder == null) {
                console.error("no holder found for eye: ",
                              this.#container.id, eye);
                continue;
            }
            holder.replaceChildren();
            if (file != null && file !== "") {
                let img = h("img", {
                    "class": "browse-image",
                    "src": `/browse/${file}`,
                });
                img.decode()
                    .then(() => holder.appendChild(img))
                    .catch((e) => holder.appendChild(h(
                        "p", { "class": "load-error" },
                        [ `Error loading /browse/${file}: `,
                          h("br"),
                          `${e}` ]
                    )));
            } else {
                holder.appendChild(h(
                    "p", { "class": "no-image", },
                    [ `No ${eye} image` ]
                ));
            }
        }
    }
}

export default BrowseImageViewer;
