/** hscrpt.js: hyperscript's h() with fewer features and no backward
 *  compatibility gunk.  See end of file for licensing.
 */

// You actually do have to pick the correct XML namespace when creating
// SVG or MathML elements to embed them in an HTML document.

const SVG_NS = "http://www.w3.org/2000/svg";
const MML_NS = "http://www.w3.org/1998/Math/MathML";

// This table should list *only* elements that exist in either SVG or
// MathML, not also in HTML (e.g. <a> and <style> should not appear)
const TAG_NAMESPACES = Object.freeze({
    // SVG
    "animate":             SVG_NS,
    "animateMotion":       SVG_NS,
    "animateTransform":    SVG_NS,
    "circle":              SVG_NS,
    "clipPath":            SVG_NS,
    "defs":                SVG_NS,
    "desc":                SVG_NS,
    "ellipse":             SVG_NS,
    "feBlend":             SVG_NS,
    "feColorMatrix":       SVG_NS,
    "feComponentTransfer": SVG_NS,
    "feComposite":         SVG_NS,
    "feConvolveMatrix":    SVG_NS,
    "feDiffuseLighting":   SVG_NS,
    "feDisplacementMap":   SVG_NS,
    "feDistantLight":      SVG_NS,
    "feDropShadow":        SVG_NS,
    "feFlood":             SVG_NS,
    "feFuncA":             SVG_NS,
    "feFuncB":             SVG_NS,
    "feFuncG":             SVG_NS,
    "feFuncR":             SVG_NS,
    "feGaussianBlur":      SVG_NS,
    "feImage":             SVG_NS,
    "feMerge":             SVG_NS,
    "feMergeNode":         SVG_NS,
    "feMorphology":        SVG_NS,
    "feOffset":            SVG_NS,
    "fePointLight":        SVG_NS,
    "feSpecularLighting":  SVG_NS,
    "feSpotLight":         SVG_NS,
    "feTile":              SVG_NS,
    "feTurbulence":        SVG_NS,
    "filter":              SVG_NS,
    "foreignObject":       SVG_NS,
    "g":                   SVG_NS,
    "image":               SVG_NS,
    "line":                SVG_NS,
    "linearGradient":      SVG_NS,
    "marker":              SVG_NS,
    "mask":                SVG_NS,
    "metadata":            SVG_NS,
    "mpath":               SVG_NS,
    "path":                SVG_NS,
    "pattern":             SVG_NS,
    "polygon":             SVG_NS,
    "polyline":            SVG_NS,
    "radialGradient":      SVG_NS,
    "rect":                SVG_NS,
    "set":                 SVG_NS,
    "stop":                SVG_NS,
    "svg":                 SVG_NS,
    "switch":              SVG_NS,
    "symbol":              SVG_NS,
    "text":                SVG_NS,
    "textPath":            SVG_NS,
    "tspan":               SVG_NS,
    "use":                 SVG_NS,
    "view":                SVG_NS,

    // MathML
    "annotation-xml": MML_NS,
    "annotation":     MML_NS,
    "maction":        MML_NS,
    "math":           MML_NS,
    "menclose":       MML_NS,
    "merror":         MML_NS,
    "mfenced":        MML_NS,
    "mfrac":          MML_NS,
    "mi":             MML_NS,
    "mmultiscripts":  MML_NS,
    "mn":             MML_NS,
    "mo":             MML_NS,
    "mover":          MML_NS,
    "mpadded":        MML_NS,
    "mphantom":       MML_NS,
    "mprescripts":    MML_NS,
    "mroot":          MML_NS,
    "mrow":           MML_NS,
    "ms":             MML_NS,
    "mspace":         MML_NS,
    "msqrt":          MML_NS,
    "mstyle":         MML_NS,
    "msub":           MML_NS,
    "msubsup":        MML_NS,
    "msup":           MML_NS,
    "mtable":         MML_NS,
    "mtd":            MML_NS,
    "mtext":          MML_NS,
    "mtr":            MML_NS,
    "munder":         MML_NS,
    "munderover":     MML_NS,
    "semantics":      MML_NS,
});

/** Given the canonical "local name" of an HTML, SVG, or MathML tag,
 *  return the XML namespace for that tag.  Tags that exist in both
 *  HTML and one of the others are always given the HTML namespace.
 */
function namespace_for_tag(tag) {
    return TAG_NAMESPACES[tag] ?? "http://www.w3.org/1999/xhtml";
}

/** Return a new object containing all of the properties PROPS that were
 *  present in OBJ, with the same values.  OBJ is not altered, but the
 *  values will be shared between OBJ and the new object.
 */
function peek_properties(obj, props) {
    let rv = {};
    for (let prop of props) {
        if (Object.hasOwn(obj, prop)) {
            rv[prop] = obj[prop];
        }
    }
    return rv;
}

/** Combine all of the PROPS_ARGS, which should be key-value
 *  dictionaries (i.e. plain objects whose properties all have string
 *  keys) into a single key-value dict.  As a special case. if the very
 *  first argument is a string, that is shorthand for { "class": "..." }.
 *  Besides that special case, null/undefined arguments are ignored,
 *  all other non-object types throw an error.
 */
export function merge_props(...props_args) {
    let merged = {};
    if (typeof props_args[0] === "string") {
        let classes = props_args.shift();
        if (classes !== classes !== "")
            merged["class"] = classes;
    }
    for (let props of props_args) {
        if (props == null) continue;
        if (typeof props !== "object")
            throw new Error("bad argument to merge_props: not a dict");

        for (let [prop, val] of Object.entries(props)) {
            merged[prop] = val;
        }
    }
    return merged;
}

/** Break up ...ARGS as ["CLASSES"], [PROPS...], [CHILDREN...]
 *  as follows:
 *
 *  1. All arguments which are either null or undefined are discarded.
 *
 *  2. If the very first (remaining) argument is a string, it is
 *     "CLASSES".  If it is the *empty* string, it is discarded,
 *     otherwise it is rewritten as { "class": "CLASSES" } and becomes
 *     the first element of PROPS.
 *
 *  3. PROPS is the initial sequence of (remaining) arguments which
 *     are all plain objects (typeof === "object" but not a Node,
 *     Date, RegExp, or Array).
 *
 *  4. All remaining arguments are CHILDREN.
 *
 * Returns a 2-tuple whose first element is the result of applying
 * `merge_props` to PROPS (if there were no PROPS, this will be an
 * empty object) and whose second element is the list of CHILDREN.
 */
export function split_props_and_children(...args) {
    let props = [];
    let children = [];
    let maybe_classes = true;
    let maybe_props = true;
    for (let arg of args) {
        if (arg === null || arg === undefined) continue;
        if (maybe_classes) {
            maybe_classes = false;
            if (typeof arg === "string") {
                if (arg !== "") {
                    props.push({ "class": arg });
                }
                continue;
            }
        }
        // This conditional must be kept in sync with the "object" case of
        // add_children::add_child.
        if (maybe_props
            && typeof arg === "object"
            && !(arg instanceof Node
                 || arg instanceof Date
                 || arg instanceof RegExp
                 || Array.isArray(arg))) {
            props.push(arg);
        } else {
            maybe_props = false;
            children.push(arg);
        }
    }
    return [merge_props(...props), children];
}


/** Subroutine of h(): Annotate DOM element ELEMENT with NOTES.
 *  See h() for the rules for how NOTES is interpreted.
 */
function annotate(element, notes) {
    for (let [k, val] of Object.entries(notes)) {
        let [, prefix, key] = /^(on|data-|attr-|)(.*)$/.exec(k);
        // "on" sets element["on"], not an event listener for ""
        if (key === "") {
            key = prefix;
            prefix = "";
        }
        if (prefix === "on") {
            let listener = null;
            let options = {};
            if (val == null
                || typeof val === "function"
                || typeof val.handleEvent !== "function") {
                listener = val;
            } else {
                listener = val.listener;
                for (let [k, v] of Object.entries(val)) {
                    if (k === "listener")
                        continue;
                    options[k] = v;
                }
            }
            element.addEventListener(key, listener, options);
        } else if (prefix === "data-") {
            element.dataset[key] = val;
        } else if (prefix === "attr-") {
            element.setAttribute(key, val);
        } else if (key === "class" || key === "className") {
            // element.className is read-only on svg elements
            element.classList.value = val;
        } else if (key === "is" || key === "customElementRegistry") {
            // these are options to createElementNS and were already
            // handled in h(); do nothing here
        } else {
            // SVG seems to like making its reflective props be read-only
            try {
                element[key] = val;
            } catch (e) {
                if (e instanceof TypeError) {
                    element.setAttribute(key, val);
                } else {
                    throw e;
                }
            }
        }
    }
}

/** Subroutine of h(): Add all of the CHILDREN to ELEMENT.  Flatten nested
 *  arrays and iterables within CHILDREN; convert strings and stringifiable
 *  items to text nodes; merge adjacent text nodes.
 */
function add_children(element, children) {
    function bad_child_type(x) {
        let label = typeof x;
        if (label === "object") {
            if (x === null)
                label = "null";
            else if (typeof x[Symbol.toStringTag] === "string")
                label = x[Symbol.toStringTag];
            else if (typeof x.constructor.name === "string")
                label = x.constructor.name;
        }
        // as a children argument to h(), this was probably a dictionary
        // literal (meant to be notes, perhaps)
        if (label === "" || label === "object" || label === "Object")
            label = "dict";

        let a = "a";
        if (/^[aeiouAEIOU]/.test(label))
            a = "an";
        throw new Error(`add_child: can't use ${a} '${label}' as a child node`);
    }

    function merge_or_cat(node) {
        if (node instanceof Text && element.lastChild instanceof Text)
            element.lastChild.appendData(node.data);
        else
            element.appendChild(node);
    }

    function add_child(l) {
        switch (typeof l) {
        case "undefined":
            break;

        case "function":
            add_child(l());
            break;

        case "string":
            merge_or_cat(new Text(l));
            break;

        case "boolean":
        case "number":
        case "bigint":
        case "symbol":
            merge_or_cat(new Text(`${l}`));
            break;

        // The handling of this case must be kept in sync with the
        // if (typeof arg === "object"' && ...) condition in
        // split_props_and_children.
        case "object":
            if (l === null)
                break;
            if (l instanceof Node) {
                merge_or_cat(l);
                break;
            }
            if (l instanceof Date || l instanceof RegExp) {
                merge_or_cat(new Text(`${l}`));
                break;
            }
            if ((typeof l[Symbol.iterator] === "function"
                 && !(l instanceof Map))) {
                for (let c of l) {
                    add_child(c);
                }
                break;
            }
            /* FALLTHRU */
        default:
            bad_child_type(l);
        }
    }

    add_child(children);
}


/** h(tag, notes..., children...) -> HTMLElement
 *
 * Constructs an HTML element.
 *
 * @arg tag        Tag name of the element.  Unlike other HTML-in-JS notation
 *                 libraries, whatever you put here is passed *verbatim* to
 *                 document.createElementNS() as the qualifiedName.  Case is
 *                 not canonicalized.  Pseudo-CSS notation for classes and IDs
 *                 is not supported.
 *
 *                 The canonical *unprefixed* names of SVG and MathML tags
 *                 that aren't also HTML tags are recognized, and they will
 *                 be put in the matching namespace; all other tags will be
 *                 in the HTML namespace.  There is currently no way to
 *                 override this behavior.
 *
 * @arg notes      Annotations for the element.  For each leading argument
 *                 (after the tag) that is a {key:value} dictionary, we try
 *                 to set each key-value pair as a property-value pair on the
 *                 DOM element object.  If attempting to set a property
 *                 throws a TypeError, we try setting it as an HTML attribute
 *                 instead.  This *usually* does the Right Thing.  We do not
 *                 attempt to account for divergences between HTML attribute
 *                 names and DOM property names, except as noted below.  Five
 *                 kinds of keys are treated specially:
 *
 *                 1. The keys "is" and "customElementRegistry" are taken as
 *                    options for document.createElementNS.  They do not set
 *                    properties on the element after creation.
 *
 *                 2. The keys "class" and "className" actually set
 *                    element.classList.value.  (If you supply both,
 *                    which one wins is unspecified.)
 *
 *                 3. Keys beginning with "data-" set properties of
 *                    element.dataset, with the leading "data-" removed.
 *
 *                 4. Keys beginning with "on" are always translated to
 *                    calls to addEventListener, whether or not a legacy
 *                    "on<event>" property exists.  The value may be either
 *                    an event listener callback, or a dictionary; dicts are
 *                    interpreted as options for addEventListener, with the
 *                    special key "listener" giving the actual listener
 *                    callback.
 *
 *                 5. Keys beginning with "attr-" skip the attempt to set a
 *                    DOM property and unconditionally set an HTML attribute,
 *                    with the leading "attr-" removed; e.g. "href" will first
 *                    try to set a property named "href", falling back to
 *                    setAttribute if that fails, but "attr-href" will only
 *                    call element.setAttribute("href", value).
 *
 *                 As an extra special case, if the very first note is a bare
 *                 string, that is shorthand for { "class": "..." }.  If you
 *                 wanted the string to be made into a text node, wrap it in
 *                 a one-element array: h("p", "star-bellied") becomes
 *                 `<p class="star-bellied"></p>`, but h("p", ["star-bellied"])
 *                 becomes `<p>star-bellied</p>`.
 *
 * @arg children   All subsequent arguments provide children for the element
 *                 being constructed.  Children may be a mix of existing DOM
 *                 nodes (including the results of nested calls to h()),
 *                 things that can reasonably be converted to text nodes
 *                 (e.g. strings, numbers, and Date objects), and iterables
 *                 of these things.  Callables will be called with no
 *                 arguments and the result used.  Iterables will be
 *                 recursively flattened, and adjacent text nodes will be
 *                 concatenated: h("div", ["a", "b"], "c") and h("div", "abc")
 *                 both produce `<div>abc</div>`.
 */
export function h(tag, ...rest) {
    if (typeof tag !== "string") {
        throw new TypeError(
            `'tag' argument to h() must be a string, not '${tag}'`
        );
    }

    let [notes, children] = split_props_and_children(...rest);

    // The list of properties here must be kept in sync with the list of "we
    // already handled this" properties in annotate().
    let ce_options = peek_properties(notes, ["is", "customElementRegistry"]);
    let e = document.createElementNS(namespace_for_tag(tag), tag, ce_options);
    annotate(e, notes);
    add_children(e, children);
    return e;
}

export default h;

// Copyright 2026 Million Concepts LLC.
// Based on hyperscript, copyright 2012 Dominic Tarr.
//
// Permission is hereby granted, free of charge, to any person
// obtaining a copy of this software and associated documentation
// files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use, copy,
// modify, merge, publish, distribute, sublicense, and/or sell copies
// of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
