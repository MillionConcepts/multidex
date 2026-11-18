/** Walk the object property graph rooted at SUBJECT, depth first,
 *  bottom up; call CALLBACK upon each property's value, working from
 *  leaves upward to the root.  CALLBACK will be called only once for
 *  each node, even if it has multiple in-edges.  Returns SUBJECT.
 *
 *  Loosely based on code samples from
 *  https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Object/freeze#deep_freezing
 *  https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/WeakSet#detecting_circular_references
 */
export function walk_property_graph(subject, callback) {
    let processed = new WeakSet();
    let recursor = (subj) => {
        if (typeof subj === "object" && subj !== null) {
            if (processed.has(subj))
                return;
            processed.add(subj);
            for (let val of Object.values(subj))
                recursor(val);
        }
        callback(subj);
    };
    recursor(subject);
    return subject;
}

/** Recursively apply Object.freeze to the object graph rooted at OBJ.  */
export function deepfreeze(obj) {
    return walk_property_graph(obj, Object.freeze);
}

/** Set each property of ATTRS (a key-value dictionary) as a DOM
 *  attribute (*not* a DOM property) of ELEMENT.
 */
export function set_attrs(element, attrs) {
    for (let [attr, val] of Object.entries(attrs)) {
        element.setAttribute(attr, val);
    }
}

/** Throw an error because an abstract method named NAME was called
 *  (i.e. some subclass neglected to override something it should have).
 */
export function abstract_method(name) {
    throw new TypeError(`abstract method ${name} was called`);
}

/** If ITEM is neither null nor undefined, append it to ARRAY.
 *  Otherwise do nothing.
 */
export function push_nonnull(array, item) {
    if (item != null)
        array.push(item);
}

/** Fetch URL and interpret the response as JSON. */
export async function fetch_json(url) {
    let resp = await fetch(url);
    if (!resp.ok) {
        throw new Error(`${resp.url}: ${resp.status} ${resp.statusText}`);
    }
    return await resp.json();
}
