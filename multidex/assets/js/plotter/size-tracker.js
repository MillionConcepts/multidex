/** A `SizeTracker` monitors the size of one or more DOM elements and sends
 *  them events when they are resized.  I don't understand why browser devs
 *  invented this "ResizeObserver" API; it would have been simpler and more
 *  logical to allow resize event listeners on elements as well as "window".
 *  Per https://developer.mozilla.org/en-US/docs/Web/API/Window/resize_event
 *  it's actually worse than that, resize event listeners *used to*
 *  work on any element and they decided that was too straightforward.
 *  At least you can manually dispatch a "resize" event to any element.
 *
 *  Possible future improvement: throttle frequency of events using
 *  requestAnimationFrame.  But then we have to merge sets of resized
 *  elements, and ResizeObserver might already be handling that for us.
 */
export class SizeTracker {
    #observer = null;
    #onResized(entries) {
        for (let e of entries) {
            e.target.dispatchEvent(new Event("resize"));
        }
    }
    constructor() {
        this.#observer = new window.ResizeObserver(
            this.#onResized.bind(this)
        );
    }
    track(element) {
        this.#observer.observe(element);
    }
    forget(element) {
        this.#observer.unobserve(element);
    }
}

/** A global SizeTracker object; only one per window is needed.  */
export const SIZE_TRACKER = new SizeTracker();
