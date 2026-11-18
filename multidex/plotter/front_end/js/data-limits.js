// slightly less typing and visual clutter
const MAX = Math.max;
const MIN = Math.min;

/** The limits of an abstract one-dimensional range.  */
export class Limits {
    constructor(a, b) {
        this.set(a, b);
    }

    /** Set this to define the limits [min(a, b), max(a, b)]. */
    set(a, b) {
        this.lo = MIN(a ?? 0, b ?? 0);
        this.hi = MAX(a ?? 0, b ?? 0);
    }

    /** Set this to a range that contains nothing.  Equivalent to
        calling set() with no arguments, but slightly more mnemonic.  */
    reset() {
        this.set();
    }

    /** Ensure VAL is within the limits defined by this.  */
    clamp(val) {
        return MIN(MAX(val, this.lo), this.hi);
    }

    /** Ensure both ends of [a, b] are within the limits defined by
        this, preserving the magnitude of RANGE as much as possible.  */
    clamp_range([a, b]) {
        let l_mag = this.hi - this.lo;

        let r_lo = MIN(a, b);
        let r_hi = MAX(a, b);
        let r_mag = r_hi - r_lo;

        // If the magnitude of RANGE is greater than or equal to the
        // magnitude of LIMITS, then the only way to bring both ends of
        // RANGE within LIMITS while preserving RANGE's magnitude as much
        // as possible ... is to return exactly LIMITS.
        //
        // Also return exactly LIMITS if RANGE is only a tiny bit smaller
        // than LIMITS, for better edge handling behavior on zooming outward.
        if (l_mag - r_mag < 0.01)
            return [this.lo, this.hi];

        // Otherwise, the magnitude of RANGE is less than the magnitude of
        // LIMITS, which means r_lo < l_lo and r_hi > l_hi cannot both
        // be true (they can, however, both be *false*), and also means
        // (l_lo + r_mag < l_hi) and (l_o < l_hi - r_mag) must both be true.
        if (r_lo < this.lo)
            return [this.lo, this.lo + r_mag];
        if (r_hi > this.hi)
            return [this.hi - r_mag, this.hi];
        return [r_lo, r_hi];
    }
}

/** The limits of a plot axis, in data coordinates. */
export class DataLimits extends Limits {
    /** Set this to the domain of the D3 linear scale SCALE. */
    set_from_domain(scale) {
        let [a, b] = scale.domain();
        this.set(a, b);
    }

    /** Set the domain of a D3 linear scale to be within the limits
     *  defined by this.
     */
    apply_to_domain(scale) {
        scale.domain([this.lo, this.hi]);
    }
}


/** The limits of a plot axis, in plot canvas coordinates. */
export class CanvasLimits extends Limits {
    /** Set this to the range of the D3 linear scale SCALE. */
    set_from_range(scale) {
        let [a, b] = scale.range();
        this.set(a, b);
    }

    /** Set the range of a D3 linear scale to be within the limits
     *  defined by this.
     */
    apply_to_range(scale) {
        scale.range([this.lo, this.hi]);
    }
}
