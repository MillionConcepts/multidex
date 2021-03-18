# deprecated spectrum ops as methods of spectra. perhaps
# will be desirable for something again in the future.

def filter_values(self, **scale_kwargs) -> dict:
    """
    dictionary of filter name: {wavelength, mean reflectance}. used to
    initialize most functions. to be replaced in some subclasses with
    whatever weird averaging and scaling rules might be desired
    """
    return {
        filt: {"wave": self.filters[filt], "mean": getattr(self, filt)}
        for filt in self.filters
        if getattr(self, filt)
    }


def intervening(
        self, wave_1: float, wave_2: float, **scale_kwargs
) -> List[Tuple[int, float]]:
    """
    wavelength, mean reflectance for all bands strictly between
    wave_1 & wave_2
    """
    return [
        (filt["wave"], filt["mean"])
        for filt in self.filter_values(**scale_kwargs).values()
        if (
                max([wave_1, wave_2]) > filt["wave"] > min([wave_1, wave_2])
                and filt["mean"] is not None  # TODO: Cruft?
        )
    ]


def band(
        self, wave_1: float, wave_2: float, **scale_kwargs
) -> List[Tuple[int, float]]:
    """
    wavelength, mean reflectance for all bands between and including
    wave_1 & wave_2
    """
    return [
        (filt["wave"], filt["mean"])
        for filt in self.filter_values(**scale_kwargs).values()
        if (
                max([wave_1, wave_2]) >= filt["wave"] >= min([wave_1, wave_2])
                and filt["mean"] is not None  # TODO: Cruft?
        )
    ]


def ref(self, filt_1: str, **scale_kwargs) -> float:
    """mean reflectance at filter given by filter_name"""
    return self.filter_values(**scale_kwargs)[filt_1]["mean"]


def band_avg(self, filt_1: str, filt_2: str, **scale_kwargs) -> float:
    """
    average of reflectance values at filt_1, filt_2, and all intervening
    bands. this currently double-counts measurements at matching
    wavelengths in cases where bands are not being virtually averaged.
    will cause issues if you ask for filters that aren't there when
    virtually averaging and also ask for things to be virtually averaged.
    """
    return stats.mean(
        [
            wave_reflectance_tuple[1]
            for wave_reflectance_tuple in self.band(
            self.all_filter_waves()[filt_1],
            self.all_filter_waves()[filt_2],
            **scale_kwargs
        )
        ]
    )


def band_max(self, filt_1: str, filt_2: str, **scale_kwargs) -> float:
    """
    max reflectance value between filt_1 and filt_2 (inclusive)
    """
    return max(
        [
            wave_reflectance_tuple[1]
            for wave_reflectance_tuple in self.band(
            self.all_filter_waves()[filt_1],
            self.all_filter_waves()[filt_2],
            **scale_kwargs
        )
        ]
    )


def band_min(self, filt_1: str, filt_2: str, **scale_kwargs) -> float:
    """
    min reflectance value between filt_1 and filt_2 (inclusive)
    """
    return min(
        [
            wave_reflectance_tuple[1]
            for wave_reflectance_tuple in self.band(
            self.all_filter_waves()[filt_1],
            self.all_filter_waves()[filt_2],
            **scale_kwargs
        )
        ]
    )


def ref_ratio(self, filt_1: str, filt_2: str, **scale_kwargs) -> float:
    """
    ratio of reflectance values at filt_1 & filt_2
    """
    filts = self.filter_values(**scale_kwargs)
    return filts[filt_1]["mean"] / filts[filt_2]["mean"]


def slope(self, filt_1: str, filt_2: str, **scale_kwargs) -> float:
    """
    where input strings are filter names defined in self.filters

    slope of line drawn between reflectance values at these two filters.
    passing filt_1 == filt_2 returns an error.
    do we allow 'backwards' lines? for now yes
    """
    if filt_1 == filt_2:
        raise ValueError(
            "slope between a wavelength and itself is undefined"
        )
    filts = self.filter_values(**scale_kwargs)
    ref_1 = filts[filt_1]["mean"]
    ref_2 = filts[filt_2]["mean"]
    wave_1 = filts[filt_1]["wave"]
    wave_2 = filts[filt_2]["wave"]
    return (ref_2 - ref_1) / (wave_2 - wave_1)


def band_depth_custom(
        self, filt_left: str, filt_right: str, filt_middle: str, **scale_kwargs
) -> float:
    """
    simple band depth at filt_middle --
    filt_middle reflectance / reflectance of 'continuum'
    (straight line between filt_left and filt_right) at filt_middle.
    passing filt_left == filt_right or filt_middle not strictly between
    them returns an error

    do we allow 'backwards' lines? for now yes (so 'left' and 'right'
    are misnomers)
    """
    if len({filt_left, filt_middle, filt_right}) != 3:
        raise ValueError(
            "band depth between a wavelength and itself is undefined"
        )
    filts = self.filter_values(**scale_kwargs)
    wave_left = filts[filt_left]["wave"]
    wave_middle = filts[filt_middle]["wave"]
    wave_right = filts[filt_right]["wave"]

    if not (
            max(wave_left, wave_right)
            > wave_middle
            > min(wave_left, wave_right)
    ):
        raise ValueError(
            "band depth can only be calculated at a band within the "
            "chosen range."
        )
    distance = wave_middle - wave_left
    slope = self.slope(filt_left, filt_right, **scale_kwargs)
    continuum_ref = filts[filt_left]["mean"] + slope * distance
    return filts[filt_middle]["mean"] / continuum_ref


def band_depth_min(
        self, filt_1: str, filt_2: str, **scale_kwargs
) -> float:
    """
    simple band depth at local minimum --
    local minimum reflectance / reflectance of 'continuum'
    (straight line between filt_1 and filt_2) at that wavelength.
    band depth between adjacent points is defined as 1.
    passing filt_1 == filt_2 returns an error.

    do we allow 'backwards' lines? for now yes
    """
    if filt_1 == filt_2:
        raise ValueError(
            "band depth between a wavelength and itself is undefined"
        )
    filts = self.filter_values(**scale_kwargs)
    wave_1 = filts[filt_1]["wave"]
    wave_2 = filts[filt_2]["wave"]

    intervening = self.intervening(wave_1, wave_2, **scale_kwargs)
    if not intervening:
        return 1

    min_ref = min([filt[1] for filt in intervening])
    min_wave = [
        (ix, wave)
        for ix, wave in enumerate([filt[0] for filt in intervening])
        if intervening[ix][1] == min_ref
    ][0]

    distance = min_wave[0] - wave_1
    slope = self.slope(filt_1, filt_2, **scale_kwargs)
    continuum_ref = filts[filt_1]["mean"] + slope * distance
    return min_ref / continuum_ref
