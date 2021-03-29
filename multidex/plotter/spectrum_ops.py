from numpy.linalg import norm
import pandas as pd


def filter_df_from_queryset(queryset, **scale_kwargs):
    filter_value_list = []
    id_list = []
    for spectrum in queryset:
        mean_dict = {
            filt: value["mean"]
            for filt, value in spectrum.filter_values(**scale_kwargs).items()
        }
        err_dict = {
            filt + "_err": value["err"]
            for filt, value in spectrum.filter_values(**scale_kwargs).items()
        }
        filter_value_list.append(mean_dict | err_dict)
        id_list.append(spectrum.id)
    return pd.DataFrame(filter_value_list, index=id_list)


def errfilter(dataframe, get_errors=False):
    return dataframe[
        [
            column
            for column in dataframe.columns
            if ("_err" in column) == get_errors
        ]
    ]


def intervening(filter_df, spec_model, wave_1, wave_2, errors=False):
    """
    wavelength, mean reflectance for all bands strictly between
    wave_1 & wave_2
    this is called by other functions in this module, not generally
    directly by user-facing functions.
    """
    mean_columns = [
        column for column in filter_df.columns if not column.endswith("_err")
    ]
    mean_df = filter_df[mean_columns].copy()
    band_df = mean_df[
        [
            column
            for column in mean_df.columns
            if max(wave_1, wave_2)
            > spec_model().all_filter_waves()[column]
            > min(wave_1, wave_2)
        ]
    ]
    if errors:
        error_df = filter_df[[column + "_err" for column in band_df.columns]]
        error_df.columns = [
            spec_model().all_filter_waves()[column] for column in
            band_df.columns
        ]
    else:
        error_df = None
    band_df.columns = [
        spec_model().all_filter_waves()[column] for column in band_df.columns
    ]
    return band_df, error_df


def band(filter_df, spec_model, wave_1, wave_2, errors=False):
    """
    wavelength, mean reflectance for all bands between and including
    wave_1 & wave_2
    this is called by other functions in this module, not generally
    directly by user-facing functions.
    """
    mean_columns = [
        column for column in filter_df.columns if not column.endswith("_err")
    ]
    mean_df = filter_df[mean_columns].copy()
    band_df = mean_df[
        [
            column
            for column in mean_df.columns
            if max(wave_1, wave_2)
            >= spec_model().all_filter_waves()[column]
            >= min(wave_1, wave_2)
        ]
    ]
    if errors:
        error_df = filter_df[[column + "_err" for column in band_df.columns]]
        error_df.columns = [
            spec_model().all_filter_waves()[column] for column in
            band_df.columns
        ]
    else:
        error_df = None
    band_df.columns = [
        spec_model().all_filter_waves()[column] for column in band_df.columns
    ]
    return band_df, error_df


def ref(filter_df, _spec_model, filt, errors=False):
    if errors:
        return filter_df[filt], filter_df[filt + "_err"]
    return filter_df[filt], None


def band_avg(filter_df, spec_model, filt_1, filt_2, errors=False):
    """
    average of reflectance values at filt_1, filt_2, and all intervening
    bands. this currently double-counts measurements at matching
    wavelengths in cases where bands are not being virtually averaged.
    will cause issues if you ask for filters that aren't there when
    virtually averaging and also ask for things to be virtually averaged.
    """
    filter_waves = spec_model().all_filter_waves()
    band_df, error_df = band(
        filter_df,
        spec_model,
        filter_waves[filt_1],
        filter_waves[filt_2],
        errors=errors,
    )
    if errors:
        return band_df.mean(axis=1), norm(error_df, axis=1)
    return band_df.mean(axis=1), None


def band_max(filter_df, spec_model, filt_1, filt_2, _):
    """
    max reflectance value between filt_1 and filt_2 (inclusive)
    note that error values aren't meaningful here so the request
    is ignored
    """
    filter_waves = spec_model().all_filter_waves()
    band_df, error_df = band(
        filter_df,
        spec_model,
        filter_waves[filt_1],
        filter_waves[filt_2],
        False,
    )
    return band_df.idxmax(axis=1), None


def band_min(filter_df, spec_model, filt_1, filt_2, _):
    """
    min reflectance value between filt_1 and filt_2 (inclusive)
    note that error values aren't meaningful here so the request
    is ignored
    """
    filter_waves = spec_model().all_filter_waves()
    band_df, error_df = band(
        filter_df,
        spec_model,
        filter_waves[filt_1],
        filter_waves[filt_2],
        False,
    )
    return band_df.idxmin(axis=1), None


def ratio(filter_df, _spec_model, filt_1, filt_2, errors=False):
    """
    ratio of reflectance values at filt_1 & filt_2
    """
    ratio_value = filter_df[filt_1] / filter_df[filt_2]
    # TODO: this is a weak approximation
    if errors:
        errs = filter_df[filt_1 + "_err"], filter_df[filt_2 + "_err"]
        return ratio_value, (errs[0] * errs[1] + errs[0]**2 + errs[1] + errs[0] + errs[1]**2)**0.5
    return ratio_value, None


def slope(filter_df, spec_model, filt_1, filt_2, errors=False):
    """
    slope of line drawn between reflectance values at these two filters.
    do we allow 'backwards' lines? for now yes
    """
    filter_waves = spec_model().all_filter_waves()
    difference = filter_df[filt_2] - filter_df[filt_1]
    distance = filter_waves[filt_2] - filter_waves[filt_1]
    slope_value = difference / distance
    if errors:
        errs = filter_df[[filt_1 + "_err", filt_2 + "_err"]]
        return slope_value, norm(errs, axis=1) / distance
    return slope_value, None


def band_depth_custom(
    filter_df,
    spec_model,
    filt_left: str,
    filt_right: str,
    filt_middle: str,
    errors=False,
):
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
    filter_waves = spec_model().all_filter_waves()
    wave_left = filter_waves[filt_left]
    wave_right = filter_waves[filt_right]
    wave_middle = filter_waves[filt_middle]
    if not (
        max(wave_left, wave_right) > wave_middle > min(wave_left, wave_right)
    ):
        raise ValueError(
            "band depth can only be calculated at a band within the "
            "chosen range."
        )
    distance = wave_middle - wave_left
    slope_series = slope(filter_df, spec_model, filt_left, filt_right)[0]
    continuum_ref = filter_df[filt_left] + slope_series * distance
    if errors:
        errs = filter_df[
            [filt + "_err" for filt in [filt_left, filt_right, filt_middle]]
        ]
        return (
            filter_df[filt_middle] / continuum_ref,
            norm(errs, axis=1) / continuum_ref,
        )


def band_depth_min(filter_df, spec_model, filt_1, filt_2, errors=False):
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
    filter_waves = spec_model().all_filter_waves()
    wave_1 = filter_waves[filt_1]
    wave_2 = filter_waves[filt_2]

    intervening_df, _ = intervening(filter_df, spec_model, wave_1, wave_2)

    min_ref = intervening_df.min(axis=1)
    min_wave = intervening_df.idxmin(axis=1)

    distance_series = min_wave - wave_1
    slope_series, _ = slope(filter_df, spec_model, filt_1, filt_2)
    continuum_ref = filter_df[filt_1] + slope_series * distance_series
    if errors:
        return (
            min_ref / continuum_ref,
            norm(filter_df[[filt_1 + "_err", filt_2 + "_err"]], axis=1)
            / continuum_ref,
        )
    else:
        return min_ref / continuum_ref, None
