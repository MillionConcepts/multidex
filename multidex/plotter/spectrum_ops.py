"""
functions for calculations on spectral features.
includes preprocessing and wrapping for marslab.spectops
"""
import math
from itertools import product
from statistics import mean
from typing import Union

import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation
from sklearn.decomposition import PCA

from marslab import spectops
from marslab.compat.xcam import INSTRUMENT_UNCERTAINTIES

def d2r(
    degrees: Union[float, np.ndarray, pd.Series]
) -> Union[float, np.ndarray, pd.Series]:
    """degrees to radians"""
    return math.pi * degrees / 180


def r2d(
    radians: Union[float, np.ndarray, pd.Series]
) -> Union[float, np.ndarray, pd.Series]:
    """radians to degrees"""
    return 180 * radians / math.pi


def compute_minmax_spec_error(filter_df, spec_model, spec_op, *filters):
    """
    crude bounds for the hull of the range of possible measurements
    for a given spectrum operation with a given instrument / filter df
    serves as a wrapper function for others in this module
    """
    # cartesian product of these sets gives all possible sign combos for
    # error high, error low, i.e.,
    if spec_op.__name__ in ["band_min", "band_max"]:
        return spec_op(filter_df, spec_model, *filters)[0], None
    unc = INSTRUMENT_UNCERTAINTIES[spec_model.instrument]
    corners = product(*[[1, -1] for _ in filters])
    bounds_df_list = []
    # apply these signs to uncertainty values, getting a list of dataframes
    # giving values of all measurements in set at the upper / lower bound
    # combinations for uncertainties associated with each relevant filter.
    for corner in corners:
        corner_series_list = []
        for filt_ix, sign in enumerate(corner):
            filt = filters[filt_ix]
            # are the filters merged? averaging the values is probably wrong
            # but it's all I've got atm
            if "_" in filt:
                filt_1, filt_2 = filt.split("_")
                filt_unc = mean((unc[filt_1], unc[filt_2]))
            else:
                filt_unc = unc[filt]
            corner_series_list.append(
                filter_df[filt]
                + filter_df[filt] * corner[filt_ix] * filt_unc / 100
            )
        corner_df = pd.concat(corner_series_list, axis=1)
        # record the value of the spectrum op for each of these bounding
        # dataframes.
        bounds_df_list.append(spec_op(corner_df, spec_model, *filters)[0])
    # compute the nominal value and compare it to values at these bounds
    possible_values = pd.concat(bounds_df_list, axis=1)
    nominal_value = spec_op(filter_df, spec_model, *filters)[0]
    offsets = possible_values.sub(nominal_value, axis=0)
    # then min / max of each of these gives us an error estimate for each
    # spectrum
    return nominal_value, (offsets.min(axis=1), offsets.max(axis=1))


def _build_base_df(average_filters, queryset, scale_to):
    filter_value_list, id_list = [], []
    for spectrum in queryset:
        mean_dict = {
            filt: value["mean"]
            for filt, value in spectrum.filter_values(
                average_filters=average_filters, scale_to=scale_to
            ).items()
        }
        err_dict = {
            filt + "_std": value["std"]
            for filt, value in spectrum.filter_values(
                average_filters=average_filters, scale_to=scale_to
            ).items()
        }
        filter_value_list.append(mean_dict | err_dict)
        id_list.append(spectrum.id)

    return pd.DataFrame(filter_value_list, index=id_list)


def data_df_from_queryset(
    queryset, spec_model, r_star=True, average_filters=False, scale_to=None
):
    filter_df = _build_base_df(average_filters, queryset, scale_to)
    filter_df["max_wrasd"] = make_roughness_metric(filter_df, spec_model)
    # TODO: I'm not actually sure this should be happening here. Assess whether
    #   it's preferable to have rules for this on models.
    if r_star is True:
        theta_i = np.cos(
            d2r(pd.Series([spec.incidence_angle for spec in queryset]))
        )
        for column in filter_df.columns:
            filter_df[column] = filter_df[column] / theta_i
    filter_avg = filter_df[
        [c for c in filter_df.columns if "std" not in c]
    ].mean(axis=1)
    filter_df["filter_avg"] = np.round(filter_avg, 5)
    filter_df["std_avg"] = np.round(
        filter_df[[c for c in filter_df.columns if "std" in c]].mean(axis=1), 5
    )
    filter_df["rel_std_avg"] = filter_df["std_avg"] / filter_df["filter_avg"]
    if queryset.model.__name__ == 'ZSpec':
        for eye in ('L', 'R'):
            narrowband = [
                f for f in queryset.model.filters
                if f.startswith(eye) and '0' not in f
            ]
            narrow = filter_df[narrowband]
            mad = median_abs_deviation(narrow, axis=1)
            quartile_high = np.percentile(narrow, 75, axis=1)
            quartile_low = np.percentile(narrow, 25, axis=1)
            qscale = (quartile_high - quartile_low) / 2
            filter_df[f'{eye.lower()}_rmad'] = mad / qscale
            rel_std = narrow.std(axis=1) / narrow.mean(axis=1)
            filter_df[f'{eye.lower()}_rstd'] = rel_std
    else:
        filter_df['l_rmad'] = None
        filter_df['r_rmad'] = None
        filter_df['l_rstd'] = None
        filter_df['r_rstd'] = None
    return filter_df


def make_roughness_metric(spike_df, spec_model):
    spike_df = spike_df.copy()
    if "permissibly_explanatory_bandpasses" in dir(spec_model):
        spike_df = spike_df[
            spec_model().permissibly_explanatory_bandpasses(spike_df.columns)
        ]
    spike_df = spike_df[[f for f in spike_df.columns if "_std" not in f]]
    scols = sorted(
        spike_df.columns, key=lambda f: spec_model().all_filter_waves()[f]
    )
    sdiff = spike_df[scols].diff(axis=1).replace(np.nan, 0)
    srev = (np.sign(sdiff).diff(axis=1).abs() == 2) + 1
    max_wrasd = (sdiff.abs() * srev).max(axis=1) / spike_df[scols].replace(np.nan, 0).mean(axis=1)
    return max_wrasd


def intervening(filter_df, spec_model, wave_1, wave_2, errors=False):
    """
    wavelength, mean reflectance for all bands strictly between
    wave_1 & wave_2
    this is called by other functions in this module, not generally
    directly by user-facing functions.
    """
    mean_columns = [
        column for column in filter_df.columns if not column.endswith("_std")
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
        error_df = filter_df[[column + "_std" for column in band_df.columns]]
        error_df.columns = [
            spec_model().all_filter_waves()[column]
            for column in band_df.columns
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
        column for column in filter_df.columns if not column.endswith("_std")
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
        error_df = filter_df[[column + "_std" for column in band_df.columns]]
        error_df.columns = [
            spec_model().all_filter_waves()[column]
            for column in band_df.columns
        ]
    else:
        error_df = None
    band_df.columns = [
        spec_model().all_filter_waves()[column] for column in band_df.columns
    ]
    return band_df, error_df


def ref(filter_df, _spec_model, filt, errors=False):
    if errors:
        return filter_df[filt], filter_df[filt + "_std"]
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
    if error_df is not None:
        error_df = error_df.T
    return spectops.band_avg(band_df.T, error_df, None)


def band_max(filter_df, spec_model, filt_1, filt_2, _errors=False):
    """
    max reflectance value between filt_1 and filt_2 (inclusive)
    note that error values aren't meaningful here so the request
    is ignored
    """
    filter_waves = spec_model().all_filter_waves()
    band_df, _error_df = band(
        filter_df,
        spec_model,
        filter_waves[filt_1],
        filter_waves[filt_2],
        False,
    )
    return spectops.band_max(band_df.T, None, band_df.columns)


def band_min(filter_df, spec_model, filt_1, filt_2, _errors=False):
    """
    min reflectance value between filt_1 and filt_2 (inclusive)
    note that error values aren't meaningful here so the request
    is ignored
    """
    filter_waves = spec_model().all_filter_waves()
    band_df, _error_df = band(
        filter_df,
        spec_model,
        filter_waves[filt_1],
        filter_waves[filt_2],
        False,
    )
    return spectops.band_min(band_df.T, None, band_df.columns)


def ratio(filter_df, _spec_model, filt_1, filt_2, errors=False):
    """
    ratio of reflectance values at filt_1 & filt_2
    """
    band_df = filter_df[[filt_1, filt_2]].T
    if errors:
        error_df = filter_df[[filt_1 + "_std", filt_2 + "_std"]].T
    else:
        error_df = None
    return spectops.ratio(band_df, error_df, None)


def slope(filter_df, spec_model, filt_1, filt_2, errors=False):
    """
    slope of line drawn between reflectance values at these two filters.
    do we allow 'backwards' lines? for now yes
    """
    band_df = filter_df[[filt_1, filt_2]].T
    if errors:
        error_df = filter_df[[filt_1 + "_std", filt_2 + "_std"]].T
    else:
        error_df = None
    filter_waves = spec_model().all_filter_waves()
    wavelengths = (filter_waves[filt_1], filter_waves[filt_2])
    return spectops.slope(band_df, error_df, wavelengths)


def band_depth(
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
    band_df = filter_df[[filt_left, filt_right, filt_middle]].T
    if errors:
        error_df = filter_df[
            [filt_left + "_std", filt_right + "_std", filt_middle + "_std"]
        ].T
    else:
        error_df = None
    filter_waves = spec_model().all_filter_waves()
    wavelengths = (
        filter_waves[filt_left],
        filter_waves[filt_right],
        filter_waves[filt_middle],
    )
    return spectops.band_depth(band_df, error_df, wavelengths)


def pca_means(filter_df, n_components=4):
    if filter_df.isna().any():
        raise ValueError("do not use pca_means on arrays containing NaN")
    pca = PCA(n_components=n_components)
    vectors = filter_df.T.to_dict("list")
    vectarray = np.array(tuple(vectors.values()))
    return pca.fit_transform(vectarray)


# TODO: can this be cached? this might not be practical b/c of scaling, or at
#  least might have to be reset whenever scales change. it's pretty fast,
#  anyway.
