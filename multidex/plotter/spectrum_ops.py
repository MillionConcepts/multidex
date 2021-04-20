"""
functions for calculations on spectral features.
includes preprocessing and wrapping for marslab.spectops
"""

import math
from itertools import product
from typing import Union

import numpy as np
import pandas as pd

from marslab.compat.xcam import INSTRUMENT_UNCERTAINTIES
from marslab import spectops
from plotter_utils import qlist


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
            corner_series_list.append(
                filter_df[filt]
                + filter_df[filt] * corner[filt_ix] * unc[filt] / 100
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


def filter_df_from_queryset(
    queryset,
    r_star=True,
    average_filters=False,
    scale_to=None,
):
    filter_value_list = []
    id_list = []
    for spectrum in queryset:
        mean_dict = {
            filt: value["mean"]
            for filt, value in spectrum.filter_values(
                average_filters=average_filters, scale_to=scale_to
            ).items()
        }
        err_dict = {
            filt + "_err": value["err"]
            for filt, value in spectrum.filter_values(
                average_filters=average_filters, scale_to=scale_to
            ).items()
        }
        filter_value_list.append(mean_dict | err_dict)
        id_list.append(spectrum.id)
    filter_df = pd.DataFrame(filter_value_list)
    # TODO: I'm not actually sure this should be happening here. Assess whether
    #  it's preferable to have rules for this on models.
    if r_star:
        theta_i = np.cos(d2r(pd.Series(qlist(queryset, "incidence_angle"))))
        for column in filter_df.columns:
            filter_df[column] = filter_df[column] / theta_i
    filter_df.index = id_list
    return filter_df


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
            spec_model().all_filter_waves()[column]
            for column in band_df.cols
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
    return spectops.band_avg(band_df, error_df, None)


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
    return spectops.band_max(band_df, error_df, band_df.columns)


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
    return spectops.band_min(band_df, error_df, band_df.columns)


def ratio(filter_df, _spec_model, filt_1, filt_2, errors=False):
    """
    ratio of reflectance values at filt_1 & filt_2
    """
    band_df = filter_df[[filt_1, filt_2]]
    if errors:
        error_df = filter_df[[filt_1+"_err", filt_2+"_err"]]
    else:
        error_df = None
    return spectops.ratio(band_df, error_df, None)


def slope(filter_df, spec_model, filt_1, filt_2, errors=False):
    """
    slope of line drawn between reflectance values at these two filters.
    do we allow 'backwards' lines? for now yes
    """
    band_df = filter_df[[filt_1, filt_2]]
    if errors:
        error_df = filter_df[[filt_1+"_err", filt_2+"_err"]]
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
    band_df = filter_df[[filt_left, filt_right, filt_middle]]
    if errors:
        error_df = filter_df[
            [filt_left + "_err", filt_right + "_err", filt_middle + "_err"]
        ]
    else:
        error_df = None
    filter_waves = spec_model().all_filter_waves()
    wavelengths = (
        filter_waves[filt_left],
        filter_waves[filt_right],
        filter_waves[filt_middle]
    )
    return spectops.band_depth(band_df, error_df, wavelengths)

# currently deprecated

# def band_depth_min(filter_df, spec_model, filt_1, filt_2, errors=False):
#     """
#     simple band depth at local minimum --
#     local minimum reflectance / reflectance of 'continuum'
#     (straight line between filt_1 and filt_2) at that wavelength.
#     band depth between adjacent points is defined as 1.
#     passing filt_1 == filt_2 returns an error.
#
#     do we allow 'backwards' lines? for now yes
#     """
#     if filt_1 == filt_2:
#         raise ValueError(
#             "band depth between a wavelength and itself is undefined"
#         )
#     filter_waves = spec_model().all_filter_waves()
#     wave_1 = filter_waves[filt_1]
#     wave_2 = filter_waves[filt_2]
#
#     intervening_df, _ = intervening(filter_df, spec_model, wave_1, wave_2)
#
#     min_ref = intervening_df.min(axis=1)
#     min_wave = intervening_df.idxmin(axis=1)
#
#     distance_series = min_wave - wave_1
#     slope_series, _ = slope(filter_df, spec_model, filt_1, filt_2)
#     continuum_ref = filter_df[filt_1] + slope_series * distance_series
#     band_depth = 1 - min_ref / continuum_ref
#     if errors:
#         # TODO: this is hugely cheating -- adding in quadrature with
#         #  gaussian assumptions and ignoring the contribution of the
#         #  center filter
#         return (
#             band_depth,
#             norm(filter_df[[filt_1 + "_err", filt_2 + "_err"]], axis=1)
#             / continuum_ref,
#         )
#     else:
#         return band_depth, None
