from math import floor
from statistics import mean
from itertools import chain, combinations
from collections.abc import Mapping, Sequence
from typing import Optional

"""
constants and functions intended specifically for compatibility with
the behavior of applications outside of this project _other than_
data / metadata format converters.
"""

# mappings between MERspect color designations and hex codes
MERSPECT_M20_COLOR_MAPPINGS = {
    "green": "#00ff3b",  # not a creative color
    "yellow": "#eeff00",
    "blue": "#1500ff",
    "red": "#ff0000",
    "magenta": "#ff00ae",
    "cyan": "#00ffee",
    "orange": "#ffa100",
    "azure": "#0094ff",
    "purple": "#ad0bbf",
    "lime": "#9bec11",
    "rust": "#dd5b10",
    "green+2": "#7fff9d",
    "green-1": "#00b229",
    "green-2": "#007f1d",
    "yellow-2": "#777f00",
    "blue+2": "#8a7fff",
    "blue-1": "#0e00b2",
    "blue-2": "#0a007f",
    "red+2": "#ff7f7f",
    "red-1": "#b20000",
    "red-2": "#7f0000",
    "magenta+2": "#ff7fd6",
    "magenta+1": "#ff4cc6",
    "magenta-1": "#b20079",
    "magenta-2": "#7f0057",
    "magenta-3": "#4c0034",
    "cyan+2": "#a6fff9",
    "cyan+1": "#73fff6",
    "cyan-1": "#00b2a6",
    "cyan-2": "#007f77",
    "cyan-3": "#004c47",
    "orange+2": "#ffd07f",
    "orange+1": "#ffbd4c",
    "orange-1": "#b27100",
    "orange-2": "#7f5000",
    "orange-3": "#4c3000",
    "azure+2": "#7fc9ff",
    "azure+1": "#4cb4ff",
    "azure-1": "#0068b2",
    "azure-2": "#004a7f",
    "azure-3": "#002c4c",
}
# Everything below here is from the MCAM / PCAM legacy functionality

MERSPECT_MSL_COLOR_MAPPINGS = {
    "red": "#dc143c",  # duplicated from above w/ different hex code
    "light green": "#7fff00",
    "light blue": "#0000ff",
    "light cyan": "#00ffff",
    "dark green": "#128012",
    "yellow": "#ffff00",  # duplicated from above w/ different hex code
    "light purple": "#ff00ff",
    "pink": "#fa8072",
    "teal": "#008080",
    "goldenrod": "#B8860b",
    "sienna": "#a0522d",
    "dark blue": "#000080",
    "bright red": "#ff0000",
    "dark red": "#800000",
    "dark purple": "#800080",
}

MERSPECT_COLOR_MAPPINGS = {
    **MERSPECT_M20_COLOR_MAPPINGS,
    **MERSPECT_MSL_COLOR_MAPPINGS,
}

"""
mappings between filter characteristic wavelengths and designations,
along with a bunch of derived related values for MCAM-family ("XCAM") 
instruments, affording consistent interpretation of operations on individual 
spectra
"""

WAVELENGTH_TO_FILTER = {
    "ZCAM": {
        "L": {
            630: "L0R",
            544: "L0G",
            480: "L0B",
            800: "L1",
            754: "L2",
            677: "L3",
            605: "L4",
            528: "L5",
            442: "L6",
        },
        "R": {
            631: "R0R",
            544: "R0G",
            480: "R0B",
            800: "R1",
            866: "R2",
            910: "R3",
            939: "R4",
            978: "R5",
            1022: "R6",
        },
    },
    "MCAM": {
        "L": {
            482: "L0B",  #
            493: "L0B",  # Accepted value of L0B has changed over time
            495: "L0B",  #
            554: "L0G",
            640: "L0R",
            527: "L1",
            445: "L2",
            751: "L3",
            676: "L4",
            867: "L5",
            1012: "L6",
        },
        "R": {
            482: "R0B",  #
            493: "R0B",  # Accepted value of R0B has changed over time
            495: "R0B",  #
            551: "R0G",
            638: "R0R",
            527: "R1",
            447: "R2",  #
            805: "R3",
            908: "R4",
            937: "R5",
            1013: "R6",  #
        },
    },
}


# rules currently in use:
# set of virtual filters === the set of pairs of real filters with nominal
# band centers within 5 nm of one another
# the virtual mean reflectance in an ROI for a virtual filter is the
# arithmetic mean of the mean reflectance values in that ROI for the two real
# filters in its associated pair.
# the nominal band center of a virtual filter is the arithmetic mean of the
# nominal band centers of the two real filters in its associated pair.


def make_xcam_filter_dict(abbreviation):
    """
    form filter: wavelength dictionary for mastcam-family instruments
    """
    left = {
        name: wavelength
        for wavelength, name in WAVELENGTH_TO_FILTER[abbreviation]["L"].items()
    }
    right = {
        name: wavelength
        for wavelength, name in WAVELENGTH_TO_FILTER[abbreviation]["R"].items()
    }
    return {
        name: wavelength
        for name, wavelength in sorted(
            {**left, **right}.items(), key=lambda item: item[1]
        )
    }


def make_xcam_filter_pairs(abbreviation):
    """
    form list of pairs of close filters for mastcam-family instruments
    """
    filter_dict = make_xcam_filter_dict(abbreviation)
    return tuple(
        [
            (filter_1, filter_2)
            for filter_1, filter_2 in combinations(filter_dict, 2)
            if abs(filter_dict[filter_1] - filter_dict[filter_2]) <= 5
        ]
    )


def make_virtual_filters(abbreviation):
    """
    form mapping from close filter names to wavelengths for mastcam-family
    """
    filter_dict = make_xcam_filter_dict(abbreviation)
    filter_pairs = make_xcam_filter_pairs(abbreviation)
    return {
        pair[0]
        + "_"
        + pair[1]: floor(mean([filter_dict[pair[0]], filter_dict[pair[1]]]))
        for pair in filter_pairs
    }


def make_virtual_filter_mapping(abbreviation):
    """
    form mapping from close filter names to filter pairs for mastcam-family
    """
    return {
        pair[0] + "_" + pair[1]: pair
        for pair in make_xcam_filter_pairs(abbreviation)
    }


def make_canonical_averaged_filters(abbreviation):
    filter_dict = make_xcam_filter_dict(abbreviation)
    virtual_filters = make_virtual_filters(abbreviation)
    virtual_filter_mapping = make_virtual_filter_mapping(abbreviation)
    retained_filters = {
        filt: filter_dict[filt]
        for filt in filter_dict
        if filt not in chain.from_iterable(virtual_filter_mapping.values())
    }
    caf = {**retained_filters, **virtual_filters}
    return {filt: caf[filt] for filt in sorted(caf, key=lambda x: caf[x])}


XCAM_ABBREVIATIONS = ["MCAM", "ZCAM"]
DERIVED_CAM_DICT = {
    abbrev: {
        "filters": make_xcam_filter_dict(abbrev),
        "virtual_filters": make_virtual_filters(abbrev),
        "virtual_filter_mapping": make_virtual_filter_mapping(abbrev),
        "canonical_averaged_filters": make_canonical_averaged_filters(abbrev),
    }
    for abbrev in XCAM_ABBREVIATIONS
}


def polish_xcam_spectrum(
    spectrum: Mapping[str, float],
    cam_info: Mapping[str, dict],
    scale_to: Optional[Sequence[str, str]] = None,
    average_filters: bool = True,
):
    """
    scale and merge values of a spectrum according to MERSPECT-style rules
    scale_to: None or tuple of (lefteye filter name, righteye filter name)
    """
    values = {}
    lefteye_scale = 1
    righteye_scale = 1
    # don't scale eyes to a value that doesn't exist or if you're asked not to
    if scale_to not in [None, "None"]:
        if all([spectrum.get(comp) for comp in scale_to]):
            scales = (spectrum[scale_to[0]], spectrum[scale_to[1]])
            filter_mean = mean(scales)
            lefteye_scale = filter_mean / scales[0]
            righteye_scale = filter_mean / scales[1]

    real_filters_to_use = list(cam_info["filters"].keys())
    if average_filters is True:
        # construct dictionary of averaged filter values
        for v_filter, comps in cam_info["virtual_filter_mapping"].items():
            # do not attempt to average filters if both filters of
            # a pair are not present
            if not all([spectrum.get(comp) for comp in comps]):
                continue
            [real_filters_to_use.remove(comp) for comp in comps]
            values[v_filter] = {
                "wave": cam_info["virtual_filters"][v_filter],
                "mean": mean(
                    (
                        spectrum[comps[0]] * lefteye_scale,
                        spectrum[comps[1]] * righteye_scale,
                    ),
                ),
            }
            if all([comp + "_ERR" in spectrum.keys() for comp in comps]):
                values[v_filter]["err"] = (
                    spectrum[comps[0] + "_ERR"] ** 2
                    + spectrum[comps[1] + "_ERR"] ** 2
                ) ** 0.5
    # construct dictionary of leftover real filter values
    for real_filter in real_filters_to_use:
        mean_value = spectrum.get(real_filter)
        if mean_value is None:
            continue
        if real_filter.startswith("r"):
            eye_scale = righteye_scale
        else:
            eye_scale = lefteye_scale
        values[real_filter] = {
            "wave": cam_info["filters"][real_filter],
            "mean": spectrum[real_filter] * eye_scale,
        }
        if real_filter + "_ERR" in spectrum.keys():
            values[real_filter]["err"] = (
                spectrum[real_filter + "_ERR"] * eye_scale
            )
    return dict(sorted(values.items(), key=lambda item: item[1]["wave"]))


INSTRUMENT_UNCERTAINTIES = {
    # table 7, hayes et al. 2021 https://doi.org/10.1007/s11214-021-00795-x
    "ZCAM": {
        "L0R": 3.3,
        "L0G": 3.3,
        "L0B": 3.7,
        "L1": 1.4,
        "L2": 1.1,
        "L3": 0.2,
        "L4": 1.8,
        "L5": 1.6,
        "L6": 0.4,
        "R0R": 3.7,
        "R0G": 4.1,
        "R0B": 4.6,
        "R1": 0.4,
        "R2": 0.3,
        "R3": 0.6,
        "R4": 0.5,
        "R5": 0.8,
        "R6": 0.4,
    },
    # table 2, bell et al. 2017 https://doi.org/10.1002/2016EA000219
    "MCAM": {
        "L0R": 1.2,
        "L0G": 0.3,
        "L0B": 5.7,
        "L1": 4.3,
        "L2": 51.0,
        "L3": 0.3,
        "L4": 0.1,
        "L5": 0.3,
        "L6": 1.0,
        "R0R": 1.9,
        "R0G": 1.5,
        "R0B": 2.5,
        "R1": 3.7,
        "R2": 24.5,
        "R3": 3.4,
        "R4": 0.4,
        "R5": 0.5,
        "R6": 1.0,
    },
}
