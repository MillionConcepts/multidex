"""
prototypes / abstractions and literals to be used in the creation
of database models in plotter.models
"""

from ast import literal_eval
import datetime as dt
from functools import cache
from itertools import chain
from typing import Optional, Sequence

from cytoolz import keyfilter
from django.db import models
import pandas as pd
import numpy as np

# TODO: ewwwww
from plotter import __version__
from marslab.compat.xcam import polish_xcam_spectrum, DERIVED_CAM_DICT
from multidex_utils import modeldict

# these groupings may not be important, but are harmless at present

# fields that notionally have to do with "observation-" level metadata,
# however that is defined wrt mission-level divisions


# default settings for SQL fields -- just a shorthand
B_N_I = {"blank": True, "null": True, "db_index": True}

XCAM_SHARED_OBSERVATION_FIELDS = {
    # name of entire sequence or observation
    "name": models.CharField("Name", max_length=100, db_index=True),
    "sol": models.IntegerField("Sol", **B_N_I),
    # ltst for first frame of sequence
    "ltst": models.TimeField("Local True Solar Time", **B_N_I),
    "seq_id": models.CharField("sequence id", max_length=20, **B_N_I),
    "rover_elevation": models.FloatField("Rover Elevation", **B_N_I),
    "target_elevation": models.FloatField(
        "Target Elevation", null=True, db_index=True
    ),
    "tau": models.FloatField("Tau", **B_N_I),
    "focal_distance": models.FloatField("Focal Distance", **B_N_I),
    "incidence_angle": models.FloatField("Incidence Angle", **B_N_I),
    "emission_angle": models.FloatField("Emission Angle", **B_N_I),
    "phase_angle": models.FloatField("Phase Angle", **B_N_I),
    "l_s": models.FloatField("Solar Longitude", **B_N_I),
    # arbitrary number for the site. part of the ROVER_NAV_FRAME
    # coordinate system
    "site": models.IntegerField("Site", **B_N_I),
    # similar
    "drive": models.IntegerField("Drive", **B_N_I),
    # planetographic lat/lon
    "lat": models.FloatField("Latitude", **B_N_I),
    "lon": models.FloatField("Longitude", **B_N_I),
    "odometry": models.FloatField("Odometry", **B_N_I),
    "filename": models.CharField(
        "Source CSV Filename", max_length=100, db_index=True
    ),
    "sclk": models.IntegerField("Spacecraft Clock", **B_N_I),
    "modification_time": models.CharField(
        "Modification Time UTC", max_length=25, **B_N_I
    ),
    # "bucket" field for categorizing lab spectra that might be
    # ingested into the database.
    "lab_spectrum_type": models.CharField(
        "Lab Spectrum Type", max_length=60, **B_N_I
    ),
    # identifier for version number of multidex used to ingest spectrum
    "multidex_version": models.CharField(
        "MultiDEx Version", max_length=15, **B_N_I
    ),
}

# fields that notionally have to do with single-spectrum (i.e., ROI)-level
# metadata, however that is defined wrt mission-level divisions
XCAM_SINGLE_SPECTRUM_FIELDS = {
    # color of associated ROI
    "color": models.CharField(
        "ROI Color", blank=True, max_length=20, db_index=True
    ),
    "feature": models.CharField("feature category", **B_N_I, max_length=45),
    "filename": models.CharField(
        "Name of archive CSV file", max_length=50, db_index=True
    ),
    # stringified dict of images associated with the spectrum
    "images": models.TextField(**B_N_I, default="{}"),
}

# TODO: consider flattening these into a dict
#  using 'value' as keys
# dictionaries defining generalized interface properties
# for spectrum operation functions (band depth, etc.)
SPECTRUM_OP_BASE_PROPERTIES = {"type": "method", "value_type": "quant"}
SPECTRUM_OP_INTERFACE_PROPERTIES = (
    {"value": "ref", "arity": 1},
    {"value": "slope", "arity": 2},
    {"value": "band_avg", "arity": 2},
    {"value": "band_max", "arity": 2},
    {"value": "band_min", "arity": 2},
    {"value": "ratio", "arity": 2},
    {"value": "band_depth", "arity": 3},
)

for op in SPECTRUM_OP_INTERFACE_PROPERTIES:
    op |= SPECTRUM_OP_BASE_PROPERTIES

REDUCTION_OP_BASE_PROPERTIES = {
    "value_type": "quant",
    "type": "decomposition",
}

PCA_INTERFACE_PROPERTIES = [{"function": "PCA", "value": "PCA"}]

# TODO: figure out how to implement decomposition parameter
#  controls; maybe this doesn't go here, it's a separate interface,
#  something like that
REDUCTION_OP_INTERFACE_PROPERTIES = PCA_INTERFACE_PROPERTIES
for op in REDUCTION_OP_INTERFACE_PROPERTIES:
    op |= REDUCTION_OP_BASE_PROPERTIES

# TODO: this is ugly as sin, flatten / concatenate this somehow
# dictionary defining generalized interface properties
# for various XCAM fields
XCAM_FIELD_INTERFACE_PROPERTIES = (
    {"value": "feature", "value_type": "qual"},
    {"value": "feature_subtype", "value_type": "qual"},
    {"value": "ltst", "value_type": "quant"},
    {"value": "sclk", "value_type": "quant"},
    {"value": "zoom", "value_type": "qual"},
    {"value": "group", "value_type": "qual"},
    {"value": "formation", "value_type": "qual"},
    {"value": "member", "value_type": "qual"},
    {"value": "target_elevation", "value_type": "quant"},
    {"value": "rover_elevation", "value_type": "quant"},
    {"value": "sol", "value_type": "quant"},
    {"value": "color", "value_type": "qual"},
    {"value": "name", "value_type": "qual"},
    {"value": "seq_id", "value_type": "qual"},
    {"value": "tau", "value_type": "quant"},
    {"value": "lat", "value_type": "quant"},
    {"value": "lon", "value_type": "quant"},
    {"value": "focal_distance", "value_type": "quant"},
    {"value": "emission_angle", "value_type": "quant"},
    {"value": "incidence_angle", "value_type": "quant"},
    {"value": "phase_angle", "value_type": "quant"},
    {"value": "rms", "value_type": "quant"},
    {"value": "target", "value_type": "qual"},
    {"value": "compression", "value_type": "qual"},
    {"value": "morphology", "value_type": "qual"},
    {"value": "distance", "value_type": "qual"},
    {"value": "location", "value_type": "qual"},
    {"value": "workspace", "value_type": "qual"},
    {"value": "compression_quality", "value_type": "quant"},
    {"value": "scam_libs", "value_type": "qual"},
    {"value": "scam_raman", "value_type": "qual"},
    {"value": "scam_rmi", "value_type": "qual"},
    {"value": "scam_visir", "value_type": "qual"},
    {"value": "scam_libs", "value_type": "qual"},
    {"value": "sherloc", "value_type": "qual"},
    {"value": "watson", "value_type": "qual"},
    {"value": "min_count", "value_type": "quant"},
    {"value": "analysis_name", "value_type": "qual"},
    {"value": "scam_libs", "value_type": "qual"},
    {"value": "scam_visir", "value_type": "qual"},
    {"value": "scam_raman", "value_type": "qual"},
    {"value": "float", "value_type": "qual"},
    {"value": "rock_surface", "value_type": "qual"},
    {"value": "grain_size", "value_type": "qual"},
    {"value": "soil_location", "value_type": "qual"},
    {"value": "soil_color", "value_type": "qual"},
    {"value": "landform_type", "value_type": "qual"},
    {"value": "odometry", "value_type": "quant"},
    {"value": "lab_spectrum_type", "value_type": "qual"},
    {"value": "distance_m", "value_type": "quant"},
    {"value": "exposure", "value_type": "quant"},
    {"value": "target_type", "value_type": "qual"},
    {"value": "temp", "value_type": "quant"},
    {"value": "target_type_shot_specific", "value_type": "qual"},
    {"value": "lmst", "value_type": "quant"},
    {"value": "instrument_elevation", "value_type": "quant"},
    {"value": "instrument_azimuth", "value_type": "quant"},
    {"value": "solar_azimuth", "value_type": "quant"},
    {"value": "solar_elevation", "value_type": "quant"},
    {"value": "temp", "value_type": "quant"},
    {"value": "libs_before", "value_type": "qual"},
    {"value": "raster_location", "value_type": "quant"},
)

XCAM_CALCULATED_PROPERTIES = (
    # slightly special cases: these are computed at runtime
    {"value": "filter_avg", "value_type": "quant", "type": "computed"},
    {"value": "err_avg", "value_type": "quant", "type": "computed"},
    {"value": "rel_err_avg", "value_type": "quant", "type": "computed"},
)
for prop in chain.from_iterable(
    [
        XCAM_FIELD_INTERFACE_PROPERTIES,
        SPECTRUM_OP_INTERFACE_PROPERTIES,
        REDUCTION_OP_INTERFACE_PROPERTIES,
        XCAM_CALCULATED_PROPERTIES,
    ]
):
    if "label" not in prop.keys():
        prop["label"] = prop["value"]
    if "type" not in prop.keys():
        prop["type"] = "attribute"


# ############### actual prototype classes #######################


class XSpec(models.Model):
    """
    abstract class representing an ROI from an XCAM-family instrument
    """

    # four-letter instrument designation: PCAM, MCAM, ZCAM, CCAM
    instrument = None
    # brief and full instrument names
    instrument_brief_name = None
    instrument_full_name = None

    # this property is populated in models.py
    field_names = None

    def clean(self, *args, **kwargs):
        self.modification_time = dt.datetime.utcnow().isoformat()[:-7] + "Z"
        self.multidex_version = __version__
        super().clean()

    @classmethod
    @cache
    def accessible_properties(cls):
        return (
            list(SPECTRUM_OP_INTERFACE_PROPERTIES)
            + list(REDUCTION_OP_INTERFACE_PROPERTIES)
            + list(XCAM_CALCULATED_PROPERTIES)
            + [
                fip
                for fip in XCAM_FIELD_INTERFACE_PROPERTIES
                if fip["value"] in cls.field_names
            ]
        )

    @classmethod
    @cache
    def graphable_properties(cls):
        return [
            ap
            for ap in cls.accessible_properties()
            if prop["value"]
            not in ("color", "seq_id", "name", "analysis_name", "target")
        ]

    @classmethod
    @cache
    def searchable_fields(cls):
        return [
            ap
            for ap in cls.accessible_properties()
            if (ap["type"] not in ("method", "decomposition"))
        ]

    def image_files(self):
        images = getattr(self, "images")
        if images == "":
            return {}
        return literal_eval(images)

    @cache
    def filter_values(
        self,
        scale_to: Optional[Sequence[str]] = None,
        average_filters: bool = False,
    ) -> dict[str, dict]:
        """
        return dictionary of filter values, optionally scaled and merged
        according to MERSPECT-style rules
        scale_to: None or tuple of (lefteye filter name, righteye filter name)
        """
        spectrum = {
            filt: getattr(self, filt.lower()) for filt in self.filters.keys()
        }
        spectrum |= {
            filt + "_ERR": getattr(self, filt.lower() + "_err")
            for filt in self.filters.keys()
        }
        return polish_xcam_spectrum(
            spectrum=spectrum,
            cam_info=DERIVED_CAM_DICT[self.instrument],
            scale_to=scale_to,
            average_filters=average_filters,
        )

    @staticmethod
    def make_scatter_annotations(metadata_df: pd.DataFrame, truncated_ids: Sequence[int]) -> np.ndarray:
        meta = metadata_df.loc[truncated_ids]
        descriptor = meta["target"].copy()
        no_feature_ix = descriptor.loc[descriptor.isna()].index
        descriptor.loc[no_feature_ix] = meta["color"].loc[no_feature_ix]
        sol = meta["sol"].copy()
        has_sol = sol.loc[sol.notna()].index
        if len(has_sol) > 0:
            # + operation throws an error if there is nothing to add to
            sol.loc[has_sol] = (
                    "sol" + sol.loc[has_sol].apply("{:.0f}".format) + " "
            )
        sol.loc[sol.isna()] = ""
        return (sol + meta["name"] + " " + descriptor).values

    def all_filter_waves(self):
        return self.filters | self.virtual_filters

    def metadata_dict(self) -> dict:
        """
        metadata-summarizing function. could be made more efficient.
        """
        aprops = [a_prop["value"] for a_prop in self.accessible_properties()]
        return keyfilter(lambda x: x in aprops, modeldict(self))

    def __str__(self):
        return f"sol {self.sol}_{self.name}_{self.seq_id}"

    class Meta:
        abstract = True


# add SQL fields to abstract xcam model
XCAM_FIELDS = XCAM_SINGLE_SPECTRUM_FIELDS | XCAM_SHARED_OBSERVATION_FIELDS
for field_name, cam_field in XCAM_FIELDS.items():
    cam_field.contribute_to_class(XSpec, field_name)


def filter_fields_factory(filter_name):
    """
    return a pair of django model fields associated with a particular
    spectral filter -- one for mean measurement value and one for
    variance / stdev / etc
    """
    mean = models.FloatField(filter_name.lower() + " mean", **B_N_I)
    err = models.FloatField(filter_name.lower() + " error", **B_N_I)
    return mean, err
