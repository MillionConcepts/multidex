"""
prototypes / abstractions and literals to be used in the creation
of database models in plotter.models
"""

from ast import literal_eval
from itertools import chain
from typing import Optional, Sequence

from django.db import models
from toolz import keyfilter

from marslab.compat.xcam import polish_xcam_spectrum, DERIVED_CAM_DICT
from plotter_utils import modeldict

# these groupings may not be important, but are harmless at present

# fields that notionally have to do with "observation-" level metadata,
# however that is defined wrt mission-level divisions

# default settings for SQL fields -- just a shorthand
B_N_I = {"blank": True, "null": True, "db_index": True}
XCAM_SHARED_OBSERVATION_FIELDS = {
    # name of entire sequence or observation
    "name": models.CharField("Name", max_length=100, db_index=True),
    "sol": models.IntegerField("Sol", db_index=True),
    # ltst for first frame of sequence
    "ltst": models.TimeField("Local True Solar Time", **B_N_I),
    "seq_id": models.CharField("sequence id", max_length=20, db_index=True),
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
}

# fields that notionally have to do with single-spectrum (i.e., ROI)-level
# metadata, however that is defined wrt mission-level divisions
XCAM_SINGLE_SPECTRUM_FIELDS = {
    # color of associated ROI
    "color": models.CharField(
        "ROI Color", blank=True, max_length=20, db_index=True
    ),
    "feature": models.CharField("feature category", **B_N_I, max_length=45),
    # ############################################
    # ## lithological information -- relevant only to rocks ###
    # #########################################################
    "float": models.BooleanField("floating vs. in-place", **B_N_I),
    # large-to-small taxonomic categories for rock clusters
    "filename": models.CharField(
        "Name of archive CSV file", max_length=50, db_index=True
    ),
    # ## end lithological ###
    # stringified dict of images associated with the spectrum
    "images": models.TextField(**B_N_I, default="{}"),
}


# TODO: consider flattening these into a dict
#  using 'value' as keys

# dictionaries defining generalized interface properties
# for spectrum operation functions (band depth, etc.)
SPECTRUM_OP_BASE_PROPERTIES = {
    "type": "method",
    "value_type": "quant",
}
SPECTRUM_OP_INTERFACE_PROPERTIES = (
    SPECTRUM_OP_BASE_PROPERTIES
    | {
        "value": "ref",
        "arity": 1,
    },
    SPECTRUM_OP_BASE_PROPERTIES
    | {
        "value": "slope",
        "arity": 2,
    },
    SPECTRUM_OP_BASE_PROPERTIES
    | {
        "value": "band_avg",
        "arity": 2,
    },
    SPECTRUM_OP_BASE_PROPERTIES
    | {
        "value": "band_max",
        "arity": 2,
    },
    SPECTRUM_OP_BASE_PROPERTIES
    | {
        "value": "band_min",
        "arity": 2,
    },
    SPECTRUM_OP_BASE_PROPERTIES
    | {
        "value": "ratio",
        "arity": 2,
    },
    SPECTRUM_OP_BASE_PROPERTIES
    | {
        "value": "band_depth",
        "arity": 3,
    },
)

# dictionary defining generalized interface properties
# for various XCAM fields
XCAM_FIELD_INTERFACE_PROPERTIES = (
    {"value": "target_elevation", "value_type": "quant"},
    {"value": "ltst", "value_type": "quant"},
    {"value": "sclk", "value_type": "quant"},
    {"value": "zoom", "value_type": "qual"},
    {"value": "formation", "value_type": "qual"},
    {"value": "member", "value_type": "qual"},
    {"value": "sol", "value_type": "quant"},
    {"value": "feature", "value_type": "qual"},
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
    {"value": "scam", "value_type": "qual"},
)
for prop in chain.from_iterable(
    [XCAM_FIELD_INTERFACE_PROPERTIES, SPECTRUM_OP_INTERFACE_PROPERTIES]
):
    if "label" not in prop.keys():
        prop["label"] = prop["value"]
    if "type" not in prop.keys():
        prop["type"] = "attribute"


# ############### actual prototype classes #######################


class XSpec(models.Model):
    """
    abstract model representing an individual ROI from an XCAM-family
    instrument
    """

    # actual four-letter instrument designation: PCAM, MCAM, ZCAM
    instrument = None
    # brief and full instrument names
    instrument_brief_name = None
    instrument_full_name = None

    @classmethod
    def field_names(cls):
        return [field.name for field in cls._meta.get_fields()]

    @classmethod
    def accessible_properties(cls):
        return list(SPECTRUM_OP_INTERFACE_PROPERTIES) + [
            fip
            for fip in XCAM_FIELD_INTERFACE_PROPERTIES
            if fip["value"] in cls.field_names()
        ]

    @classmethod
    def graphable_properties(cls):
        return [
            ap
            for ap in cls.accessible_properties()
            if prop["value"] not in ("color", "seq_id", "name")
        ]

    @classmethod
    def searchable_fields(cls):
        return [
            ap
            for ap in cls.accessible_properties()
            if (ap["type"] != "method") and ap["value"] not in "ltst"
        ]

    def image_files(self):
        images = getattr(self, "images")
        if images == "":
            return {}
        return literal_eval(images)

    def filter_values(
        self,
        scale_to: Optional[Sequence] = None,
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

    def all_filter_waves(self):
        return self.filters | self.virtual_filters

    def metadata_dict(self) -> dict:
        """
        metadata-summarizing function. could be made more efficient.
        """
        return keyfilter(
            lambda x: x
            in [a_prop["value"] for a_prop in self.accessible_properties()],
            modeldict(self),
        )

    def __str__(self):
        return "sol" + str(self.sol) + "_" + self.name + "_" + self.seq_id

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
