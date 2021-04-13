"""
prototypes / abstractions and literals to be used in the creation
of database models in plotter.models
"""

from ast import literal_eval
from typing import Optional, Sequence

import PIL
import fs
from django.db import models
from marslab.compatibility import polish_xcam_spectrum, DERIVED_CAM_DICT
from toolz import keyfilter

from plotter_utils import modeldict

# these groupings may not be important, but are harmless at present

# fields that notionally have to do with "observation-" level metadata,
# however that is defined wrt mission-level divisions

XCAM_SHARED_OBSERVATION_FIELDS = {
    # target name, specifically, but conventionally just this
    "name": models.CharField("Name", max_length=100, db_index=True),
    "sol": models.IntegerField("Sol", db_index=True),
    # in MCAM, this is ltst for first frame of sequence
    # (usually left eye 'clear')
    "ltst": models.TimeField(
        "Local True Solar Time", blank=True, null=True, db_index=True
    ),
    # not sure what this actually is. format is of sequence
    # number in PDS header, but _value_ corresponds to request id in the PDS
    # header
    "seq_id": models.CharField(
        "sequence id, e.g. 'mcam00001'", max_length=20, db_index=True
    ),
    "rover_elevation": models.FloatField(
        "Rover Elevation", blank=True, null=True, db_index=True
    ),
    "target_elevation": models.FloatField(
        "Target Elevation", null=True, db_index=True
    ),
    "tau": models.FloatField("Tau", blank=True, null=True, db_index=True),
    "focal_distance": models.FloatField(
        "Focal Distance", blank=True, null=True, db_index=True
    ),
    "incidence_angle": models.FloatField(
        "Incidence Angle", blank=True, null=True, db_index=True
    ),
    "emission_angle": models.FloatField(
        "Emission Angle", blank=True, null=True, db_index=True
    ),
    "phase_angle": models.FloatField(
        "Phase Angle", blank=True, null=True, db_index=True
    ),
    "l_s": models.FloatField(
        "Solar Longitude", blank=True, null=True, db_index=True
    ),
    # arbitrary number for the site. part of the ROVER_NAV_FRAME
    # coordinate system
    "site": models.IntegerField("Site", blank=True, null=True, db_index=True),
    # similar
    "drive": models.IntegerField(
        "Drive", blank=True, null=True, db_index=True
    ),
    # planetographic lat/lon -- for MCAM, not in the image labels in PDS;
    # derived from localization team products?
    "lat": models.FloatField("Latitude", blank=True, null=True, db_index=True),
    "lon": models.FloatField(
        "Longitude", blank=True, null=True, db_index=True
    ),
    "odometry": models.FloatField(
        "Odometry", blank=True, null=True, db_index=True
    ),
    "filename": models.CharField(
        "Archive CSV File", max_length=30, db_index=True
    ),
    "sclk": models.IntegerField(
        "Spacecraft Clock", blank=True, null=True, db_index=True
    ),
}

# fields that notionally have to do with single-spectrum (i.e., ROI)-level
# metadata, however that is defined wrt mission-level divisions
XCAM_SINGLE_SPECTRUM_FIELDS = {
    # color of associated ROI
    "color": models.CharField(
        "ROI Color", blank=True, max_length=20, db_index=True
    ),
    "feature": models.CharField(
        "feature category",
        blank=True,
        null=True,
        db_index=True,
        max_length=45,
    ),
    # ############################################
    # ## lithological information -- relevant only to rocks ###
    # #########################################################
    "float": models.BooleanField(
        "floating vs. in-place", blank=True, null=True, db_index=True
    ),
    # large-to-small taxonomic categories for rock clusters
    "formation": models.CharField(
        "Formation", blank=True, null=True, max_length=50, db_index=True
    ),
    "member": models.CharField(
        "Member", blank=True, null=True, max_length=50, db_index=True
    ),
    "filename": models.CharField(
        "Name of archive CSV file", max_length=50, db_index=True
    ),
    # ## end lithological ###
    "notes": models.CharField(
        "Notes", blank=True, null=True, max_length=100, db_index=True
    ),
    # stringified dict of images associated with the spectrum
    "images": models.TextField(blank=True, null=True, db_index=True),
}

# dictionary defining generalized interface properties
# for spectrum operation functions (band depth, etc.)
SPECTRUM_OP_INTERFACE_PROPERTIES = (
    {
        "label": "band value",
        "value": "ref",
        "type": "method",
        "arity": 1,
        "value_type": "quant",
    },
    {
        "label": "band slope",
        "value": "slope",
        "type": "method",
        "arity": 2,
        "value_type": "quant",
    },
    {
        "label": "band average",
        "value": "band_avg",
        "type": "method",
        "arity": 2,
        "value_type": "quant",
    },
    {
        "label": "band maximum",
        "value": "band_max",
        "type": "method",
        "arity": 2,
        "value_type": "quant",
    },
    {
        "label": "band minimum",
        "value": "band_min",
        "type": "method",
        "arity": 2,
        "value_type": "quant",
    },
    {
        "label": "ratio",
        "value": "ratio",
        "type": "method",
        "arity": 2,
        "value_type": "quant",
    },
    {
        "label": "band depth (middle)",
        "value": "band_depth_custom",
        "type": "method",
        "arity": 3,
        "value_type": "quant",
    },
    {
        "label": "band depth (minimum)",
        "value": "band_depth_min",
        "type": "method",
        "arity": 2,
        "value_type": "quant",
    },
)

# dictionary defining generalized interface properties
# for various XCAM fields
XCAM_FIELD_INTERFACE_PROPERTIES = (
    {
        "label": "target_elevation",
        "value": "target_elevation",
        "type": "attribute",
        "value_type": "quant",
    },
    {
        "label": "ltst",
        "value": "ltst",
        "type": "attribute",
        "value_type": "quant",
    },
    {
        "label": "sclk",
        "value": "sclk",
        "type": "attribute",
        "value_type": "quant",
    },
    {
        "label": "zoom",
        "value": "zoom",
        "type": "attribute",
        "value_type": "qual",
    },
    {
        "label": "formation",
        "value": "formation",
        "type": "attribute",
        "value_type": "qual",
    },
    {
        "label": "member",
        "value": "member",
        "type": "attribute",
        "value_type": "qual",
    },
    {
        "label": "sol",
        "value": "sol",
        "type": "attribute",
        "value_type": "quant",
    },
    {
        "label": "feature",
        "value": "feature",
        "type": "attribute",
        "value_type": "qual",
    },
    {
        "label": "color",
        "value": "color",
        "type": "attribute",
        "value_type": "qual",
    },
    {
        "label": "name",
        "value": "name",
        "type": "attribute",
        "value_type": "qual",
    },
    {
        "label": "seq_id",
        "value": "seq_id",
        "type": "attribute",
        "value_type": "qual",
    },
    {
        "label": "tau",
        "value": "tau",
        "type": "attribute",
        "value_type": "quant",
    },
    {
        "label": "lat",
        "value": "lat",
        "type": "attribute",
        "value_type": "quant",
    },
    {
        "label": "lon",
        "value": "lon",
        "type": "attribute",
        "value_type": "quant",
    },
    {
        "label": "focal_distance",
        "value": "focal_distance",
        "type": "attribute",
        "value_type": "quant",
    },
    {
        "label": "emission_angle",
        "value": "emission_angle",
        "type": "attribute",
        "value_type": "quant",
    },
    {
        "label": "incidence_angle",
        "value": "incidence_angle",
        "type": "attribute",
        "value_type": "quant",
    },
    {
        "label": "phase_angle",
        "value": "phase_angle",
        "type": "attribute",
        "value_type": "quant",
    },
)


# actual prototype classes


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

    accessible_properties = (
        XCAM_FIELD_INTERFACE_PROPERTIES + SPECTRUM_OP_INTERFACE_PROPERTIES
    )
    graphable_properties = [
        prop
        for prop in accessible_properties
        if prop["value"] not in ("color", "seq_id", "name")
    ]
    searchable_fields = [
        prop
        for prop in accessible_properties
        if (prop["type"] != "method") and prop["value"] not in "ltst"
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

    def metadata_dict(self) -> dict:
        """
        metadata-summarizing function. could be made more efficient.
        """
        return keyfilter(
            lambda x: x
            in [prop["value"] for prop in self.accessible_properties],
            modeldict(self),
        )

    def __str__(self):
        return "sol" + str(self.sol) + "_" + self.name + "_" + self.seq_id

    class Meta:
        abstract = True


# add SQL fields to abstract xcam model
XCAM_FIELDS = XCAM_SINGLE_SPECTRUM_FIELDS | XCAM_SHARED_OBSERVATION_FIELDS
for field_name, field in XCAM_FIELDS.items():
    field.contribute_to_class(XSpec, field_name)


def filter_fields_factory(filter_name):
    """
    return a pair of django model fields associated with a particular
    spectral filter -- one for mean measurement value and one for
    variance / stdev / etc
    """
    mean = models.FloatField(
        filter_name.lower() + " mean", blank=True, null=True, db_index=True
    )
    err = models.FloatField(
        filter_name.lower() + " error", blank=True, null=True, db_index=True
    )
    return mean, err
