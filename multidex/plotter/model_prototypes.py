"""
prototypes / abstractions and literals to be used in the creation
of database models in plotter.models
"""

from ast import literal_eval
import datetime as dt
from functools import cache
from typing import Optional, Sequence

from cytoolz import keyfilter
from django.db import models
from marslab.compat.xcam import polish_xcam_spectrum, DERIVED_CAM_DICT
import pandas as pd
import numpy as np

from plotter import __version__
from multidex_utils import modeldict
from plotter.field_interface_definitions import (
    METADATA_PROPERTIES,
    DYNAMIC_PROPERTIES,
    UNGRAPHABLE_FIELDS,
)

# default settings for SQL fields -- just a shorthand\
B_N_I = {"blank": True, "null": True, "db_index": True}

# model / SQL fields that notionally have to do with "observation-" level
# metadata, however that is defined wrt mission-level divisions
# these groupings may not be important, but are harmless at present
SHARED_OBSERVATION_FIELDS = {
    # name of entire sequence or observation
    "name": models.CharField("name", max_length=100, db_index=True),
    "sol": models.IntegerField("sol", **B_N_I),
    # ltst for first frame of sequence
    "ltst": models.TimeField("local true solar time", **B_N_I),
    "seq_id": models.CharField("sequence id", max_length=20, **B_N_I),
    "rover_elevation": models.FloatField("rover elevation", **B_N_I),
    "incidence_angle": models.FloatField("incidence angle", **B_N_I),
    "emission_angle": models.FloatField("emission angle", **B_N_I),
    "phase_angle": models.FloatField("phase angle", **B_N_I),
    "l_s": models.FloatField("solar longitude", **B_N_I),
    # arbitrary number for the site. part of the ROVER_NAV_FRAME
    # coordinate system
    "site": models.IntegerField("site", **B_N_I),
    # similar
    "drive": models.IntegerField("drive", **B_N_I),
    # planetographic lat/lon
    "lat": models.FloatField("rover latitude", **B_N_I),
    "lon": models.FloatField("rover longitude", **B_N_I),
    "odometry": models.FloatField("odometry (m)", **B_N_I),
    "filename": models.CharField(
        "source CSV filename", max_length=100, db_index=True
    ),
    "sclk": models.FloatField("sclk (spacecraft clock)", **B_N_I),
    "modification_time": models.CharField(
        "modification time (UTC)", max_length=25, **B_N_I
    ),
    # "bucket" field for categorizing lab spectra that might be
    # ingested into the database.
    "lab_spectrum_type": models.CharField(
        "lab spectrum type", max_length=60, **B_N_I
    ),
    # identifier for version number of multidex used to ingest spectrum
    "multidex_version": models.CharField(
        "MultiDEx version", max_length=15, **B_N_I
    ),
}

# fields that notionally have to do with single-spectrum
# (for imaging spectrometers, ROI)-level metadata, however that is defined
# wrt mission-level divisions
SINGLE_SPECTRUM_FIELDS = {
    # color used to render a spectrum. for XCAM models, corresponds to the
    # color of the ROI drawn by the original analyst.
    "color": models.CharField(
        "spectrum color", blank=True, max_length=20, db_index=True
    ),
    "feature": models.CharField("feature category", **B_N_I, max_length=45),
    # stringified dict of images associated with the spectrum
    "images": models.TextField(**B_N_I, default="{}"),
}


# ############### prototype classes #######################
class RoverSpectrum(models.Model):
    """
    abstract class representing a spectrum extracted from data taken
    by a rover-borne instrument
    """

    # instrument designation, currently by convention four letters:
    # MCAM, ZCAM, CCAM
    instrument = None
    # brief and full instrument names
    instrument_brief_name = None
    instrument_full_name = None

    # these properties are populated dynamically in models.py
    field_names = None
    filters = {}
    # TODO: should be removed as a default (needs modifications to marslab)
    virtual_filters = {}

    def clean(self, *args, **kwargs):
        self.modification_time = dt.datetime.utcnow().isoformat()[:-7] + "Z"
        self.multidex_version = __version__
        for filt in self.filters.keys():
            if getattr(self, filt.lower()) is not None:
                if getattr(self, f"{filt.lower()}_std") is None:
                    setattr(self, f"{filt.lower()}_std", 0)
        # noinspection PyUnresolvedReferences
        if self.incidence_angle is None:
            self.incidence_angle = 0
        super().clean()

    @classmethod
    @cache
    def accessible_properties(cls):
        relevant_metadata_properties = filter(
            lambda record: record["value"] in cls.field_names,
            METADATA_PROPERTIES,
        )
        return tuple(
            list(relevant_metadata_properties) + list(DYNAMIC_PROPERTIES)
        )

    @classmethod
    @cache
    def graphable_properties(cls):
        return tuple(
            filter(
                lambda prop: prop["value"] not in UNGRAPHABLE_FIELDS,
                cls.accessible_properties(),
            )
        )

    @classmethod
    @cache
    def searchable_fields(cls):
        return tuple(
            filter(
                lambda prop: prop["type"] not in ("method", "decomposition"),
                cls.accessible_properties(),
            )
        )

    def image_files(self):
        images = getattr(self, "images")
        if images == "":
            return {}
        return literal_eval(images)

    # TODO: consider removing potential binocularity as a default.
    #  that needs to happen at the marslab level. that might also make
    #  polish_xcam_spectrum less horrifying.
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
            filt + "_STD": getattr(self, filt.lower() + "_std")
            for filt in self.filters.keys()
        }
        return polish_xcam_spectrum(
            spectrum=spectrum,
            cam_info=DERIVED_CAM_DICT[self.instrument],
            scale_to=scale_to,
            average_filters=average_filters,
        )

    # TODO: remove as a default
    def all_filter_waves(self):
        return self.filters | self.virtual_filters

    def metadata_dict(self) -> dict:
        """
        metadata-summarizing function. could be made more efficient.
        """
        aprops = [a_prop["value"] for a_prop in self.accessible_properties()]
        return keyfilter(lambda x: x in aprops, modeldict(self))

    # noinspection PyUnresolvedReferences
    def __str__(self):
        return f"sol {self.sol}_{self.name}_{self.seq_id}"

    class Meta:
        abstract = True


# add SQL fields to abstract rover spectrum model
SPECTRUM_FIELDS = SINGLE_SPECTRUM_FIELDS | SHARED_OBSERVATION_FIELDS
for field_name, cam_field in SPECTRUM_FIELDS.items():
    cam_field.contribute_to_class(RoverSpectrum, field_name)


class XSpec(RoverSpectrum):
    """
    abstract class representing a spectrum extracted from data taken
    by an XCAM-family instrument (MCAM, ZCAM, perhaps PCAM)
    """
    formation = models.CharField("formation", **B_N_I, max_length=50)
    member = models.CharField("member", **B_N_I, max_length=50)
    float = models.CharField("floating / in-place", **B_N_I, max_length=15)

    @staticmethod
    def make_scatter_annotations(
        metadata_df: pd.DataFrame, truncated_ids: Sequence[int]
    ) -> np.ndarray:
        meta = metadata_df.loc[truncated_ids]
        descriptor = meta["feature"].copy()
        no_feature_ix = descriptor.loc[descriptor.isna()].index
        descriptor.loc[no_feature_ix] = meta["color"].loc[no_feature_ix]
        solstrings = meta["sol"].astype(str).copy()
        has_sol = meta['sol'].loc[meta['sol'].notna()].index
        if len(has_sol) > 0:
            # + operation throws an error if there is nothing to add to
            solstrings.loc[has_sol] = (
                "sol" + meta['sol'].loc[has_sol].apply("{:.0f}".format) + " "
            )
        solstrings.loc[meta['sol'].isna()] = ""
        return (solstrings + meta["name"] + " " + descriptor).values

    # noinspection PyUnresolvedReferences
    def roi_hex_code(self) -> str:
        return self.color_mappings[self.color]

    class Meta:
        abstract = True


def filter_fields_factory(filter_name):
    """
    return a pair of django model fields associated with a particular
    spectral filter -- one for mean measurement value and one for
    stdev
    """
    mean = models.FloatField(filter_name.lower() + " mean", **B_N_I)
    std = models.FloatField(filter_name.lower() + " stdev", **B_N_I)
    return mean, std
