from types import MappingProxyType
from typing import Sequence

import numpy as np
import pandas as pd
from django.db import models
from marslab.compat.mertools import (
    MERSPECT_M20_COLOR_MAPPINGS,
    MERSPECT_MSL_COLOR_MAPPINGS,
)
from marslab.compat.xcam import DERIVED_CAM_DICT

from plotter.model_prototypes import (
    XSpec,
    filter_fields_factory,
    B_N_I,
    RoverSpectrum,
)


class ZSpec(XSpec):
    feature_subtype = models.CharField(
        "feature subtype", **B_N_I, max_length=45
    )
    zoom = models.CharField("Zoom Code", max_length=10, **B_N_I)
    # shared target identifier with other instruments, usually null
    target = models.CharField("Target", max_length=60, **B_N_I)
    # TODO: some of this wants to move up to XCAM after MCAM asdf products
    #  are ready for ingest
    # rover motion counter for the mast -- for repointed stereo
    # observations, this is the first RSM in the sequence
    rsm = models.IntegerField("RSM", **B_N_I)
    # timestamp of file if automatically produced by asdf
    file_timestamp = models.CharField(max_length=30, null=True)
    compression = models.CharField("compression", max_length=40, **B_N_I)
    compression_quality = models.IntegerField("compression quality", **B_N_I)
    grain_size = models.CharField("grain size", max_length=20, **B_N_I)
    distance = models.CharField("distance", max_length=20, **B_N_I)
    location = models.CharField("location", max_length=60, **B_N_I)
    analysis_name = models.CharField("analysis name", max_length=30, **B_N_I)
    min_count = models.IntegerField("minimum pixel count", **B_N_I)
    outcrop = models.CharField("outcrop", **B_N_I, max_length=50)
    # radiometric calibration file metadata fields
    rc_caltarget_file = models.CharField(
        "caltarget file", max_length=80, **B_N_I
    )
    rc_sol = models.IntegerField("caltarget sol", **B_N_I)
    rc_seq_id = models.CharField("caltarget seq id", max_length=10, **B_N_I)
    rc_ltst = models.TimeField("caltarget ltst", **B_N_I)
    rc_solar_azimuth = models.FloatField("caltarget solar azimuth", **B_N_I)
    # caltarget geometry based on values at black chip center
    rc_incidence_angle = models.FloatField(
        "caltarget incidence angle", **B_N_I
    )
    rc_azimuth_angle = models.FloatField("caltarget azimuth angle", **B_N_I)
    rc_emission_angle = models.FloatField("caltarget emission angle", **B_N_I)
    rc_scaling_factor = models.FloatField("rad-to-iof scaling factor", **B_N_I)
    rc_uncertainty = models.FloatField("rc uncertainty", **B_N_I)
    # value given in rc files -- perhaps temporary
    azimuth_angle = models.FloatField("azimuth angle (rc)", **B_N_I)
    caltarget_element = models.CharField(
        "caltarget_element", max_length=30, **B_N_I
    )
    instrument = "ZCAM"
    instrument_brief_name = "Mastcam-Z"

    color_mappings = MERSPECT_M20_COLOR_MAPPINGS | {"black": "#000000"}

    def overlay_browse_file_info(self, image_directory: str) -> dict:
        files = self.image_files()
        images = {}
        for image_type, filename in files.items():
            images[image_type + "_file"] = filename
        return images

    # filters we are allowed to use for PCA -- implementing this as
    # a method in order to flexibly handle virtual filters etc.
    @staticmethod
    def permissibly_explanatory_bandpasses(filts):
        # return filts
        return [
            f for f in filts if not f.endswith(("R", "G", "B"))
        ]


class MSpec(XSpec):
    group = models.CharField("group", **B_N_I, max_length=50)
    feature_subtype = models.CharField(
        "feature subtype", **B_N_I, max_length=45
    )
    rock_class = models.CharField("rock class", **B_N_I, max_length=45)
    soil_class = models.CharField("soil class", **B_N_I, max_length=45)
    instrument = "MCAM"
    instrument_brief_name = "Mastcam"
    color_mappings = MERSPECT_MSL_COLOR_MAPPINGS | {"black": "#000000"}


class CSpec(RoverSpectrum):
    target = models.CharField("Target", **B_N_I, max_length=50)
    type_of_product = models.CharField(
        "Type of Product", **B_N_I, max_length=50
    )
    target_distance = models.FloatField("Distance (m)", max_length=20, **B_N_I)
    lmst = models.TimeField("Local Mean Solar Time", **B_N_I)
    exposure = models.IntegerField("Exposure (ms)", **B_N_I)
    target_type = models.CharField("Target Type", max_length=30, **B_N_I)
    target_type_shot_specific = models.CharField(
        "Target Type (shot specific)", max_length=60, **B_N_I
    )
    target = models.CharField("Target", **B_N_I)
    instrument_elevation = models.FloatField(
        "Instrument Elevation (deg)", **B_N_I
    )
    instrument_azimuth = models.FloatField("Instrument Azimuth (deg)", **B_N_I)
    solar_azimuth = models.FloatField("Solar Azimuth (deg)", **B_N_I)
    solar_elevation = models.FloatField("Solar Elevation (deg)", **B_N_I)
    temp = models.FloatField("Instrument Temperature (C)", **B_N_I)
    libs_before = models.CharField(
        "LIBS before or after passive", max_length=30, **B_N_I
    )
    raster_location = models.IntegerField("Raster Location #", **B_N_I)
    group = models.CharField("Group", **B_N_I)
    formation = models.CharField("Formation", **B_N_I)
    member = models.CharField("Member", **B_N_I)
    tau = models.FloatField("tau", **B_N_I)

    instrument = "CCAM"
    instrument_brief_name = "ChemCam"

    @staticmethod
    def make_scatter_annotations(
            metadata_df: pd.DataFrame, truncated_ids: Sequence[int]
    ) -> np.ndarray:
        meta = metadata_df.loc[truncated_ids]
        descriptor = meta["target"].copy()
        no_feature_ix = descriptor.loc[descriptor.isna()].index
        descriptor.loc[no_feature_ix] = meta["target"].loc[no_feature_ix]
        sol = meta["sol"].copy()
        has_sol = sol.loc[sol.notna()].index
        if len(has_sol) > 0:
            # + operation throws an error if there is nothing to add to
            sol.loc[has_sol] = sol.loc[has_sol].apply("{:.0f}".format) + " "
        sol.loc[sol.isna()] = ""
        raster = meta["raster_location"].copy()
        has_raster = raster.loc[raster.notna()].index
        raster.loc[has_raster] = (
                raster.loc[has_raster].apply("{:.0f}".format) + " "
        )
        return (
                meta["name"]
                + "<br>sol: "
                + sol
                + "<br>target: "
                + descriptor
                + "<br>raster #: "
                + raster
        ).values


class SSpec(RoverSpectrum):
    target = models.CharField("Target", **B_N_I, max_length=50)
    type_of_product = models.CharField(
        "Type of Product", **B_N_I, max_length=50
    )
    lmst = models.TimeField("Local Mean Solar Time", **B_N_I)
    target_type = models.CharField("Target Type", max_length=30, **B_N_I)
    instrument_elevation = models.FloatField("Instrument Elevation (deg)", **B_N_I)
    instrument_azimuth = models.FloatField("Instrument Azimuth (deg)", **B_N_I)
    solar_azimuth = models.FloatField("Solar Azimuth (deg)", **B_N_I)
    solar_elevation = models.FloatField("Solar Elevation (deg)", **B_N_I)
    raster_location = models.IntegerField("Raster Location #", **B_N_I)
    rsm_azimuth = models.FloatField("RSM Azimuth (deg)", **B_N_I)
    rsm_elevation = models.FloatField("RSM Elevation (deg)", **B_N_I)
    tau = models.FloatField("Tau", **B_N_I)
    uv_rows = models.IntegerField("UV Rows")
    vio_rows = models.IntegerField("VIO Rows")
    red_rows = models.TextField("Red Rows")
    t_integ_real = models.FloatField("Integration Time (real)")
    p1400 = models.FloatField("P1400")
    p1900 = models.FloatField("P1900")
    p2300 = models.FloatField("P2300")
    formation = models.CharField("Formation", **B_N_I)
    member = models.CharField("Member", **B_N_I)
    powerfail = models.CharField("Power Fail", **B_N_I)
    saturation = models.CharField("Saturation", **B_N_I)
    focus_position_mm = models.CharField("Focus Position (mm)")

    instrument = "SCAM"
    instrument_brief_name = "SuperCam"

    @staticmethod
    def make_scatter_annotations(
            metadata_df: pd.DataFrame, truncated_ids: Sequence[int]
    ) -> np.ndarray:
        meta = metadata_df.loc[truncated_ids]
        return (
                meta["name"]
        ).values


class TestSpec(RoverSpectrum):
    """mock spectrum class for tests"""
    instrument = "TEST"
    filters = {"test": 100}
    virtual_filters = {"test": 100}
    field_names = ()


# bulk setup for each instrument
for spec_model in [ZSpec, MSpec, CSpec, SSpec, TestSpec]:
    if spec_model.instrument not in DERIVED_CAM_DICT.keys():
        continue

    # mappings from filter name to nominal band centers, in nm
    setattr(
        spec_model,
        "filters",
        DERIVED_CAM_DICT[spec_model.instrument]["filters"],
    )
    setattr(
        spec_model,
        "virtual_filters",
        DERIVED_CAM_DICT[spec_model.instrument]["virtual_filters"],
    )
    # which real filters do virtual filters correspond to?
    setattr(
        spec_model,
        "virtual_filter_mapping",
        DERIVED_CAM_DICT[spec_model.instrument]["virtual_filter_mapping"],
    )
    # if we're only giving options for averaged filters,
    # what is the canonical list?
    setattr(
        spec_model,
        "canonical_averaged_filters",
        DERIVED_CAM_DICT[spec_model.instrument]["canonical_averaged_filters"],
    )

    # set up SQL fields for each filter
    for filt in DERIVED_CAM_DICT[spec_model.instrument]["filters"].keys():
        mean_field, err_field = filter_fields_factory(filt)
        mean_field.contribute_to_class(spec_model, filt.lower())
        err_field.contribute_to_class(spec_model, filt.lower() + "_std")

    # add fields to each model
    setattr(
        spec_model,
        "field_names",
        [field.name for field in spec_model._meta.fields],
    )

# for automated model selection
INSTRUMENT_MODEL_MAPPING = MappingProxyType(
    {"ZCAM": ZSpec, "MCAM": MSpec, "CCAM": CSpec, "SCAM": SSpec, "TEST": TestSpec}
)
