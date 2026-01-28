from functools import cache
from types import MappingProxyType
from typing import Sequence

from django.db import models
import numpy as np
import pandas as pd
from marslab.compat.mertools import (
    MERSPECT_M20_COLOR_MAPPINGS, MERSPECT_MSL_COLOR_MAPPINGS,
)
from marslab.compat.xcam import DERIVED_CAM_DICT

from multidex.plotter.field_interface_definitions import (
    ASDF_CART_COLS, ASDF_PHOT_COLS
)
from multidex.plotter.model_prototypes import (
    B_N_I, filter_fields_factory, RoverSpectrum, XSpec,
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
    outcrop = models.CharField("outcrop", **B_N_I, max_length=50)
    # spatial data quality flag produced during ingest
    spatial_flag = models.CharField("spatial_flag", **B_N_I, max_length=15)
    phot_flag = models.CharField("photometry_flag", **B_N_I, max_length=15)
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
    tau = models.FloatField("Tau", **B_N_I)
    instrument = "ZCAM"
    instrument_brief_name = "Mastcam-Z"

    color_mappings = MERSPECT_M20_COLOR_MAPPINGS | {"black": "#000000"}

    # TODO: check if unused image_directory argument is cruft or oversight
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

    # TODO: messy
    @classmethod
    @cache
    def accessible_properties(cls):
        props = super().accessible_properties()
        rc_qa_props = [
            {
                'value': name,
                'value_type': 'quant',
                'type': 'non_filter_computed',
                'label': name
            }
            for name in ('rc_goodness', 'rc_ltst_off', 'rc_sol_off')
        ]
        return props + tuple(rc_qa_props)

    @staticmethod
    def cal_goodness(calframe):
        goodness_series = pd.Series(
            index=calframe.index, name='rc_goodness', dtype='f4'
        )
        ltst_off_series = goodness_series.copy()
        ltst_off_series.name = 'rc_ltst_off'
        sol_off_series = goodness_series.copy()
        sol_off_series.name = 'rc_sol_off'
        hascal = calframe.loc[calframe.notna().all(axis=1)].astype('f4')
        goodness = (
           abs(hascal['sol'] - hascal['rc_sol'])
           + abs(hascal['ltst'] - hascal['rc_ltst']) / 3600
        )
        goodness_series.loc[hascal.index] = np.log((1 / goodness) + 1)
        ltst_off_series.loc[hascal.index] = (
            abs(hascal['ltst'] - hascal['rc_ltst'])
        )
        sol_off_series.loc[hascal.index] = (
            abs(hascal['sol'] - hascal['rc_sol'])
        )
        return {
            'rc_goodness': goodness_series,
            'rc_ltst_off': ltst_off_series,
            'rc_sol_off': sol_off_series
        }


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
    rsm_l = models.IntegerField("RSM_L", **B_N_I)
    rsm_r = models.IntegerField("RSM_R", **B_N_I)


class CSpec(RoverSpectrum):
    target = models.CharField("Target", **B_N_I, max_length=50)
    type_of_product = models.CharField(
        "Type of Product", **B_N_I, max_length=50
    )
    target_distance = models.FloatField("Distance (m)", **B_N_I)
    lmst = models.TimeField("Local Mean Solar Time", max_length=30, **B_N_I)
    exposure = models.IntegerField("Exposure (ms)", **B_N_I)
    target_type = models.CharField("Target Type", max_length=30, **B_N_I)
    target_type_shot_specific = models.CharField(
        "Target Type (shot specific)", max_length=60, **B_N_I
    )
    target = models.CharField("Target", max_length=60, **B_N_I)
    instrument_elevation = models.FloatField(
        "Instrument Elevation (deg)", max_length=10, **B_N_I
    )
    instrument_azimuth = models.FloatField("Instrument Azimuth (deg)", **B_N_I)
    solar_azimuth = models.FloatField("Solar Azimuth (deg)", **B_N_I)
    solar_elevation = models.FloatField("Solar Elevation (deg)", **B_N_I)
    temp = models.FloatField("Instrument Temperature (C)", **B_N_I)
    libs_before = models.CharField(
        "LIBS before or after passive", max_length=30, **B_N_I
    )
    raster_location = models.IntegerField("Raster Location #", **B_N_I)
    group = models.CharField("Group", max_length=45, **B_N_I)
    formation = models.CharField("Formation", max_length=45, **B_N_I)
    member = models.CharField("Member", max_length=45, **B_N_I)
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
        intpat = r"\.\d+|nan"
        sol = meta["sol"].astype(str).str.replace(intpat, "", regex=True)
        raster = meta[
            "raster_location"
        ].astype(str).str.replace(intpat, "", regex=True)
        return (
                meta["name"]
                + "<br>sol: "
                + sol
                + "<br>target: "
                + descriptor
                + "<br>raster #: "
                + raster
        ).values

    @staticmethod
    def rearrange_band_depth_for_title(text: str) -> str:
        filts = text.split()
        return (
            f"{filts[0]} {filts[3]}, " f"shoulders at {filts[1]} and " f"{filts[2]}"
        )



class SSpec(RoverSpectrum):
    target = models.CharField("Target", **B_N_I, max_length=50)
    type_of_product = models.CharField(
        "Type of Product", **B_N_I, max_length=50
    )
    lmst = models.TimeField("Local Mean Solar Time", max_length=30, **B_N_I)
    target_type = models.CharField("Target Type", max_length=30, **B_N_I)
    instrument_elevation = models.FloatField("Instrument Elevation (deg)", **B_N_I)
    instrument_azimuth = models.FloatField("Instrument Azimuth (deg)", **B_N_I)
    solar_azimuth = models.FloatField("Solar Azimuth (deg)", **B_N_I)
    solar_elevation = models.FloatField("Solar Elevation (deg)", **B_N_I)
    raster_location = models.IntegerField("Raster Location #", **B_N_I)
    rsm_azimuth = models.FloatField("RSM Azimuth (deg)", **B_N_I)
    rsm_elevation = models.FloatField("RSM Elevation (deg)", **B_N_I)
    tau = models.FloatField("Tau", **B_N_I)
    uv_rows = models.IntegerField("UV Rows", **B_N_I)
    vio_rows = models.IntegerField("VIO Rows", **B_N_I)
    red_rows = models.CharField("Red Rows", max_length=10, **B_N_I)
    t_integ_real = models.FloatField("Integration Time (real)", **B_N_I)
    p750 = models.FloatField("P750", **B_N_I)
    p1400 = models.FloatField("P1400", **B_N_I)
    p1900 = models.FloatField("P1900", **B_N_I)
    p2300 = models.FloatField("P2300", **B_N_I)
    formation = models.CharField("Formation", max_length=45, **B_N_I)
    member = models.CharField("Member", max_length=45, **B_N_I)
    powerfail = models.IntegerField("Power Fail", **B_N_I)
    saturation = models.FloatField("Saturation", **B_N_I)
    focus_position_mm = models.FloatField("Focus Position (mm)", **B_N_I)
    tdb_name = models.CharField("TDB name", max_length=45, **B_N_I)
    corrected = models.BooleanField("Corrected", default=False)

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

    @staticmethod
    def rearrange_band_depth_for_title(text: str) -> str:
        filts = text.split()
        return (
            f"{filts[0]} {filts[3]}, " f"shoulders at {filts[1]} and " f"{filts[2]}"
        )


class TestSpec(RoverSpectrum):
    """mock spectrum class for tests"""
    instrument = "TEST"
    filters = {"test": 100}
    virtual_filters = {"test": 100}
    field_names = ()


# "mag" fields are power-of-10 magnitude, computed during multidex ingest
for field_name in ASDF_CART_COLS + ASDF_PHOT_COLS:
    field = models.FloatField(field_name.lower(), **B_N_I)
    field.contribute_to_class(ZSpec, field_name.lower())
    if field_name in ASDF_CART_COLS:
        magfield = models.FloatField(field_name.lower() + "mag", **B_N_I)
        magfield.contribute_to_class(ZSpec, field_name.lower() + "mag")
del field, magfield

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
