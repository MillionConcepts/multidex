from pathlib import Path
from types import MappingProxyType

import PIL
from django.db import models
from marslab.compat.mertools import (
    MERSPECT_M20_COLOR_MAPPINGS,
    MERSPECT_MSL_COLOR_MAPPINGS,
)

from marslab.compat.xcam import DERIVED_CAM_DICT
from plotter.model_prototypes import XSpec, filter_fields_factory, B_N_I


class ZSpec(XSpec):
    zoom = models.CharField("Zoom Code", max_length=10, **B_N_I)
    # shared target identifier with other instruments, usually null
    target = models.CharField("Target", max_length=60, **B_N_I)
    # rover motion counter for the mast -- for repointed stereo
    # observations, this is the first RMS in the sequence
    rms = models.IntegerField("RMS", **B_N_I)
    # timestamp of file if automatically produced by asdf
    file_timestamp = models.CharField(max_length=30, null=True)
    compression = models.CharField("Compression", max_length=40, **B_N_I)
    compression_quality = models.IntegerField("Compression Quality", **B_N_I)
    float = models.CharField("Rock Float Status", max_length=20, **B_N_I)
    rock_surface = models.CharField("Rock Surface", max_length=50, **B_N_I)
    grain_size = models.CharField("Grain Size", max_length=20, **B_N_I)
    soil_location = models.CharField("Soil Location", max_length=50, **B_N_I)
    soil_color = models.CharField("Soil Color", max_length=25, **B_N_I)
    morphology = models.CharField("Morphology", max_length=25, **B_N_I)
    distance = models.CharField("Distance", max_length=20, **B_N_I)
    location = models.CharField("Location", max_length=60, **B_N_I)
    landform_type = models.CharField("Landform Type", max_length=25, **B_N_I)
    workspace = models.CharField("Workspace", max_length=40, **B_N_I)
    analysis_name = models.CharField(
        "analysis / ROI set name", max_length=30, **B_N_I
    )
    min_count = models.IntegerField("Minimum Pixel Count", **B_N_I)

    instrument = "ZCAM"
    instrument_brief_name = "Mastcam-Z"

    color_mappings = MERSPECT_M20_COLOR_MAPPINGS | {"black": "#000000"}

    def roi_hex_code(self) -> str:
        return self.color_mappings[self.color]

    def overlay_browse_file_info(self, image_directory: str) -> dict:
        files = self.image_files()
        images = {}
        for image_type, filename in files.items():
            images[image_type + "_file"] = filename
            with PIL.Image.open(Path(image_directory, filename)) as image:
                images[image_type + "_size"] = image.size
        return images


class MSpec(XSpec):
    # large-to-small taxonomic categories for rock clusters
    group = models.CharField("Group", **B_N_I, max_length=50)
    formation = models.CharField("Formation", **B_N_I, max_length=50)
    member = models.CharField("Member", **B_N_I, max_length=50)
    notes = models.CharField("Notes", **B_N_I, max_length=100)
    float = models.CharField("floating vs. in-place", **B_N_I, max_length=15)
    feature_subtype = models.CharField(
        "feature subtype", **B_N_I, max_length=45
    )
    rock_class = models.CharField("rock class", **B_N_I, max_length=45)
    soil_class = models.CharField("soil class", **B_N_I, max_length=45)
    instrument = "MCAM"
    instrument_brief_name = "Mastcam"
    color_mappings = MERSPECT_MSL_COLOR_MAPPINGS | {"black": "#000000"}

    def roi_hex_code(self) -> str:
        return self.color_mappings[self.color]

    # TODO: hacky?
    #  yes, it absolutely is. Deprecate this completely, just fold direct
    #  object introspection back in, as is currently happening in
    #  graph.make_mspec_browse_image_components
    def overlay_browse_file_info(self, image_directory: str) -> dict:
        files = self.image_files()
        images = {}
        for image_type, filename in files.items():
            if "roi" not in filename.lower():
                continue
            browse_filename = Path(filename).stem + "_browse.jpg"
            images[image_type + "_file"] = browse_filename
            with PIL.Image.open(
                Path(image_directory, browse_filename)
            ) as image:
                images[image_type + "_size"] = image.size
        return images


class CSpec(XSpec):
    target = models.CharField("Target", **B_N_I, max_length=50)
    type_of_product = models.CharField("Type of Product", **B_N_I, max_length=50)
    distance = models.CharField("Distance", max_length=20, **B_N_I)
    lmst = models.TimeField("Local Mean Solar Time", **B_N_I)
    num_spectra = models.IntegerField("Number of Spectra", **B_N_I)
    shots_averaged = models.IntegerField("Shots Averaged", **B_N_I)
    observation_type = models.CharField("Type of Observation", **B_N_I)
    exposure = models.CharField("Exposure", **B_N_I)
    target_type = models.CharField("Target Type", **B_N_I)
    raster_location = models.IntegerField("Raster Location", **B_N_I)
    instrument_elevation = models.FloatField("Instrument Elevation (deg)", **B_N_I)
    instrument_azimuth = models.FloatField("Instrument Azimuth (deg)", **B_N_I)
    solar_azimuth = models.FloatField("Solar Azimuth (deg)", **B_N_I)
    solar_elevation = models.FloatField("Solar Elevation (deg)", **B_N_I)
    temp = models.FloatField("Instrument Temperature (C)", **B_N_I)
    libs_before = models.BooleanField("LIBS before pass", **B_N_I)
    notes = models.CharField("Notes", **B_N_I, max_length=100)

    instrument = "CCAM"
    instrument_brief_name = "ChemCam"


ZCAM_COREGISTERED_INSTRUMENTS = (
    "SCAM LIBS",
    "SCAM VISIR",
    "SCAM RMI",
    "SCAM Raman",
    "PIXL",
    "SHERLOC",
    "WATSON",
)
for instrument in ZCAM_COREGISTERED_INSTRUMENTS:
    coreg_field = models.BooleanField(instrument, **B_N_I)
    coreg_field.contribute_to_class(
        ZSpec, "_".join(instrument.lower().split(" "))
    )

# bulk setup for each XCAM instrument
for spec_model in [ZSpec, MSpec, CSpec]:
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
        err_field.contribute_to_class(spec_model, filt.lower() + "_err")

    # add fields to each model
    setattr(
        spec_model,
        "field_names",
        [field.name for field in spec_model._meta.fields],
    )

# for automated model selection
INSTRUMENT_MODEL_MAPPING = MappingProxyType({"ZCAM": ZSpec, "MCAM": MSpec, "CCAM": CSpec})
