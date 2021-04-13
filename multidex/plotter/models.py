import PIL
import fs
from django.db import models

from marslab.compatibility import (
    DERIVED_CAM_DICT,
    MERSPECT_MSL_COLOR_MAPPINGS,
    MERSPECT_M20_COLOR_MAPPINGS,
)
from plotter.model_prototypes import (
    XSpec,
    filter_fields_factory,
)


class ZSpec(XSpec):
    zoom = models.CharField(
        "Zoom Code", blank=True, null=True, max_length=10, db_index=True
    )
    instrument = "ZCAM"

    def roi_hex_code(self) -> str:
        return MERSPECT_M20_COLOR_MAPPINGS[self.color]

    # TODO: placeholder pending asdf
    def overlay_browse_file_info(self, image_directory: str) -> dict:
        files = self.image_files()
        images = {}
        for image_type, filename in files.items():
            images[image_type + "_file"] = filename
            with PIL.Image.open(
                fs.path.join(image_directory, filename)
            ) as image:
                images[image_type + "_size"] = image.size
        return images


class MSpec(XSpec):
    instrument = "MCAM"

    def roi_hex_code(self) -> str:
        return MERSPECT_MSL_COLOR_MAPPINGS[self.color]

    # TODO: hacky?
    def overlay_browse_file_info(self, image_directory: str) -> dict:
        files = self.image_files()
        images = {}
        for image_type, filename in files.items():
            if "roi" not in filename.lower():
                continue
            browse_filename = fs.path.splitext(filename)[0] + "_browse.jpg"
            images[image_type + "_file"] = browse_filename
            with PIL.Image.open(
                fs.path.join(image_directory, browse_filename)
            ) as image:
                images[image_type + "_size"] = image.size
        return images


# bulk setup for each XCAM instrument
for spec_model in [ZSpec, MSpec]:
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
