from django.db import models

from marslab.compatibility import (
    DERIVED_CAM_DICT,
    MERSPECT_MSL_COLOR_MAPPINGS, MERSPECT_M20_COLOR_MAPPINGS,
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


class MSpec(XSpec):
    instrument = "MCAM"

    def roi_hex_code(self) -> str:
        return MERSPECT_MSL_COLOR_MAPPINGS[self.color]


# bulk setup for each XCAM instrument
for spec_model in [ZSpec, MSpec]:
    # mappings from filter name to nominal band centers, in nm
    filters = DERIVED_CAM_DICT[spec_model.instrument]["filters"]
    virtual_filters = DERIVED_CAM_DICT[spec_model.instrument][
        "virtual_filters"
    ]
    # which real filters do virtual filters correspond to?
    virtual_filter_mapping = DERIVED_CAM_DICT[spec_model.instrument][
        "virtual_filter_mapping"
    ]
    # if we're only giving options for averaged filters,
    # what is the canonical list?
    canonical_averaged_filters = DERIVED_CAM_DICT[spec_model.instrument][
        "canonical_averaged_filters"
    ]

    # set up SQL fields for each filter
    for filter_name in filters.keys():
        mean_field, err_field = filter_fields_factory(filter_name)
        mean_field.contribute_to_class(spec_model, filter_name)
        err_field.contribute_to_class(spec_model, filter_name + "_err")
