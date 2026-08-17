"""
Tables of UI properties for the various columns of an xCAM data set.
Columns not mentioned in these tables will not be visible in the
MultiDEx API.
"""

from dataclasses import dataclass
from enum import Enum


# spatial / per-ROI photometry fields from asdf
ASDF_CART_COLS = [
    f"{eye}_{suffix}"
    for eye in ["LEFT", "RIGHT"]
    for suffix in ["H", "W", "HW", "A", "D"]
]

ASDF_PHOT_COLS = [
    f"{eye}_{suffix}"
    for eye in ["LEFT", "RIGHT"]
    for suffix in ["I", "E", "P"]
]

# TODO: figure out how to implement decomposition parameter
#  controls; maybe this doesn't go here, it's a separate interface,
#  something like that
REDUCTION_OP_FIELDS = ("PCA",)

# metadata fields we should treat as qualitative / categorical
QUALITATIVE_METADATA_FIELDS = (
    "analysis_name",
    "color",
    "compression",
    "distance",
    "drive",
    "feature",
    "feature_subtype",
    "filename",
    "float",
    "formation",
    "grain_size",
    "group",
    "id",
    "lab_spectrum_type",
    "landform_type",
    "libs_before",
    "location",
    "member",
    "morphology",
    "name",
    "notes",
    "outcrop",
    "pixl",
    "rock_class",
    "rock_surface",
    "scam",
    "seq_id",
    "site",
    "soil_class",
    "soil_color",
    "soil_location",
    "srlc_spec",
    "target",
    "target_type",
    "target_type_shot_specific",
    "type_of_product",
    "wtsn",
    "zoom",
    # quality flags computed during index for asdf-generated spatial data
    "spatial_flag",
    "phot_flag",
    # rc properties
    "rc_caltarget_file",
    "rc_seq_id",
    # caltarget roi only
    "caltarget_element",
    # supercam
    "red_rows",
    "powerfail",
    "tdb_name",
    "workspace",
)

# metadata fields we should treat as quantitative / continuous
QUANTITATIVE_METADATA_FIELDS = (
    "compression_quality",
    "emission_angle",
    "exposure",
    "file_timestamp",
    "focal_distance",
    "incidence_angle",
    "instrument_azimuth",
    "instrument_elevation",
    "lat",
    "lmst",
    "lon",
    "l_s",
    "ltst",
    "min_count",
    "modification_time",
    "odometry",
    "phase_angle",
    "raster_location",
    "rc_ltst",
    "rover_elevation",
    "rsm",
    "rsm_l",
    "rsm_r",
    "sclk",
    "sol",
    "solar_azimuth",
    "solar_elevation",
    "target_distance",
    "target_elevation",
    "target_lat",
    "target_lon",
    "tau",
    "temp",
    # rc-file fields
    "rc_sol",
    "rc_solar_azimuth",
    "rc_incidence_angle",
    "rc_azimuth_angle",
    "rc_emission_angle",
    "rc_scaling_factor",
    "rc_uncertainty",
    # rc data field
    "azimuth_angle",
    *[c.lower() for c in ASDF_CART_COLS],
    *[f"{c.lower()}mag" for c in ASDF_CART_COLS],
    *[c.lower() for c in ASDF_PHOT_COLS],
    # supercam
    "uv_rows",
    "vio_rows",
    "rsm_azimuth",
    "rsm_elevation",
    "t_integ_real",
    "p750",
    "p1400",
    "p1900",
    "p2300",
    "saturation",
    "focus_position_mm",
)

# properties computed at runtime from metadata
CALCULATED_FIELDS = (
    "filter_avg",
    "std_avg",
    "rel_std_avg",
    "l_rmad",
    "r_rmad",
    "l_rstd",
    "r_rstd",
    "mean_wrasd",
    "max_wrasd",
    "mean_wasd",
    "max_wasd",
    "p2p",
)

# fields from the above categories we would like users to search but not graph
UNGRAPHABLE_FIELDS = ("color", "id", "seq_id", "name", "analysis_name", "target")


# nothing below this point should need to be modified simply to add more fields
class ValueType(Enum):
    QUAL  = "qual"
    QUANT = "quant"

class FieldType(Enum):
    ATTRIBUTE = "attribute"
    METHOD = "method"
    COMPUTED = "computed"
    DECOMPOSITION = "decomposition"

@dataclass(frozen=True, slots=True, kw_only=True)
class FieldProperties:
    field:      str        # the name of the field
    value_type: ValueType
    type:       FieldType = FieldType.ATTRIBUTE
    arity:      int = 0

    # backcompat, to be eliminated
    @property
    def value(self) -> str:
        return self.field

    # backcompat / space for "friendlier" labels
    @property
    def label(self) -> str:
        return self.field


QUALITATIVE_METADATA_PROPERTIES = tuple(
    FieldProperties(field = field, value_type = ValueType.QUAL)
    for field in QUALITATIVE_METADATA_FIELDS
)

QUANTITATIVE_METADATA_PROPERTIES = tuple(
    FieldProperties(field = field, value_type = ValueType.QUANT)
    for field in QUANTITATIVE_METADATA_FIELDS
)

CALCULATED_PROPERTIES = tuple(
    FieldProperties(field = field, value_type = ValueType.QUANT, type = FieldType.COMPUTED)
    for field in CALCULATED_FIELDS
)
REDUCTION_OP_PROPERTIES = tuple(
    FieldProperties(field = field, value_type = ValueType.QUANT, type = FieldType.DECOMPOSITION)
    for field in REDUCTION_OP_FIELDS
)

# shorthand
def spop(field: str, arity: int) -> FieldProperties:
    return FieldProperties(field = field, value_type = ValueType.QUANT, type = FieldType.METHOD, arity = arity)

# spectrum operation / band math functions (band depth, etc.) require the additional 'arity' parameter
SPECTRUM_OP_PROPERTIES = (
    spop("ref", 1),
    spop("slope", 2),
    spop("band_avg", 2),
    spop("band_max", 2),
    spop("band_min", 2),
    spop("ratio", 2),
    spop("band_depth", 3),
)

del spop

METADATA_PROPERTIES = (
    QUANTITATIVE_METADATA_PROPERTIES + QUALITATIVE_METADATA_PROPERTIES
)

DYNAMIC_PROPERTIES = (
    SPECTRUM_OP_PROPERTIES + REDUCTION_OP_PROPERTIES + CALCULATED_PROPERTIES
)
