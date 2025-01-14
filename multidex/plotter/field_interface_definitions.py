"""
assemble objects defining interface properties for various metadata fields
and calculation types. fields must be defined in this module in order to
be accessible to users for search and plotting.
"""
from itertools import product

# spatial / per-ROI photometry fields from asdf
ASDF_SPATIAL_SUFFIXES = ('H', 'W', 'HW', 'A', 'D')
ASDF_CART_COLS = [
    f'{eye}_{suffix}'
    for eye, suffix in product(('LEFT', 'RIGHT'), ASDF_SPATIAL_SUFFIXES)
]
ASDF_PHOT_SUFFIXES = ('I', 'E', 'P')
ASDF_PHOT_COLS = [
    f'{eye}_{suffix}'
    for eye, suffix in product(('LEFT', 'RIGHT'), ASDF_PHOT_SUFFIXES)
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
    "feature",
    "feature_subtype",
    "float",
    "formation",
    "grain_size",
    "group",
    "lab_spectrum_type",
    "libs_before",
    "location",
    "member",
    "name",
    "notes",
    "outcrop",
    "pixl",
    "rock_class",
    "scam",
    "seq_id",
    "soil_class",
    "soil_color",
    "srlc_spec",
    "target",
    "target_type",
    "target_type_shot_specific",
    "wtsn",
    "zoom",
    # quality flags computed during index for asdf-generated spatial data
    "spatial_flag",
    "phot_flag",
    # rc properties
    "rc_seq_id",
    # caltarget roi only
    "caltarget_element",
)
# metadata fields we should treat as quantitative / continuous
QUANTITATIVE_METADATA_FIELDS = (
    "compression_quality",
    "emission_angle",
    "exposure",
    "focal_distance",
    "incidence_angle",
    "instrument_azimuth",
    "instrument_elevation",
    "lat",
    "lmst",
    "lon",
    "ltst",
    "min_count",
    "odometry",
    "phase_angle",
    "raster_location",
    "rc_ltst",
    "rover_elevation",
    "rsm",
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
)
# properties computed at runtime from metadata
CALCULATED_FIELDS = (
    "filter_avg",
    "std_avg",
    "rel_std_avg",
    "l_rmad",
    "r_rmad",
    'l_rstd',
    'r_rstd',
    "max_wrasd"
)


# assemble property records: these statements should not need to be modified
# simply to add new fields
def make_property_records(fields, base_properties):
    return tuple(({"value": field} | base_properties for field in fields))


QUALITATIVE_METADATA_PROPERTIES = make_property_records(
    QUALITATIVE_METADATA_FIELDS, {"value_type": "qual"}
)
QUANTITATIVE_METADATA_PROPERTIES = make_property_records(
    QUANTITATIVE_METADATA_FIELDS, {"value_type": "quant"}
)
CALCULATED_PROPERTIES = make_property_records(
    CALCULATED_FIELDS, {"value_type": "quant", "type": "computed"}
)
REDUCTION_OP_PROPERTIES = make_property_records(
    REDUCTION_OP_FIELDS, {"value_type": "quant", "type": "decomposition"}
)

# spectrum operation / band math functions (band depth, etc.)
# are always-already defined as records to give arity
SPECTRUM_OP_BASE_PROPERTIES = {"type": "method", "value_type": "quant"}
SPECTRUM_OP_PROPERTIES = (
    {"value": "ref", "arity": 1},
    {"value": "slope", "arity": 2},
    {"value": "band_avg", "arity": 2},
    {"value": "band_max", "arity": 2},
    {"value": "band_min", "arity": 2},
    {"value": "ratio", "arity": 2},
    {"value": "band_depth", "arity": 3},
)
for op in SPECTRUM_OP_PROPERTIES:
    op |= SPECTRUM_OP_BASE_PROPERTIES


# fields from the above categories we would like users to search but not graph
UNGRAPHABLE_FIELDS = ("color", "seq_id", "name", "analysis_name", "target")


# assemble categories, add labels, etc. nothing below this comment should
# need to be modified simply to add new fields.
METADATA_PROPERTIES = (
    QUANTITATIVE_METADATA_PROPERTIES + QUALITATIVE_METADATA_PROPERTIES
)
DYNAMIC_PROPERTIES = (
    SPECTRUM_OP_PROPERTIES + REDUCTION_OP_PROPERTIES + CALCULATED_PROPERTIES
)
for prop_record in METADATA_PROPERTIES + DYNAMIC_PROPERTIES:
    if "label" not in prop_record.keys():
        prop_record["label"] = prop_record["value"]
    if "type" not in prop_record.keys():
        prop_record["type"] = "attribute"
