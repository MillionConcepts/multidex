"""
assemble objects defining interface properties for various metadata fields
and calculation types. fields must be defined in this module in order to
be accessible to users for search and plotting.
"""
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
    "landform_type",
    "libs_before",
    "location",
    "member",
    "morphology",
    "name",
    "notes",
    "outcrop",
    "pixl",
    "rock_surface",
    "scam",
    "seq_id",
    "soil_color",
    "soil_location",
    "srlc_spec",
    "target",
    "target_type",
    "target_type_shot_specific",
    "workspace",
    "wtsn",
    "zoom",
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
    "temp",
)
# properties computed at runtime from metadata
CALCULATED_FIELDS = ("filter_avg", "err_avg", "rel_err_avg")


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