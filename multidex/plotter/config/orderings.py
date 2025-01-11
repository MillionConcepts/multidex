"""
Special ordering/sorting rules for qualitative fields.

At present, this _only_ affects colorbar tick order.

This does not have a 'user' override module at present, but we may create one
if there is demand.
"""
from multidex_utils import freeze_nested_mapping

SPECIAL_ORDERINGS = {
    "ZCAM": {
        "formation": (
            "Maaz",
            "Seitah",
            "Delta",
            "Margin Unit",
            "Neretva Vallis",
            "Bright Angel",
            "Crater Rim",
        )
    }
}

SPECIAL_ORDERINGS = freeze_nested_mapping(SPECIAL_ORDERINGS)
