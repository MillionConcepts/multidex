import sys
from inspect import getmembers

from multidex_utils import patch_settings_from_module
import plotter.config.user_output_style

GRAPH_SETTINGS = {
    "paper_bgcolor": "white",
    "margin": {"l": 60, "r": 285, "t": 30, "b": 30},
}
MARKER_SETTINGS = {}
COLORBAR_SETTINGS = {
    "tickfont": {"size": 18},
    "tickangle": 0,
    "title": {"font": {"size": 24, "color": "black"}},
}
AXIS_SETTINGS = {
    "title": {"font": {"size": 24, "color": "black"}, "standoff": 30},
    "tickfont": {"size": 18, "color": "black"},
}
BASE_SIZE_SETTING = 1200
SCATTER_POINT_SCALE_SETTING = 1.5


patch_settings_from_module(
    getmembers(sys.modules[__name__]), "plotter.config.user_output_style"
)
