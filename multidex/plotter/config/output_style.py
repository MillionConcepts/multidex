from inspect import getmembers
import sys

from multidex.multidex_utils import patch_settings_from_module
# noinspection PyUnresolvedReferences
import multidex.plotter.config.user_output_style

GRAPH_SETTINGS = {
    "paper_bgcolor": "white",
    "margin": {"l": 62, "r": 255, "t": 27, "b": 33},
}
MARKER_SETTINGS = {}
COLORBAR_SETTINGS = {
    "tickfont": {"size": 19, "color": "black"},
    "tickangle": 0,
    "title": {"font": {"size": 24, "color": "black"}},
}
AXIS_SETTINGS = {
    "title": {"font": {"size": 25, "color": "black"}, "standoff": 31},
    "tickfont": {"size": 19, "color": "black"},
}
BASE_SIZE_SETTING = 1200
SCATTER_POINT_SCALE_SETTING = 1.5


patch_settings_from_module(
    getmembers(sys.modules[__name__]),
    "multidex.plotter.config.user_output_style"
)
