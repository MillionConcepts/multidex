from inspect import getmembers
from pathlib import Path
import sys

import multidex.plotter.application
from multidex.multidex_utils import patch_settings_from_module
# noinspection PyUnresolvedReferences
import multidex.plotter.config.user_output_style

font_root = Path(
    multidex.plotter.application.__file__
).parent / "assets" / "fonts"
FONT_PATH = font_root / "TitilliumWeb-Light.ttf"
FONT_PATH_BOLD = font_root / "TitilliumWeb-Bold.ttf"

LABEL_TEXT_SIZE = 30
TICK_TEXT_SIZE = 24
FITLINE_TEXT_SIZE = 26

patch_settings_from_module(
    getmembers(sys.modules[__name__]),
    "multidex.plotter.config.user_output_style"
)
