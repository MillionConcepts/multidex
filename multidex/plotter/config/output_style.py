from pathlib import Path

import multidex.plotter.application
from multidex.multidex_utils import patch_mappings_from_module_members
# noinspection PyUnresolvedReferences
import multidex.plotter.config.user_output_style

font_root = Path(
    multidex.plotter.application.__file__
).parent / "assets" / "fonts"

FONT_SETTINGS = {
    "LABEL_FONT_PATH": font_root / "TitilliumWeb-Light.ttf",
    "TICK_FONT_PATH": font_root / "TitilliumWeb-Light.ttf",
    "FITLINE_FONT_PATH": font_root / "TitilliumWeb-Bold.ttf"
}

SIZE_SETTINGS = {
    "LABEL_TEXT_SIZE": 30,
    "TICK_TEXT_SIZE": 24,
    "FITLINE_TEXT_SIZE": 26
}

patch_mappings_from_module_members(
    (FONT_SETTINGS, SIZE_SETTINGS),
    "multidex.plotter.config.user_output_style"
)
