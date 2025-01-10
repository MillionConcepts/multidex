from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

# these import backflips are intended to maintain compatibility with
# "cd multidex && python multidex.py" workflows now that multidex is a real
# package
mdex_cli_script_spec = spec_from_file_location(
    "mdex_cli_script", str((Path(__file__).parent / "multidex.py").absolute())
)
mdex_cli_script = module_from_spec(mdex_cli_script_spec)
mdex_cli_script_spec.loader.exec_module(mdex_cli_script)
sys.modules["mdex_cli_script"] = mdex_cli_script
# noinspection PyUnresolvedReferences
from mdex_cli_script import multidex_run_hook

__version__ = "0.10.0"
