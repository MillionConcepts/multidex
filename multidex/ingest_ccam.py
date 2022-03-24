"""
user-facing noninteractive script for ingesting CCAM spectra in marslab
format into multidex
"""

from clize import run

import ingest.ccam.cli


# tell clize to handle command line call
if __name__ == '__main__':
    run(ingest.ccam.cli.ingest_multidex)
