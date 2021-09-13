"""
user-facing noninteractive script for ingesting ZCAM spectra in marslab
format into multidex
"""

from clize import run

import ingest.zcam.cli


# tell clize to handle command line call
if __name__ == '__main__':
    run(ingest.zcam.cli.ingest_multidex)
