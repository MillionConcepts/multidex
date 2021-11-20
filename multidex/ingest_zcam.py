"""
user-facing noninteractive script for ingesting ZCAM spectra in marslab
format into multidex
"""

from clize import run

import ingest.zcam


# tell clize to handle command line call
if __name__ == '__main__':
    run(ingest.zcam.perform_ingest)
