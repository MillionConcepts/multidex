"""
user-facing noninteractive script for ingesting SuperCam spectra in marslab
format into multidex
"""

from clize import run

import ingest.scam.cli


# tell clize to handle command line call
if __name__ == '__main__':
    run(ingest.scam.cli.ingest_multidex)
