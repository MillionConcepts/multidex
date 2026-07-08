"""
user-facing noninteractive script for ingesting CCAM spectra in marslab
format into multidex
"""

import fire

import multidex.ingest.ccam.cli


# tell fire to handle command line call
if __name__ == '__main__':
    fire.Fire(multidex.ingest.ccam.cli.ingest_multidex)
