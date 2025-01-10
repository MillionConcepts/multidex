"""
user-facing noninteractive script for ingesting ZCAM spectra in marslab
format into multidex
"""

import fire

import multidex.ingest.zcam


# tell fire to handle command line call
if __name__ == '__main__':
    fire.Fire(multidex.ingest.zcam.perform_ingest)
