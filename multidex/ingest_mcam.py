"""
user-facing noninteractive script for ingesting MCAM spectra in marslab
format into multidex
"""

import fire

import multidex.ingest.mcam


# tell fire to handle command line call
if __name__ == '__main__':
    fire.Fire(multidex.ingest.mcam.perform_ingest)
