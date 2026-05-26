"""
user-facing noninteractive script for ingesting MER Pancam spectra in marslab
format into multidex
"""

import fire

import multidex.ingest.pcam


# tell fire to handle command line call
if __name__ == '__main__':
    fire.Fire(multidex.ingest.pcam.perform_ingest)
