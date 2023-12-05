"""
user-facing noninteractive script for ingesting MCAM spectra in marslab
format into multidex
"""

import fire

import ingest.mcam


# tell clize to handle command line call
# if __name__ == '__main__':
#     fire.Fire(ingest.mcam.perform_ingest)

ingest.mcam.perform_ingest("/home/michael/Downloads")
