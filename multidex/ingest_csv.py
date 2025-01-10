"""user-facing noninteractive script for ingesting flat CSV files"""

import fire

import ingest.csv2


# tell fire to handle command line call
if __name__ == '__main__':
    fire.Fire(ingest.csv2.perform_ingest)
