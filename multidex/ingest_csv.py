"""user-facing noninteractive script for ingesting flat CSV files"""

from clize import run

import ingest.csv2


# tell clize to handle command line call
if __name__ == '__main__':
    run(ingest.csv2.perform_ingest)
