"""user-facing noninteractive script for ingesting simulated VISOR spectra"""

from clize import run

import ingest.csv.cli


# tell clize to handle command line call
if __name__ == '__main__':
    run(ingest.csv.cli.ingest_csv)
