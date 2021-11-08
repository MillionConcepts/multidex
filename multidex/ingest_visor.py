"""user-facing noninteractive script for ingesting simulated VISOR spectra"""

from clize import run

import ingest.visor.cli


# tell clize to handle command line call
if __name__ == '__main__':
    run(ingest.visor.cli.ingest_multidex)
