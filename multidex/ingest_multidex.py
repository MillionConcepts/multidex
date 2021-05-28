"""user-facing noninteractive pplot utility"""

from clize import run

import ingest.cli


# tell clize to handle command line call
if __name__ == '__main__':
    run(ingest.cli.ingest_multidex)
