"""user-facing noninteractive script for ingesting simulated VISOR spectra"""

import fire

import multidex.ingest.visor


# tell clize to handle command line call
if __name__ == '__main__':
    fire.Fire(multidex.ingest.visor.perform_ingest)
