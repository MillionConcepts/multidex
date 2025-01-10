"""user-facing noninteractive script for ingesting simulated VISOR spectra"""

import fire

import ingest.visor


# tell clize to handle command line call
if __name__ == '__main__':
    fire.Fire(ingest.visor.perform_ingest)
