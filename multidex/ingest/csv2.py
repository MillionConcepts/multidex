"""
simple tools to ingest simulated spectra produced by VISOR.
does not handle images at the moment.
"""
import os

import django
import pandas as pd

from multidex.multidex_utils import modeldict

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "multidex.multidex.settings")
os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"
django.setup()

from multidex.plotter.models import INSTRUMENT_MODEL_MAPPING


# TODO: how do I track full file provenance? It would be great if there were
#  a PRODUCT_ID or similar for the original CSVs...
def perform_ingest(csv_fn, instrument_code, *, dry_run=False, quiet=False):
    frame = pd.read_csv(csv_fn)
    if "UNITS" in frame.columns:
        if "R*" in frame["UNITS"].values:
            raise ValueError(
                "please do not attempt to ingest data valued in R*"
            )
    undesirable_columns = [
        "UNITS",
        "ID",
        "MODIFICATION_TIME",
        "INGEST_TIME",
        "MULTIDEX_VERSION",
        "FILTER_AVG",
        "STD_AVG",
        "REL_STD_AVG",
    ]
    for col in undesirable_columns:
        if col in frame.columns:
            frame = frame.drop(col, axis=1)
    model = INSTRUMENT_MODEL_MAPPING[instrument_code]
    for col in frame.columns:
        if col.lower() not in model.field_names:
            print(f"dropping {col} as apparently irrelevant")
            frame = frame.drop(col, axis=1)
    print("********* BEGINNING FRAME INGEST *********")
    if dry_run is True:
        print("******** DRY RUN ONLY *********")
    for ix, row in frame.iterrows():
        try:
            row = row.replace('-', float('nan')).dropna().to_dict()
            spectrum = model(
                **{key.lower(): value for key, value in row.items()}
            )
            spectrum.clean()
            if dry_run is not True:
                spectrum.save()
            if quiet is False:
                print(f"ingested {modeldict(spectrum)}")
        except KeyboardInterrupt:
            raise
        except Exception as ex:
            print(f"failed on row {ix}: {type(ex)}: {ex}")
