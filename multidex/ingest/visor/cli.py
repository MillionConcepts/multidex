"""
simple tools to ingest simulated spectra produced by VISOR.
does not handle images at the moment.
"""

import datetime as dt
import re
from functools import partial
import os
from pathlib import Path
from typing import Callable

import django
from fs.osfs import OSFS
import numpy as np
import pandas as pd

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "multidex.settings")
os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"
django.setup()

from plotter.models import INSTRUMENT_MODEL_MAPPING


def directory_of(path: Path) -> str:
    if path.is_dir():
        return str(path)
    return str(path.parent)


def find_ingest_files(
    path: Path,
    recursive: bool = False,
    csv_predicate: Callable[[str], bool] = lambda x: False,
    image_predicate: Callable[[str], bool] = lambda x: False,
):
    if recursive:
        tree = OSFS(directory_of(path))
        csv_files = map(
            tree.getsyspath, filter(csv_predicate, tree.walk.files())
        )
        image_files = map(
            tree.getsyspath, filter(image_predicate, tree.walk.files())
        )
    elif path.is_dir:
        csv_files = filter(csv_predicate, map(str, path.iterdir()))
        image_files = filter(image_predicate, map(str, path.iterdir()))
    else:
        csv_files = [str(path)]
        image_files = filter(image_predicate, map(str, path.iterdir()))
    return csv_files, image_files


def visor_stemmer(fn: str):
    return re.sub(r"(_simulated)?\..*", "", Path(fn).name)


# ignoring images for now. really need more examples to
# make this worth it.

# def process_image_files(image_files):
#     image_df = pd.DataFrame(image_files, columns=["path"])
#     image_df["stem"] = [
#         stem for stem in image_df["path"].apply(visor_stemmer)
#     ]
#     image_df["save"] = False
#     return image_df

# def match_images(csv_fn, image_df):
#     file_stem = visor_stemmer(csv_fn)
#     image_matches = image_df.loc[image_df["stem"] == str(file_stem)]
#     if len(image_matches) > 1:
#         raise ValueError("too many matching images")
#     return  file_stem + "-thumb.jpg", image_matches.index


def looks_like_visor_file(path, instrument_code):
    brief_name = INSTRUMENT_MODEL_MAPPING[
        instrument_code
    ].instrument_brief_name
    return str(path).endswith(f"simulated_{brief_name}.csv")


def process_visor_file(visor_fn, instrument_code):
    visor_lines = [
        line.strip().split(",", maxsplit=1)
        for line in open(visor_fn).readlines()
    ]
    visor_dict = {k: v for k, v in visor_lines}
    model = INSTRUMENT_MODEL_MAPPING[instrument_code]
    present_filters = set(model.filters).intersection(visor_dict.keys())
    if len(present_filters) == 0:
        raise ValueError(
            "No filters in this file match this instrument's filters."
        )
    model_dict = {}
    for filt in present_filters:
        # VISOR simulated columns for filter rows are:
        # filter name, wavelength, solar illuminated reflectance, reflectance
        # only the last is wanted here
        model_dict[filt.lower()] = float(visor_dict[filt].split(",")[2])
        model_dict[filt.lower() + "_err"] = 0
    model_dict[
        "name"
    ] = f"{visor_dict['Sample Name']} - {visor_dict['Sample ID']}"
    model_dict["filename"] = Path(visor_fn).name
    model_dict["ingest_time"] = dt.datetime.utcnow().isoformat()
    model_dict["feature"] = "lab spectrum"
    model_dict["color"] = "red"
    model_dict["incidence_angle"] = 0
    try:
        spectrum = model(**model_dict)
        spectrum.clean()
        spectrum.save()
    except KeyboardInterrupt:
        raise
    except Exception as ex:
        print(f"failed: {type(ex)}({str(ex)})")
        return None
    return spectrum
