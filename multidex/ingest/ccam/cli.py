import datetime as dt
import os
import re
from operator import attrgetter
from pathlib import Path

import django.db.models
import numpy as np
from fs.osfs import OSFS
import pandas as pd

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "multidex.settings")
os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"

django.setup()

from plotter.models import CSpec

CSPEC_FIELD_NAMES = list(map(attrgetter("name"), CSpec._meta.fields))


def looks_like_marslab(fn: str) -> bool:
    return bool(
        ("marslab" in fn) and not ("extended" in fn) and (fn.endswith(".csv"))
    )


def looks_like_context(fn: str) -> bool:
    return fn.endswith(".png") and ("context" in fn) and ("pixmap" not in fn)


def directory_of(path: Path) -> str:
    if path.is_dir():
        return str(path)
    return str(path.parent)


def find_ingest_files(path: Path, recursive: bool = False):
    if recursive:
        tree = OSFS(directory_of(path))
        marslab_files = map(
            tree.getsyspath, filter(looks_like_marslab, tree.walk.files())
        )
        context_files = map(
            tree.getsyspath, filter(looks_like_context, tree.walk.files())
        )
    elif path.is_dir():
        marslab_files = filter(looks_like_marslab, map(str, path.iterdir()))
        context_files = filter(looks_like_context, map(str, path.iterdir()))
    else:
        marslab_files = [str(path)]
        context_files = filter(looks_like_context, map(str, path.iterdir()))
    return marslab_files, context_files


def y_to_bool(df, bool_fields):
    df.loc[:, bool_fields] = df.loc[:, bool_fields] == "Y"


CCAM_BOOL_FIELDS = [
    field.name.upper()
    for field in CSpec._meta.fields
    if isinstance(field, django.db.models.fields.BooleanField)
]


def process_context_files(context_files):
    context_df = pd.DataFrame(context_files, columns=["path"])
    context_df["stem"] = [
        Path(path).name for path in context_df["path"].apply(asdf_stemmer)
    ]
    context_df["eye"] = context_df["path"].str.extract(r"(left|right)")
    context_df["save"] = False
    return context_df


def process_marslab_row(row, marslab_file):
    row = row.dropna()
    relevant_indices = [ix for ix in row.index if ix in CSPEC_FIELD_NAMES]
    for filt in set(CSpec.filters).intersection(row.index):
        row[filt] = float(row[filt])
    metadata = dict(row[relevant_indices]) | {
        "filename": Path(marslab_file).name,
        "images": [],
        #"ingest_time": dt.datetime.utcnow().isoformat()[:-7] + "Z",
        #"min_count": row[row.index.str.contains("count")].astype(float).min(),
    }
    try:
        spectrum = CSpec(**metadata)
        spectrum.clean()
        spectrum.save()
        row_color = row["color"] + " " + str(row.get("feature"))
    except KeyboardInterrupt:
        raise
    except Exception as ex:
        print("failed on " + row["color"] + ": " + str(ex))
        return None
    return row_color


def format_for_multidex(frame):
    frame.columns = [col.upper().replace(" ", "_") for col in frame.columns]
    y_to_bool(frame, CCAM_BOOL_FIELDS)
    frame = frame.replace(["-", "", " "], np.nan)
    frame.columns = [col.lower() for col in frame.columns]
    return frame


def ingest_marslab_file(marslab_file, context_df):
    frame = pd.read_csv(marslab_file)
    if frame["INSTRUMENT"].iloc[0] != "CCAM":
        print("skipping non-CCAM file: " + marslab_file)
        return False, context_df
    print("ingesting spectra from " + Path(marslab_file).name)
    # regularize various ways people may have rendered metadata fields
    try:
        frame = format_for_multidex(frame)
    except KeyboardInterrupt:
        raise
    except Exception as ex:
        print("failed on " + marslab_file + " reformat: " + str(ex))
        return False, context_df
    colors = []
    for _, row in frame.iterrows():
        row_color = process_marslab_row(row, marslab_file)
        if row_color is not None:
            colors.append(row_color)
    print("successfully ingested " + ", ".join(colors))
    return True, context_df


def ingest_multidex(
    path_or_file, *, recursive: "r" = False, skip_thumbnails: "t" = False
):
    """
    ingests zcam -marslab.csv files and context image thumbnails generated
    by asdf into a multidex database. expects all products in an observation
    to have matching filenames -- if you modify the filenames output by
    asdf or related applications, this script will likely fail, although you
    might be able to convince it if you really try.

    param path_or_file: marslab file or directory containing marslab files.
        looks for matching context images within the same directory.
    param recursive: attempts to ingest all marslab files and context images
        in directory tree, regardless of what specific file you passed it
    param skip_thumbnails: don't process context images or make thumbnails.
    """
    path = Path(path_or_file)
    marslab_files, context_files = find_ingest_files(path, recursive)
    context_df = None
    successfully_ingested_marslab_files = []
    for marslab_file in marslab_files:
        successful, context_df = ingest_marslab_file(marslab_file, context_df)
        if successful:
            successfully_ingested_marslab_files.append(marslab_file)
