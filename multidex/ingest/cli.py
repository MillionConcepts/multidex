import datetime as dt
import os
from operator import attrgetter
from pathlib import Path

import django.db.models
import numpy as np
from fs.osfs import OSFS
import pandas as pd

from ingest.thumbpipe import default_thumbnailer

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "multidex.settings")
os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"

django.setup()

from plotter.models import ZSpec


# TODO: do this better
THUMB_PATH = "plotter/application/assets/browse/zcam/"


def looks_like_marslab(fn):
    if Path(fn).name.endswith("-marslab.csv"):
        return True
    return False


def looks_like_context(fn):
    name = Path(fn).name
    return (
        name.endswith(".png")
        and ("context" in name)
        and ("pixmap" not in name)
    )


def directory_of(path):
    if path.is_dir():
        return path
    return path.parent


def find_ingest_files(path, recursive=False):
    if recursive:
        tree = OSFS(directory_of(path))
        marslab_files = map(
            tree.getsyspath, filter(looks_like_marslab, tree.walk.files())
        )
        context_files = map(
            tree.getsyspath, filter(looks_like_context, tree.walk.files())
        )
    elif path.is_dir():
        marslab_files = map(str, filter(looks_like_marslab, path.iterdir()))
        context_files = map(str, filter(looks_like_context, path.iterdir()))
    else:
        marslab_files = [str(path)]
        context_files = map(str, filter(looks_like_context, path.iterdir()))
    return marslab_files, context_files


def y_to_bool(df, bool_fields):
    df.loc[:, bool_fields] = df.loc[:, bool_fields] == "Y"


ZCAM_BOOL_FIELDS = [
    field.name.upper()
    for field in ZSpec._meta.fields
    if isinstance(field, django.db.models.fields.BooleanField)
]


def process_context_files(context_files, nailpipe, make_thumbnails=True):
    context_df = pd.DataFrame(context_files, columns=["path"])
    context_df["stem"] = [
        Path(path).name
        for path in context_df["path"].str.replace(
            r"[ -]context.*", "", regex=True
        )
    ]
    context_df["eye"] = context_df["path"].str.extract(r"(left|right)")
    if make_thumbnails:
        print("making thumbnails")
        context_df["buffer"] = context_df["path"].apply(nailpipe.execute)
        context_df["save"] = False
    return context_df


def match_obs_images(marslab_file, context_df):
    file_stem = Path(marslab_file).name.replace("-marslab.csv", "")
    context_matches = context_df.loc[context_df["stem"] == str(file_stem)]
    context_df.loc[context_matches.index, "save"] = True
    obs_images = {}
    for record in context_matches[["stem", "eye"]].to_dict(orient="records"):
        eye = record["eye"]
        obs_images[eye] = record["stem"] + "-" + eye + "-thumb.jpg"
    return obs_images


ZSPEC_FIELD_NAMES = list(map(attrgetter("name"), ZSpec._meta.fields))


def save_relevant_thumbs(context_df):
    to_save = context_df.loc[context_df["save"] == True]
    thumb_path = THUMB_PATH
    for _, row in to_save.iterrows():
        filename = thumb_path + row["stem"] + "-" + row["eye"] + "-thumb.jpg"
        print("writing " + filename)
        with open(filename, "wb") as file:
            file.write(row["buffer"].getbuffer())


def ingest_multidex(path_or_file, *, recursive: "r" = False):
    """
    ingests zcam -marslab.csv files and context image thumbnails generated
    by asdf into a multidex database. expects all products in an observation
    to have matching filenames -- if you modify the filenames output by
    asdf, this script will likely fail.

    param path_or_file: marslab file or directory containing marslab files.
        looks for matching context images within the same directory.
    param recursive: attempts to ingest all marslab files and context images
        in directory tree, regardless of what specific file you passed it
    """
    path = Path(path_or_file)
    marslab_files, context_files = find_ingest_files(path, recursive)
    context_df = process_context_files(context_files, default_thumbnailer())
    for marslab_file in marslab_files:
        frame = pd.read_csv(marslab_file)
        if frame["INSTRUMENT"].iloc[0] != "ZCAM":
            print("skipping non-ZCAM file: " + marslab_file)
            continue
        print("ingesting spectra from " + Path(marslab_file).name)
        obs_images = str(match_obs_images(marslab_file, context_df))
        if obs_images != "{}":
            print("found matching images: " + obs_images)
        else:
            print("no matching images found")
        # regularize various ways people may have rendered metadata fields
        try:
            frame.columns = [
                col.upper().replace(" ", "_") for col in frame.columns
            ]
            y_to_bool(frame, ZCAM_BOOL_FIELDS)
            frame = frame.replace(["-", "", " "], np.nan)
            frame.columns = [col.lower() for col in frame.columns]
        except KeyboardInterrupt:
            raise
        except Exception as ex:
            print("failed on " + marslab_file + " reformat: " + str(ex))
            continue
        colors = []
        for _, row in frame.iterrows():
            row = row.dropna()
            relevant_indices = [
                ix for ix in row.index if ix in ZSPEC_FIELD_NAMES
            ]
            for filt in set(ZSpec.filters).intersection(row.index):
                row[filt] = float(row[filt])
            metadata = dict(row[relevant_indices]) | {
                "filename": Path(marslab_file).name,
                "images": obs_images,
                "ingest_time": dt.datetime.utcnow().isoformat()[:-7] + "Z",
                "min_count": row[
                    row.index.str.contains("count")
                ].astype(float).min(),
            }
            try:
                spectrum = ZSpec(**metadata)
                spectrum.clean()
                spectrum.save()
                colors.append(row["color"] + " " + str(row.get("feature")))
            except KeyboardInterrupt:
                raise
            except Exception as ex:
                print("failed on " + row["color"] + ": " + str(ex))
        print("successfully ingested " + ", ".join(colors))
    save_relevant_thumbs(context_df)
