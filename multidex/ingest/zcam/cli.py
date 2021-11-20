import os
import re
from operator import attrgetter
from pathlib import Path

import django.db.models
import numpy as np
from fs.osfs import OSFS
import pandas as pd

from ingest.zcam.thumbpipe import default_thumbnailer

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "multidex.settings")
os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"

django.setup()

from plotter.models import ZSpec

ASDF_STEM_PATTERN = re.compile(r"SOL\d{4}_zcam\d{5}_RSM\d{1,4}")
ASDF_LEGACY_STEM_PATTERN = re.compile(
    r"SOL\d{4}_SEQIDzcam\d{5}_SITE\d{1,4}_DRIVE\d{1,4}_RMS\d{1,4}_"
    r"ZOOM\d{1,3}"
)

# TODO: do this better, requires making people install this better
THUMB_PATH = "plotter/application/assets/browse/zcam/"

ZSPEC_FIELD_NAMES = list(map(attrgetter("name"), ZSpec._meta.fields))


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
        context_files = filter(
            looks_like_context, map(str, path.parent.iterdir())
        )
    return marslab_files, context_files


def y_to_bool(df, bool_fields):
    df.loc[:, bool_fields] = df.loc[:, bool_fields] == "Y"


ZCAM_BOOL_FIELDS = [
    field.name.upper()
    for field in ZSpec._meta.fields
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


def asdf_stemmer(asdf_fn: str):
    for pattern in (ASDF_STEM_PATTERN, ASDF_LEGACY_STEM_PATTERN):
        stem = re.search(pattern, asdf_fn)
        if stem is not None:
            return stem.group(0)
    return None


def match_obs_images(marslab_file, context_df):
    file_stem = asdf_stemmer(Path(marslab_file).name)
    context_matches = context_df.loc[context_df["stem"] == str(file_stem)]
    obs_images = {}
    for record in context_matches[["stem", "eye"]].to_dict(orient="records"):
        eye = record["eye"]
        obs_images[eye] = record["stem"] + "-" + eye + "-thumb.jpg"
    return obs_images, context_matches.index


def save_thumb(filename, row):
    print("writing " + filename)
    try:
        with open(filename, "wb") as file:
            file.write(row["buffer"].getbuffer())
        return True, None
    except KeyboardInterrupt:
        raise
    except Exception as ex:
        print(f"failed on {filename}: {type(ex)}: {ex}")
        return False, ex

def save_relevant_thumbs(context_df):
    to_save = context_df.loc[context_df["save"] == True]
    thumb_path = THUMB_PATH
    results = []
    for _, row in to_save.iterrows():
        filename = thumb_path + row["stem"] + "-" + row["eye"] + "-thumb.jpg"
        success, ex = save_thumb(filename, row)
        results.append(
            {
                "file": row["path"],
                "filetype": "thumb",
                "status": success,
                "exception": ex,
            }
        )
    return results


def process_marslab_row(row, marslab_file, obs_images):
    row = row.dropna()
    relevant_indices = [ix for ix in row.index if ix in ZSPEC_FIELD_NAMES]
    for filt in set(ZSpec.filters).intersection(row.index):
        row[filt] = float(row[filt])
    metadata = dict(row[relevant_indices]) | {
        "filename": Path(marslab_file).name,
        "images": obs_images,
        # "ingest_time": dt.datetime.utcnow().isoformat()[:-7] + "Z",
        "min_count": row[row.index.str.contains("count")].astype(float).min(),
    }
    try:
        spectrum = ZSpec(**metadata)
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
    y_to_bool(frame, ZCAM_BOOL_FIELDS)
    frame = frame.replace(["-", "", " "], np.nan)
    frame.columns = [col.lower() for col in frame.columns]
    return frame


def ingest_marslab_file(marslab_file, context_df):
    frame = pd.read_csv(marslab_file)
    if frame["INSTRUMENT"].iloc[0] != "ZCAM":
        print("skipping non-ZCAM file: " + marslab_file)
        return False, "does not appear to be a ZCAM file", context_df
    if frame["COLOR"].eq("-").all():
        print(f"no spectra in {marslab_file}, skipping")
        return False, "no spectra in file", context_df
    print("ingesting spectra from " + Path(marslab_file).name)
    if context_df is not None:
        obs_images, match_index = match_obs_images(marslab_file, context_df)
        if obs_images != {}:
            print(f"found matching images: {obs_images}")
            context_df.loc[match_index, "save"] = True
        else:
            print("no matching images found")
    else:
        obs_images = {}
    # regularize various ways people may have rendered metadata fields
    try:
        frame = format_for_multidex(frame)
    except KeyboardInterrupt:
        raise
    except Exception as ex:
        print("failed on " + marslab_file + " reformat: " + str(ex))
        return False, ex, context_df
    colors = []
    for _, row in frame.iterrows():
        row_color = process_marslab_row(row, marslab_file, obs_images)
        if row_color is not None:
            colors.append(row_color)
    print("successfully ingested " + ", ".join(colors))
    return True, None, context_df


def ingest_marslab_files(
    path_or_file,
    *,
    recursive: "r" = False,
    skip_thumbnails: "t" = False,
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
    if not skip_thumbnails:
        context_df = process_context_files(context_files)
    else:
        context_df = None
    results = []
    for file in marslab_files:
        success, ex, context_df = ingest_marslab_file(file, context_df)
        results.append(
            {
                "file": file,
                "filetype": "marslab",
                "status": success,
                "exception": ex,
            }
        )
    if not skip_thumbnails:
        print("making thumbnails")
        nailpipe = default_thumbnailer()
        context_df = context_df.loc[context_df["save"]].copy()
        context_df["buffer"] = context_df["path"].apply(nailpipe.execute)
    thumb_results = save_relevant_thumbs(context_df)
    return results + thumb_results
