import os
from operator import attrgetter
from pathlib import Path

import django.db.models
import numpy as np
from fs.osfs import OSFS
import pandas as pd

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "multidex.settings")
os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"

django.setup()

from plotter.models import SSpec

SSPEC_FIELD_NAMES = list(map(attrgetter("name"), SSpec._meta.fields))


def looks_like_sspec(fn: str) -> bool:
    return bool(
        ("sspec" in fn) and not ("extended" in fn) and (fn.endswith(".csv"))
    )


def looks_like_context(fn: str) -> bool:
    return fn.endswith(".png") and ("scam" in fn) and ("pixmap" not in fn)


def directory_of(path: Path) -> str:
    if path.is_dir():
        return str(path)
    return str(path.parent)


def find_ingest_files(path: Path, recursive: bool = False):
    if recursive:
        tree = OSFS(directory_of(path))
        marslab_files = map(
            tree.getsyspath, filter(looks_like_sspec, tree.walk.files())
        )
        context_files = map(
            tree.getsyspath, filter(looks_like_context, tree.walk.files())
        )
    elif path.is_dir():
        marslab_files = filter(looks_like_sspec, map(str, path.iterdir()))
        context_files = filter(looks_like_context, map(str, path.iterdir()))
    else:
        marslab_files = [str(path)]
        context_files = filter(looks_like_context, map(str, path.iterdir()))
    return marslab_files, context_files


def y_to_bool(df, bool_fields):
    df.loc[:, bool_fields] = df.loc[:, bool_fields] == "Y"


SCAM_BOOL_FIELDS = [
    field.name.upper()
    for field in SSpec._meta.fields
    if isinstance(field, django.db.models.fields.BooleanField)
]


def process_context_files(context_files):
    context_df = pd.DataFrame(context_files, columns=["path"])
    context_df["save"] = False
    return context_df


def process_marslab_row(row, marslab_file, context_df):
    row = row.dropna()
    relevant_indices = [ix for ix in row.index if ix in SSPEC_FIELD_NAMES]
    for filt in set(SSpec.filters).intersection(row.index):
        row[filt] = float(row[filt])

    # see if there is a matching image
    if context_df is not None:
        sol = row['sol']
        seq_id = row['seq_id']
        img_file = '{sol:04d}_crm_{seq_id}'.format(sol=sol, seq_id=seq_id)
        context_matches = context_df.loc[context_df['path'].str.contains(img_file, case=False)]
        obs_image = None

        if len(context_matches.index) > 1:
            print("more than one!")
        if not context_matches.empty:
            for record in context_matches[["path"]].to_dict(orient="records"):
                obs_image = os.path.basename(record["path"])

    metadata = dict(row[relevant_indices]) | {
        "filename": Path(marslab_file).name,
        "images": [obs_image],
        # "ingest_time": dt.datetime.utcnow().isoformat()[:-7] + "Z",
        # "min_count": row[row.index.str.contains("count")].astype(float).min(),
    }

    # fix negative distances from darks
    # if metadata['target_distance'] < 0:
    #     metadata['target_distance'] = 5000
    # convert all target types to lower case
    # metadata['target_type_shot_specific'] = str.lower(metadata['target_type_shot_specific'])
    # get sclk from filename
    # sclk = int(metadata['name'].split('_')[1].replace('psv', ''))
    # metadata['sclk'] = sclk
    try:
        spectrum = SSpec(**metadata)
        spectrum.clean()
        spectrum.save()
        row_target = str(row.get("target"))
    except KeyboardInterrupt:
        raise
    except Exception as ex:
        print("failed on " + row_target + ": " + str(ex))
        return None
    return row_target


def format_for_multidex(frame):
    frame.columns = [col.upper().replace(" ", "_") for col in frame.columns]
    y_to_bool(frame, SCAM_BOOL_FIELDS)
    frame = frame.replace(["-", "", " "], np.nan)
    frame.columns = [col.lower() for col in frame.columns]
    return frame


def ingest_sspec_file(sspec_file, context_df):
    frame = pd.read_csv(sspec_file)
    if frame["INSTRUMENT"].iloc[0] != "SCAM":
        print("skipping non-SCAM file: " + sspec_file)
        return False, context_df
    print("ingesting spectra from " + Path(sspec_file).name)
    # regularize various ways people may have rendered metadata fields
    try:
        frame = format_for_multidex(frame)
    except KeyboardInterrupt:
        raise
    except Exception as ex:
        print("failed on " + sspec_file + " reformat: " + str(ex))
        return False, context_df
    colors = []
    for _, row in frame.iterrows():
        row_color = process_marslab_row(row, sspec_file, context_df)
        if row_color is not None:
            colors.append(row_color)
    print("successfully ingested " + ", ".join(colors))
    return True, context_df


def ingest_multidex(
    path_or_file, *, recursive: "r" = False, skip_thumbnails: "t" = False
):
    """
    ingests scam -sspec.csv files and context image thumbnails
    into a multidex database. expects all products in an observation
    to have matching filenames -- if you modify the filenames output by
    asdf or related applications, this script will likely fail, although you
    might be able to convince it if you really try.

    param path_or_file: sspec file or directory containing sspec files.
        looks for matching context images within the same directory.
    param recursive: attempts to ingest all sspec files and context images
        in directory tree, regardless of what specific file you passed it
    param skip_thumbnails: don't process context images or make thumbnails.
    """
    path = Path(path_or_file)
    sspec_files, context_files = find_ingest_files(path, recursive)
    if not skip_thumbnails:
        context_df = process_context_files(context_files)
    else:
        context_df = None
    successfully_ingested_sspec_files = []
    for sspec_file in sspec_files:
        successful, context_df = ingest_sspec_file(sspec_file, context_df)
        if successful:
            successfully_ingested_sspec_files.append(sspec_file)
