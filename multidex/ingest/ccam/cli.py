import os
import shutil
from operator import attrgetter
from pathlib import Path

import django.db.models
import numpy as np
from fs.osfs import OSFS
import pandas as pd
import warnings

# warnings.filterwarnings("error")

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "multidex.settings")
os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"

django.setup()

from plotter.models import CSpec

CSPEC_FIELD_NAMES = list(map(attrgetter("name"), CSpec._meta.fields))

# set up absolute path to thumbs so this can be run from anywhere
ABS_PATH = os.path.dirname(__file__)
REL_THUMB_PATH = "plotter/application/assets/browse/ccam/"
THUMB_PATH = os.path.join(Path(ABS_PATH).parents[1], REL_THUMB_PATH)


def looks_like_cspec(fn: str) -> bool:
    return bool(
        ("cspec" in fn) and not ("extended" in fn) and (fn.endswith(".csv"))
    )


def looks_like_context(fn: str) -> bool:
    return fn.endswith(".jpg") and ("ccam" in fn) and ("pixmap" not in fn)


def directory_of(path: Path) -> str:
    if path.is_dir():
        return str(path)
    return str(path.parent)


def find_ingest_files(path: Path, recursive: bool = False):
    if recursive:
        tree = OSFS(directory_of(path))
        marslab_files = map(
            tree.getsyspath, filter(looks_like_cspec, tree.walk.files())
        )
        context_files = map(
            tree.getsyspath, filter(looks_like_context, tree.walk.files())
        )
    elif path.is_dir():
        marslab_files = filter(looks_like_cspec, map(str, path.iterdir()))
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
    context_df["save"] = False
    return context_df


def process_marslab_row(row, marslab_file, context_df):
    row = row.dropna()
    relevant_indices = [ix for ix in row.index if ix in CSPEC_FIELD_NAMES]
    for filt in set(CSpec.filters).intersection(row.index):
        row[filt] = float(row[filt])

    # see if there is a matching image
    if context_df is not None:
        sol = row['sol']
        seq_id = row['seq_id']
        img_file = '{sol:04d}_crm_{seq_id}'.format(sol=sol, seq_id=seq_id)
        context_matches = context_df.loc[context_df['path'].str.contains(img_file, case=False)]
        obs_image = None

        if len(context_matches.index) > 1:
            print("more than one image for this spectra?!  " + img_file)
            # try adding the target name
            target = row['target']
            img_file = '{sol:04d}_crm_{seq_id}_{target}'.format(sol=sol, seq_id=seq_id, target=target)
            context_matches = context_df.loc[context_df['path'].str.contains(img_file, case=False)]
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
    if metadata['target_distance'] < 0:
        metadata['target_distance'] = 5000

    # convert all target types to lower case
    if 'target_type_shot_specific' in metadata:
        metadata['target_type_shot_specific'] = str.lower(metadata['target_type_shot_specific'])

    # get sclk from filename
    sclk = int(metadata['name'].split('_')[1].replace('psv', ''))
    metadata['sclk'] = sclk
    try:
        spectrum = CSpec(**metadata)
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
    y_to_bool(frame, CCAM_BOOL_FIELDS)
    frame = frame.replace(["-", "", " "], np.nan)
    frame.columns = [col.lower() for col in frame.columns]
    return frame


def ingest_cspec_file(cspec_file, context_df):
    frame = pd.read_csv(cspec_file, index_col=False)
    if frame["INSTRUMENT"].iloc[0] != "CCAM":
        print("skipping non-CCAM file: " + cspec_file)
        return False, context_df
    print("ingesting spectra from " + Path(cspec_file).name)
    # regularize various ways people may have rendered metadata fields
    try:
        frame = format_for_multidex(frame)
    except KeyboardInterrupt:
        raise
    except Exception as ex:
        print("failed on " + cspec_file + " reformat: " + str(ex))
        return False, context_df
    colors = []
    for _, row in frame.iterrows():
        row_color = process_marslab_row(row, cspec_file, context_df)
        if row_color is not None:
            colors.append(row_color)
    print("successfully ingested " + ", ".join(colors))
    return True, context_df


def save_thumb(filename, row):
    print("writing " + filename)
    try:
        shutil.copy(row['path'], filename)
        return True, None
    except KeyboardInterrupt:
        raise
    except Exception as ex:
        print(f"failed on {filename}: {type(ex)}: {ex}")
        return False, ex


def save_relevant_thumbs(context_df):
    if "save" not in context_df.columns:
        return {}
    to_save = context_df.loc[context_df["save"] == True]
    thumb_path = THUMB_PATH
    results = []
    for _, row in to_save.iterrows():
        filename = os.path.join(thumb_path, os.path.basename(row["path"]))
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


def ingest_multidex(
    path_or_file, *, recursive: "r" = False, skip_thumbnails: "t" = False
):
    """
    ingests ccam -cspec.csv files and context image thumbnails
    into a multidex database. expects all products in an observation
    to have matching filenames -- if you modify the filenames output by
    asdf or related applications, this script will likely fail, although you
    might be able to convince it if you really try.

    param path_or_file: cspec file or directory containing cspec files.
        looks for matching context images within the same directory.
    param recursive: attempts to ingest all cspec files and context images
        in directory tree, regardless of what specific file you passed it
    param skip_thumbnails: don't process context images or make thumbnails.
    """
    path = Path(path_or_file)
    cspec_files, context_files = find_ingest_files(path, recursive)
    if not skip_thumbnails:
        context_df = process_context_files(context_files)
        save_relevant_thumbs(context_df)
    else:
        context_df = None
    successfully_ingested_cspec_files = []
    for cspec_file in cspec_files:
        successful, context_df = ingest_cspec_file(cspec_file, context_df)
        if successful:
            successfully_ingested_cspec_files.append(cspec_file)


