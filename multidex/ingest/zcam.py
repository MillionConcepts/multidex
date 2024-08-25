from functools import reduce
import io
from multiprocessing import Pool
from operator import attrgetter, and_
import os
from pathlib import Path
import re
from typing import Callable

import django.db.models
from dustgoggles.composition import Composition
from fs.osfs import OSFS
import numpy as np
import pandas as pd
from PIL import Image

pd.set_option('future.no_silent_downcasting', True)

# NOTE: do not mess with this nontsandard import order. it is necessary to
#  run this environment setup before touching any django modules.

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "multidex.settings")
os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"

django.setup()

# noinspection PyUnresolvedReferences
from plotter import div0
from plotter.field_interface_definitions import ASDF_CART_COLS, ASDF_PHOT_COLS
from plotter.models import ZSpec


# make consistently-sized thumbnails out of the asdf context images. we
# might eventually want a type of output earmarked for this, or to write the
# online thumbnails out locally. (something like this can be inserted as a
# send into look pipelines...?)

def remove_alpha(image):
    return image.convert("RGB")


def pil_crop(image, bounds):
    bounds = (
        bounds[0],
        bounds[3],
        image.size[0] - bounds[1],
        image.size[1] - bounds[2],
    )
    cropped = image.crop(bounds)
    return cropped


def thumber(image, scale):
    image.thumbnail((image.size[0] / scale, image.size[1] / scale))
    return image


def jpeg_buffer(image):
    buffer = io.BytesIO()
    image.save(buffer, "jpeg")
    buffer.seek(0)
    return buffer


def default_thumbnailer():
    inserts = {"crop": {"bounds": (20, 20, 122, 5)}, "thumb": {"scale": 2}}
    steps = {
        "load": Image.open,
        "flatten": remove_alpha,
        "crop": pil_crop,
        "thumb": thumber,
        "write": jpeg_buffer,
    }
    return Composition(steps=steps, inserts=inserts)


ASDF_STEM_PATTERN = re.compile(
    r'SOL\d{4}_zcam\d{5}_RSM\d{1,4}(-\w+)?', re.UNICODE
)


# TODO: do this better, requires making people install this better
THUMB_PATH = "plotter/application/assets/browse/zcam/"

ZSPEC_FIELD_NAMES = list(map(attrgetter("name"), ZSpec._meta.fields))


def marslab_looker(ingest_rc: bool = False) -> Callable[[str], bool]:
    def looks_like_marslab(fn: str) -> bool:
        filts = [
            ("marslab" in fn), (not ("extended" in fn)), (fn.endswith(".csv"))
        ]
        if ingest_rc is False:
            filts.append((not ("_rc_" in fn)))
        return reduce(and_, filts)

    return looks_like_marslab


def looks_like_context(fn: str) -> bool:
    return fn.endswith(".png") and ("context" in fn) and ("pixmap" not in fn)


def directory_of(path: Path) -> str:
    if path.is_dir():
        return str(path)
    return str(path.parent)


def find_ingest_files(
    path: Path, recursive: bool = False, ingest_rc: bool = False
):
    looks_like_marslab = marslab_looker(ingest_rc)
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
    relevant_bool_fields = [c for c in df.columns if c in bool_fields]
    df.loc[:, relevant_bool_fields] = df.loc[:, relevant_bool_fields] == "Y"


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
    stem = re.search(ASDF_STEM_PATTERN, asdf_fn)
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
    if "save" not in context_df.columns:
        return {}
    to_save = context_df.loc[context_df["save"] == True]
    thumb_path = THUMB_PATH
    procs = []
    pool = Pool(4)
    for _, row in to_save.iterrows():
        filename = thumb_path + row["stem"] + "-" + row["eye"] + "-thumb.jpg"
        procs.append(pool.apply_async(save_thumb, (filename, row)))
    pool.close()
    pool.join()
    results = [p.get() for p in procs]
    pool.terminate()
    records = []
    for (success, ex), path in zip(results, to_save['path']):
        records.append(
            {
                "file": path,
                "filetype": "thumb",
                "status": success,
                "exception": ex,
            }
        )
    return records


def process_marslab_row(row, marslab_file, obs_images):
    row = row.dropna()
    relevant_indices = [ix for ix in row.index if ix in ZSPEC_FIELD_NAMES]
    for filt in set(ZSpec.filters).intersection(row.index):
        row[filt] = float(row[filt])
    metadata = dict(row[relevant_indices]) | {
        "filename": Path(marslab_file).name,
        "images": obs_images,
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
    frame.columns = [col.lower() for col in frame.columns]
    return frame


def _cart_flagchecks(table, color):
    cart = table.loc[
        table['COLOR'] == color, ASDF_CART_COLS
    ].iloc[0]
    if cart.isna().any():
        return "no_data"
    # TODO, maybe: check for missing eye (rare case)
    if (cart == 0).any():
        # i think this generally indicates that the ROI overlaps but is
        # not completely within a missing-data region, or is right on an
        # edge, so calculation gets screwy. might be other causes. might
        # need a high bounds check as well, or a looser low bounds check.
        return 'bad'
    dubious_a = False
    for eye in ('LEFT', 'RIGHT'):
        if abs(
            np.log10(cart[f'{eye}_HW']) - np.log10(cart[f'{eye}_A'])
        ) > 1.25:
            return 'dubious_a'
    if dubious_a is True:
        return 'dubious_a'
    for dim in ('H', 'W'):
        if abs(
            cart[f'LEFT_{dim}'] - cart[f'RIGHT_{dim}']
        ) > (cart[[f'LEFT_{dim}', f'RIGHT_{dim}']].min() * 4):
            return 'dubious_bounds'
    if (cart > 800).any():
        return 'dubious_size'
    return 'ok'


def _phot_flagchecks(table, color):
    phot = table.loc[
        table['COLOR'] == color, ASDF_PHOT_COLS
    ].iloc[0]
    if phot.isna().any():
        return 'no_data'
    if (phot < 0).any():
        return 'negative_angles'
    if (phot > 180).any():
        return 'dubious_angles'
    return 'ok'


def spatial_flags(table):
    cartflags, photflags = [], []
    for color in table['COLOR']:
        cartflags.append(_cart_flagchecks(table, color))
        photflags.append(_phot_flagchecks(table, color))
    return cartflags, photflags


def insert_spatial_metadata(table):
    space = table.copy()
    if not (('LEFT_H' in space.columns) or ('RIGHT_H' in space.columns)):
        # asdf found no suitable XYR for this observation
        space['spatial_flag'] = "no_data"
        space['phot_flag'] = "no_data"
    else:
        space[ASDF_CART_COLS] = space[ASDF_CART_COLS].astype(np.float32)
        space['spatial_flag'], space['phot_flag'] = spatial_flags(space)
        for c in ASDF_CART_COLS:
            space[f'{c}MAG'] = np.log10(space[c]).astype(np.float32)
    return space


def ingest_marslab_file(marslab_file, context_df):
    frame = pd.read_csv(marslab_file)
    if "INSTRUMENT" in frame.columns:
        # TODO: maybe put the hard version of this check back after getting
        #  INSTRUMENT into the rc_marslab files
        if frame["INSTRUMENT"].iloc[0] != "ZCAM":
            print("skipping non-ZCAM file: " + marslab_file)
            return False, "does not appear to be a ZCAM file", context_df
    frame = frame.replace(["-", "", " ", "--"], np.nan)
    # don't ingest duplicate copies of rc-file-derived caltarget values
    if 'FEATURE' in frame.columns:
        if (frame['FEATURE'] == 'caltarget').all():
            # TODO: make this nicer
            sol = ZSpec.objects.filter(
                sol__iexact=frame['SOL'].iloc[0]
            )
            seq_id = sol.filter(seq_id__iexact=frame['SEQ_ID'].iloc[0])
            geometry = seq_id.filter(
                incidence_angle__iexact=frame["INCIDENCE_ANGLE"].iloc[0]
            )
            if len(geometry) > 0:
                print(f"dupe caltarget values {marslab_file}, skipping")
                return False, "dupe caltarget file", context_df
    if frame["COLOR"].isna().all():
        print(f"no spectra in {marslab_file}, skipping")
        return False, "no spectra in file", context_df
    # TODO: temporary hack
    if "TARGET_ELEV" in frame.columns:
        frame["TARGET_ELEVATION"] = frame["TARGET_ELEV"]
    print("ingesting spectra from " + Path(marslab_file).name)
    if (context_df is not None) and ("_rc_" not in marslab_file):
        obs_images, match_index = match_obs_images(marslab_file, context_df)
        if obs_images != {}:
            print(f"found matching images: {obs_images}")
            context_df.loc[match_index, "save"] = True
        else:
            print("no matching images found")
    else:
        obs_images = {}

    frame = insert_spatial_metadata(frame)
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


def nailpipe(image_path):
    return default_thumbnailer().execute(image_path)


def make_thumb_blobs(paths):
    pool = Pool(4)
    thumbproc = [pool.apply_async(nailpipe, (p,)) for p in paths]
    pool.close()
    pool.join()
    thumb_blobs = [proc.get() for proc in thumbproc]
    pool.terminate()
    return thumb_blobs


def perform_ingest(
    path_or_file,
    *,
    recursive=False,
    skip_thumbnails=False,
    ingest_rc=False
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
    param ingest_rc: ingest tables of spectra for caltarget observations
        generated from "rc" (radiometric calibration) files.
    """
    path = Path(path_or_file)
    marslab_files, context_files = find_ingest_files(
        path, recursive, ingest_rc
    )
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
    if skip_thumbnails:
        return results
    print("making thumbnails")
    context_df = context_df.loc[context_df["save"]].copy()
    context_df['buffer'] = make_thumb_blobs(context_df['path'])
    print("saving thumbnails")
    thumb_results = save_relevant_thumbs(context_df)
    return results + thumb_results
