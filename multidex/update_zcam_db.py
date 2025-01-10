import datetime as dt
import json
import os
import time
from functools import reduce
from itertools import chain
from operator import and_
from pathlib import Path
import re
import sys

import fire
import pandas as pd
from dustgoggles.dynamic import exc_report
from dustgoggles.structures import MaybePool
from hostess.directory import index_breadth_first
from hostess.subutils import runv, run
from googleapiclient.errors import Error as GoogleError
from more_itertools import chunked
from oauth2client.service_account import ServiceAccountCredentials

from multidex.ingest.local_settings.zcam import (
    ASDF_CLIENT_SECRET,
    LOCAL_MSPEC_ROOT,
    DRIVE_MSPEC_ROOT,
    MULTIDEX_ROOT,
    SHARED_DRIVE_ID,
    DRIVE_DB_FOLDER,
)

OBS_TITLE_PATTERN = re.compile(
    r"(?P<SEQ_ID>zcam\d{5}) (?P<NAME>.*?) RSM (?P<RSM>\d+)"
)

MARSLAB_FN_PATTERN = re.compile(
    r"(?P<FTYPE>marslab|roi)_((?P<FORMAT>extended|rc)_)?SOL(?P<SOL>\d{4})_"
    r"(?P<SEQ_ID>\w+)_RSM(?P<RSM>\d+)(-(?P<ANALYSIS_NAME>.+?))?\."
    r"(?P<EXTENSION>fits\.gz|fits|csv)"
)

LOCAL_DB_PATH = Path(MULTIDEX_ROOT, "data/ZCAM.sqlite3")
LOCAL_THUMB_PATH = Path(
    MULTIDEX_ROOT, "plotter/application/assets/browse/zcam"
)
DJANGO_MANAGE_PATH = Path("manage.py")


def stamp() -> str:
    return dt.datetime.utcnow().isoformat()[:19]


def marslab_nameparse(fn):
    fn = fn.name if isinstance(fn, Path) else fn
    try:
        parsed = MARSLAB_FN_PATTERN.search(fn).groupdict()
    except AttributeError:
        return {k: None for k in MARSLAB_FN_PATTERN.groupindex}
    if parsed['FORMAT'] is None and parsed['FTYPE'] == 'marslab':
        parsed['FORMAT'] = 'compact'
    return parsed


def calendar_stamp():
    return dt.datetime.utcnow().isoformat()[:10].replace("-", "_")


def initialize_database(db_name):
    initproc = runv(
        f"{sys.executable} {DJANGO_MANAGE_PATH}", "migrate",
        f"--database={db_name}"
    )
    initproc.wait()
    log(f"init_out\n{initproc.out}")
    log(f"init_err\n{initproc.err}")
    if initproc.returncode() != 0:
        log("****database did not initialize successfully:****")
        raise ValueError("init fail, bailing out")


def make_mspec_drivebot():
    from silencio.gdrive3 import DriveBot
    scope = ["https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(
        ASDF_CLIENT_SECRET, scope
    )
    return DriveBot(creds, shared_drive_id=SHARED_DRIVE_ID)


def rebuild_database(ingest_rc):
    if os.path.exists(LOCAL_DB_PATH):
        os.unlink(LOCAL_DB_PATH)
    initialize_database("ZCAM")
    import ingest.zcam

    return ingest.zcam.perform_ingest(
        LOCAL_MSPEC_ROOT, recursive=True, ingest_rc=ingest_rc
    )


def log(line):
    with open(LOGFILE, "a+") as logfile:
        logfile.write(line + "\n")


def log_exception(message: str, ex: Exception):
    if isinstance(ex, KeyboardInterrupt):
        raise
    log(f"{stamp()}: {message}")
    log("****exc_report****")
    log(json.dumps(exc_report(ex)))
    log("****end exc_report****")


def csvify(sequence):
    return ",".join([str(item) for item in sequence])


def _check_correspondence(marslab_files, filtered_parseframe):
    corrpreds = [
        marslab_files[k] == filtered_parseframe[k]
        for k in ('SOL', 'SEQ_ID', 'RSM')
    ]
    corresponds = reduce(and_, corrpreds)
    # noinspection PyUnresolvedReferences
    if not corresponds.all():
        misplaced = '\n'.join(marslab_files.loc[~corresponds, 'name'].tolist())
        log(
            f"*****WARNING: some files are possibly misplaced:\n"
            f"{misplaced}\n*******"
        )
        with open("maybe_misplaced.log", "w") as stream:
            stream.write(misplaced)


def _investigate_drive_solfolder(solname, solfolder):
    bot = make_mspec_drivebot()
    obsfolders, _top_level_files = bot.manifest(solfolder)
    datafile_frames = []
    for _, obsrow in obsfolders.iterrows():
        try:
            datafolder = bot.ls(folder_id=obsrow['id'])['data']
        except (GoogleError, KeyError):
            continue
        _datafolders, datafiles = bot.manifest(datafolder)
        if len(datafiles) == 0:
            continue
        try:
            folderparse = OBS_TITLE_PATTERN.match(obsrow['name']).groupdict()
            for k, v in folderparse.items():
                datafiles[k] = v
        except AttributeError:
            # nonstandard folder name
            for k in OBS_TITLE_PATTERN.groupindex:
                datafiles[k] = None
        datafile_frames.append(datafiles)
    if len(datafile_frames) == 0:
        return []
    datafile_frame = pd.concat(datafile_frames)
    datafile_frame['SOL'] = solname
    return datafile_frame


def index_drive_data_folders():
    bot = make_mspec_drivebot()
    soldirs = {
        name: fid for name, fid in bot.ls(folder_id=DRIVE_MSPEC_ROOT).items()
        if name.isnumeric()
    }
    pool = MaybePool(6)
    pool.map(
        _investigate_drive_solfolder,
        [{'key': name, 'args': (name, id_)} for name, id_ in soldirs.items()]
    )
    pool.close()
    i = 0
    while not pool.ready():
        i += 1
        time.sleep(1)
        if i % 5 != 0:
            continue
        done = [r for r in pool.results_ready().values() if r is True]
        log(f"{stamp()}: {len(done)}/{len(soldirs)} sols indexed")
    log(f"{stamp()}: {len(soldirs)}/{len(soldirs)} sols indexed")
    solresults = pool.get()
    pool.terminate()
    exceptions = {
        sol: e for sol, e in solresults.items() if isinstance(e, Exception)
    }
    if len(exceptions) > 0:
        log("****errors in indexing****")
        if len(exceptions) > 10:
            log("truncating list due to length")
        for sol, ex in exceptions.items():
            log(f"**sol {sol}**")
            log(json.dumps(exc_report(ex)))
        raise ValueError("indexing errors, bailing out")
    manifest = pd.concat(
        list(solresults.values())
    ).sort_values(by='SOL').reset_index(drop=True)
    manifest = manifest.rename(columns={'name': 'fn'})
    parseframe = pd.DataFrame.from_records(
        manifest['fn'].map(marslab_nameparse)
    )
    filtered_parseframe = parseframe.dropna(subset='FTYPE')
    marslab_files = manifest.loc[filtered_parseframe.index]
    _check_correspondence(marslab_files, filtered_parseframe)
    marslabs = marslab_files.copy().reset_index(drop=True)
    marsparse = filtered_parseframe.copy().reset_index(drop=True)
    for c, s in marsparse.items():
        marslabs[c] = s
    additional = manifest.loc[manifest['fn'].str.match(r"context_|.*\.sel$")]
    return marslabs, additional, manifest


def _row2path(row):
    return Path(
        f"{row['SOL']}/{row['SEQ_ID']} {row['NAME']} RSM {row['RSM']}"
        f"/{row['fn']}"
    )


def _download_chunk_from_drive(rowchunk, logfile):
    # TODO: hack for spawning, should rewrite some other way
    global LOGFILE
    LOGFILE = logfile
    bot, success = make_mspec_drivebot(), []
    success = []
    for row in rowchunk:
        log(f"downloading {row['target']}")
        target = Path(LOCAL_MSPEC_ROOT) / row['target']
        target.parent.mkdir(exist_ok=True, parents=True)
        bot.get(row["file_id"], target)
        success.append(row['target'])
    return success


def check_sync(getspecs, local):
    local = local.loc[local['directory'] == False]
    rel = local['path'].map(lambda p: Path(p).relative_to(LOCAL_MSPEC_ROOT))
    extras = local.loc[~rel.isin(getspecs['target'].to_list())]
    for _, extra in extras.iterrows():
        log(f"deleting {extra['path']} not found in remote")
        Path(extra['path']).unlink()
    existing = local.loc[rel.isin(getspecs['target'].to_list())]
    timeskip = set()
    drive_mtimes = pd.to_datetime(getspecs['time'])
    tzname = dt.datetime.now().astimezone().tzname()
    local_mtimes = pd.to_datetime(existing['MTIME']).dt.tz_localize(tzname)
    for ix in existing.index:
        omod = drive_mtimes.loc[getspecs['target'] == rel.loc[ix]]
        if omod.iloc[0] < local_mtimes.loc[ix]:
            log(f"not copying older version of {rel.loc[ix]}")
            timeskip.update(omod.index)
    return getspecs.loc[getspecs.index.difference(timeskip)]


def sync_mspec_tree():
    log(f"{stamp()}: indexing {DRIVE_MSPEC_ROOT} ")
    marslabs, additional, manifest = index_drive_data_folders()
    targets = pd.concat([marslabs, additional])
    log(
        f"{stamp()}: beginning Google Drive sync from "
        f"{DRIVE_MSPEC_ROOT} to {LOCAL_MSPEC_ROOT}"
    )
    getspecs = [
        {'file_id': r['id'], 'target': _row2path(r), 'time': r['modifiedTime']}
        for _, r in targets.iterrows()
    ]
    getspecs = pd.DataFrame(getspecs)
    if (dupepred := getspecs.duplicated(subset="target", keep="first")).any():
        for dupe in getspecs.loc[dupepred, "target"]:
            log(f"refusing to sync duplicates of {dupe}")
        getspecs = getspecs.drop_duplicates(subset="target", keep="first")
    if len(getspecs) == 0:
        log("nothing to download")
        return manifest, [], []
    local = pd.DataFrame(index_breadth_first(LOCAL_MSPEC_ROOT))
    if len(local) != 0:
        getspecs = check_sync(getspecs, local)
    log(f"{stamp()}: initiating download of {len(getspecs)} files")
    getchunks = chunked((spec for _, spec in getspecs.iterrows()), 20)
    pool = MaybePool(4)
    pool.map(
        _download_chunk_from_drive,
        [{'args': (chunk, LOGFILE)} for chunk in getchunks]
    )
    pool.close()
    pool.join()
    results = pool.get()
    exceptions = {e for e in results.values() if isinstance(e, Exception)}
    if len(exceptions) > 0:
        log("****errors in download****")
        if len(exceptions) > 10:
            log("truncating list due to length")
        for ex in exceptions:
            log(json.dumps(exc_report(ex)))
        raise ValueError("indexing errors, bailing out")
    successes = list(
        chain(
            *[v for v in results.values() if not isinstance(v, Exception)]
        )
    )
    pool.terminate()
    if len(successes) != len(getspecs):
        log("***warning: some files may not have downloaded:***")
        [log(f.name) for f in set(getspecs['target']).difference(successes)]
        log("***end of list***")
    return manifest, successes


def update_mdex_from_drive(
    local_only=False,
    force_rebuild=False,
    shutdown_on_completion=False,
    ingest_rc=False,
    upload=True
):
    if local_only is False:
        try:
            manifest, saved_files = sync_mspec_tree()
        except Exception as ex:
            log_exception("sync with Google Drive failed", ex)
            return
    else:
        saved_files = []
    folder_suffix = ""
    if len(saved_files) == 0:
        log(f"{stamp()}: no new files found in Drive")
        if force_rebuild is True or (local_only is True):
            log(
                f"{stamp()}: force_rebuild or local_only passed, "
                f"rebuilding db anyway"
            )
        else:
            folder_suffix = " [no updates]"
    else:
        log(f"{stamp()}: sync complete; {len(saved_files)} files downloaded")
    log(f"{stamp()}: building {LOCAL_DB_PATH} from {LOCAL_MSPEC_ROOT}")
    try:
        ingest_results = rebuild_database(ingest_rc)
    except Exception as ex:
        if 'bailing' in str(ex):
            # logging already handled
            return
        log("****database build handler failed****")
        log(json.dumps(exc_report(ex)))
        return
    if len(ingest_results) == 0:
        log(f"{stamp()}: unusual error: no files ingested")
        return
    log(f"{stamp()}: database build complete")
    log("BEGIN FILE MANIFEST")
    # log paths relative to tree root rather than local drive root
    for file in ingest_results:
        file["file"] = file["file"].replace(LOCAL_MSPEC_ROOT + "/", "")
    with open(LOGFILE, "a+") as logfile:
        logfile.write(f"{csvify(ingest_results[0].keys())}\n")
        for result in ingest_results:
            logfile.write(f"{csvify(result.values())}\n")
    log("END FILE MANIFEST")
    try:
        log(f"{stamp()}: dumping database as CSV")
        from multidex.plotter.graph import dump_model_table
        dump_model_table("ZCAM", f"ZCAM_db_dump.csv")
    except Exception as ex:
        log_exception("db dump failed", ex)
    try:
        log(f"{stamp()}: adding lab spectra to database")
        import ingest.csv2
        ingest.csv2.perform_ingest(
            "simulated_spectra/lab_spectra_zcam_multidex.csv", "ZCAM"
        )
    except Exception as ex:
        log_exception("lab spectra ingest failed", ex)
    # TODO, maybe: lazy but sort of whatever
    log(f"{stamp()}: compressing thumbnails")
    tarproc = runv(
        f"tar -cf {Path(os.getcwd(), 'zcam_thumbs.tar')} "
        f"-C {LOCAL_THUMB_PATH} ."
    )
    tarproc.wait()
    if tarproc.returncode() != 0:
        log(f"****thumbnail tar failed, bailing out****")
        log('\n'.join(tarproc.out))
        log('\n'.join(tarproc.err))
        return
    if upload is False:
        log(f"{stamp()}: upload=False, terminating")
        if shutdown_on_completion is True:
            run("sudo shutdown now")
        return
    try:
        bot = make_mspec_drivebot()
        log(f"{stamp()}: creating build folder on Drive")
        output_folder = bot.cd(DRIVE_DB_FOLDER, f"{stamp()}{folder_suffix}")
    except Exception as ex:
        log_exception("couldn't create Drive folder", ex)
        return
    try:
        log(f"{stamp()}: transferring db file to Drive")
        bot.put(LOCAL_DB_PATH, folder_id=output_folder)
        log(f"{stamp()}: transferring db dump to Drive")
        bot.put("ZCAM_db_dump.csv", folder_id=output_folder)
        log(f"{stamp()}: compressing thumbnails and transferring to drive")
        run("igzip -f zcam_thumbs.tar")
        bot.put("zcam_thumbs.tar.gz", folder_id=output_folder)
    except Exception as ex:
        log_exception("file transfer failed", ex)
    log(f"{stamp()}: transferring log to Drive")
    try:
        bot.put(LOGFILE, folder_id=output_folder)
    except Exception as ex:
        log_exception(f"{stamp()}: log transfer failed", ex)
    if shutdown_on_completion is True:
        run("sudo shutdown now")


if __name__ == "__main__":
    LOGFILE = f"zcam_db_{calendar_stamp()}.log"

    fire.Fire(update_mdex_from_drive)
