import datetime as dt
import os
import shutil
from pathlib import Path
import re

from clize import run
import dateutil.parser as dtp
from oauth2client.service_account import ServiceAccountCredentials
from pydrive2.auth import GoogleAuth
from silencio.gdrive import stamp
import sh


def calendar_stamp():
    return dt.datetime.utcnow().isoformat()[:10].replace("-", "_")


ASDF_CLIENT_SECRET = (
    "/home/michael/Desktop/silencio/secrets/google_client_secrets.json"
)
LOGFILE = f"zcam_db_{calendar_stamp()}.log"
LOCAL_MSPEC_ROOT = "/datascratch/zcam_mspec_sync/"
DRIVE_MSPEC_ROOT = "110wJGkFyqx9cWZJjLs08lYntTRskKFOh"
MULTIDEX_ROOT = Path("/home/ubuntu/multidex/multidex")
LOCAL_DB_PATH = Path(MULTIDEX_ROOT, "data/ZCAM.sqlite3")
LOCAL_THUMB_PATH = Path(
    MULTIDEX_ROOT, "plotter/application/assets/browse/zcam"
)
DJANGO_MANAGE_PATH = Path("manage.py")
DRIVE_DB_FOLDER = "1P-7d3F6Ho0qh6fjqR6-_Vqf2a2LVDLzw"
# test folder in zcam_debug
# DRIVE_DB_FOLDER = "1KGwIVB9yW6I9NYsrVxeUlV_eluKG9nJv"
MDEX_FILE_PATTERN = re.compile(r"(marslab_SOL)|(context_image)")


# TODO, maybe: sloppy but whatever
def initialize_database(db_name):
    # also note that to make this work consistently via ssh you need to
    # link the env python to path
    sh.multidex_python(
        f"{DJANGO_MANAGE_PATH}", "migrate", f"--database={db_name}"
    )


# TODO: [endless screaming]
def make_asdf_pydrive_client():
    from silencio.gdrive import DriveBot
    gauth = GoogleAuth()
    scope = ["https://www.googleapis.com/auth/drive"]
    gauth.credentials = ServiceAccountCredentials.from_json_keyfile_name(
        ASDF_CLIENT_SECRET, scope
    )
    return DriveBot(gauth)


# TODO: [endless screaming]
def make_asdf_pydrive_client_3():
    from silencio.gdrive3 import DriveBot
    scope = ["https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(
        ASDF_CLIENT_SECRET, scope
    )
    return DriveBot(creds)

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
    log(f"{stamp()}: {message},{type(ex)},{ex}")


def csvify(sequence):
    return ",".join([str(item) for item in sequence])


def update_mdex_from_drive(
    local_only=False,
    force_rebuild=False,
    shutdown_on_completion=False,
    ingest_rc=False,
    upload=True
):
    bot = make_asdf_pydrive_client()
    if local_only is False:
        try:
            saved_files = sync_mspec_tree()
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
    log(f"{stamp()}: sync complete; {len(saved_files)} files downloaded")
    log(f"{stamp()}: building {LOCAL_DB_PATH} from {LOCAL_MSPEC_ROOT}")
    try:
        ingest_results = rebuild_database(ingest_rc)
    except Exception as ex:
        log_exception("database rebuild failed", ex)
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
        from plotter.graph import dump_model_table
        dump_model_table("ZCAM", f"ZCAM_db_dump.csv")
    except Exception as ex:
        log_exception("db dump failed", ex)
    try:
        log(f"{stamp()}: adding lab spectra to database")
        import ingest.csv2
        ingest.csv2.perform_ingest(
            "data/lab_spectra_zcam_multidex.csv", "ZCAM"
        )
    except Exception as ex:
        log_exception("lab spectra ingest failed", ex)
    # TODO, maybe: lazy but sort of whatever
    log(f"{stamp()}: compressing thumbnails")
    sh.tar(
        "cf",
        str(Path(os.getcwd(), "zcam_thumbs.tar")),
        "-C",
        LOCAL_THUMB_PATH,
        "."
    )
    if upload is False:
        log(f"{stamp()}: upload=False, terminating")
        if shutdown_on_completion is True:
            sh.sudo("shutdown", "now")
        return
    try:
        # TODO: implement these methods for drivebot3
        log(f"{stamp()}: creating build folder on Drive")
        output_folder = bot.cd(f"{stamp()}{folder_suffix}", DRIVE_DB_FOLDER)
    except Exception as ex:
        log_exception("couldn't create Drive folder", ex)
        return
    try:
        log(f"{stamp()}: transferring db file to Drive")
        bot.cp(LOCAL_DB_PATH, output_folder)
        log(f"{stamp()}: transferring db dump to Drive")
        bot.cp("ZCAM_db_dump.csv", output_folder)
        log(f"{stamp()}: compressing thumbnails and transferring to drive")
        sh.igzip("-f", "zcam_thumbs.tar")
        bot.cp("zcam_thumbs.tar.gz", output_folder)
    except Exception as ex:
        log_exception("file transfer failed", ex)
    log(f"{stamp()}: transferring log to Drive")
    try:
        bot.cp(LOGFILE, output_folder)
    except Exception as ex:
        log_exception(f"{stamp()}: log transfer failed", ex)
    if shutdown_on_completion is True:
        sh.sudo("shutdown", "now")


def sync_mspec_tree():
    from silencio.gdrive3 import DriveScanner

    bot = make_asdf_pydrive_client_3()
    log(
        f"{stamp()}: beginning Google Drive sync from "
        f"{DRIVE_MSPEC_ROOT} to {LOCAL_MSPEC_ROOT}"
    )
    scanner = DriveScanner(
        bot, query=(
            "name contains 'marslab' "
            "or name contains 'context_image' "
            "or name contains '.fits.' "
            "or mimeType = 'application/vnd.google-apps.folder'"
        )
    )
    scanner.get()
    tree = scanner.get_file_trees()[DRIVE_MSPEC_ROOT]
    directories, files = scanner.make_manifest()
    in_filesystem = files.loc[
        files['id'].isin(tree.keys())
    ]
    relevant_files = in_filesystem.loc[
        in_filesystem['name'].str.startswith('marslab_SOL')
        | in_filesystem['name'].str.startswith('context_')
        | in_filesystem['name'].str.contains('.fits.')
    ]
    relevant_directories = directories.loc[
        directories['id'].isin(relevant_files['parents'].unique())]
    drive_paths = set(
        tree[folder_id] for folder_id in relevant_directories['id'])
    sol_paths = {path for path in drive_paths if re.match(r"\d{4}", path)}
    new_paths = sol_paths
    while any(filter(lambda p: '/' in p, new_paths)):
        new_paths = [str(Path(path).parent) for path in new_paths]
        sol_paths.update(new_paths)
    reverse_tree = {v: k for k, v in tree.items()}
    extant_paths = [
        path.replace(LOCAL_MSPEC_ROOT, "").strip("/")
        for path, _, __ in os.walk(LOCAL_MSPEC_ROOT)
    ]
    extras = set(extant_paths).difference(set(sol_paths))
    for extra in filter(None, extras):
        log(f"deleting {extra} not found in remote")
        shutil.rmtree(Path(LOCAL_MSPEC_ROOT, extra), ignore_errors=True)
    saved_files = []
    for sol_path in sol_paths:
        if sol_path == '.':
            continue
        Path(LOCAL_MSPEC_ROOT, sol_path).mkdir(parents=True, exist_ok=True)
        folder_id = reverse_tree[sol_path]
        remote_files = relevant_files.loc[
            relevant_files['parents'] == folder_id
        ]
        uniques = remote_files["name"].unique()
        for extra in filter(
            lambda f: (f.name not in uniques and not f.is_dir()),
            Path(LOCAL_MSPEC_ROOT, sol_path).iterdir()
        ):
            log(f"deleting {extra} not found in remote")
            extra.unlink()
        for name, files in remote_files.groupby("name"):
            path = Path(sol_path, name)
            if len(files) > 1:
                log(f"refusing to sync duplicates of {path}")
                continue
            local = Path(LOCAL_MSPEC_ROOT, path)
            if local.exists():
                local_mtime = dt.datetime.fromtimestamp(
                    os.path.getmtime(local), tz=dt.timezone.utc
                )
                remote_mtime = dtp.parse(files['modifiedTime'].iloc[0])
                if local_mtime > remote_mtime:
                    log(f"skipping older version of {path}")
                    continue
            log(f"copying {path} to local")
            try:
                blob = bot.read_file(files['id'].iloc[0])
                with local.open('wb+') as stream:
                    stream.write(blob)
                saved_files.append(path)
            except KeyboardInterrupt:
                raise
            except Exception as ex:
                log(
                    f"couldn't sync {path}: {type(ex)}: {ex}"
                )
    return saved_files


if __name__ == "__main__":
    run(update_mdex_from_drive)
