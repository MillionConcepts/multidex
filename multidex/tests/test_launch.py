from threading import Thread
import os
from pathlib import Path
import time

import django
import requests
import requests.exceptions


# def run_multidex():
#     import subprocess
#     start_script = str(Path(Path(__name__).parent.parent, "multidex.py"))
#     subprocess.run(["python", start_script, "TEST"])


def test_launch_multidex():

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "multidex.settings")
    os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"
    django.setup()
    from plotter.application.run import run_multidex
    proc = Thread(target=run_multidex, args=("TEST",), daemon=True)
    proc.start()
    attempts = 0
    successful = False
    response_text = ""
    while attempts < 5:
        print("...checking multidex...")
        time.sleep(1)
        try:
            response = requests.get("http://127.0.0.1:49303")
            successful = True
            response_text = response.text
            break
        except requests.exceptions.ConnectionError:
            attempts += 1
            continue
    if successful is False:
        raise ConnectionError("server did not appear to initialize correctly")
    assert "fake-output-for-callback-with-only-side-effects" in response_text, \
        "server did not serve expected content"
