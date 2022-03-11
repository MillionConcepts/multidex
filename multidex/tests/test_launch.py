from multiprocessing import Process
import os
import time

import django
import requests
import requests.exceptions


def test_launch_multidex():
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "multidex.settings")
    os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"
    django.setup()
    import plotter.application.run
    proc = Process(
        target=plotter.application.run.run_multidex, args=("TEST",)
    )
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
    proc.terminate()
    if successful is False:
        raise ConnectionError("server did not appear to initialize correctly")
    assert (
        "fake-output-for-callback-with-only-side-effects" in response_text,
        "server did not serve expected content"
    )
