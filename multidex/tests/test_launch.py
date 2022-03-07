import os
import requests

import django


def test_launch_multidex():
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "multidex.settings")
    os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"
    django.setup()
    import plotter.application.run

    plotter.application.run.run_multidex(
        "TEST", debug=False, use_notepad_cache=False
    )
    response = requests.get("http://127.0.0.1:49303")
    print(response.content)
