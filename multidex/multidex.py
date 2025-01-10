import os

try:
    import fire
except ModuleNotFoundError:
    raise ImportError(
        "Unable to perform imports. Did you activate the multidex "
        "environment?"
    )

def multidex_run_hook():
    import django

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "multidex.multidex.settings")
    os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"
    django.setup()
    from multidex.plotter.application.run import run_multidex

    import sys
    sys.argv += ["ZCAM"]
    fire.Fire(run_multidex)


# tell fire to handle command line call
if __name__ == "__main__":
    multidex_run_hook()
