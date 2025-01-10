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

    fire.Fire(run_multidex)


# tell fire to handle command line call
if __name__ == "__main__":
    try:
        multidex_run_hook()
    except ModuleNotFoundError:
        print(
            "Execution failure: MultiDEx was restructured as a Python package "
            "in version 0.10.0 and now requires installation. Please run 'pip "
            "install -e .' from the MultiDEx repository root. Note that "
            "after doing this, it will be possible to execute MultiDEx by "
            "running 'multidex INSTRUMENT_CODE' from any working directory, "
            "although running 'python multidex.py INSTRUMENT_CODE' from this "
            "directory will still work."
        )
