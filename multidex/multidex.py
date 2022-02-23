import os

try:
    from clize import run
except ModuleNotFoundError:
    raise ImportError(
        "Unable to perform imports. Did you activate the multidex "
        "environment?"
    )
import django


# tell clize to handle command line call
if __name__ == "__main__":
    # note: ignore any PEP 8-based linter / IDE complaints about import
    # order: the following statements _must_ come before we import all the
    # django dependencies
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "multidex.settings")
    os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"
    django.setup()
    import plotter.application.run

    run(plotter.application.run.run_multidex)
