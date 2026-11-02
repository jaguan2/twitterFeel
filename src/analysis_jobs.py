"""Compatibility entry point; implementation: twitterfeel.inference.jobs."""
import sys

if __name__ == "__main__":
    from runpy import run_module
    run_module("twitterfeel.inference.jobs", run_name="__main__")
else:
    from importlib import import_module
    sys.modules[__name__] = import_module("twitterfeel.inference.jobs")
