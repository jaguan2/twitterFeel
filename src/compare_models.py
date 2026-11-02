"""Compatibility entry point; implementation: twitterfeel.evaluation.compare_models."""
import sys

if __name__ == "__main__":
    from runpy import run_module
    run_module("twitterfeel.evaluation.compare_models", run_name="__main__")
else:
    from importlib import import_module
    sys.modules[__name__] = import_module("twitterfeel.evaluation.compare_models")
