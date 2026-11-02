"""Compatibility entry point; implementation: twitterfeel.evaluation.threshold_tuning."""
import sys

if __name__ == "__main__":
    from runpy import run_module
    run_module("twitterfeel.evaluation.threshold_tuning", run_name="__main__")
else:
    from importlib import import_module
    sys.modules[__name__] = import_module("twitterfeel.evaluation.threshold_tuning")
