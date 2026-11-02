"""Compatibility entry point; implementation: twitterfeel.research.forecasting.train_v3."""
import sys

if __name__ == "__main__":
    from runpy import run_module
    run_module("twitterfeel.research.forecasting.train_v3", run_name="__main__")
else:
    from importlib import import_module
    sys.modules[__name__] = import_module("twitterfeel.research.forecasting.train_v3")
