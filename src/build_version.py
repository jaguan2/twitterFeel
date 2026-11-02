"""Compatibility entry point; implementation: twitterfeel.desktop.build_version."""
import sys

if __name__ == "__main__":
    from runpy import run_module
    run_module("twitterfeel.desktop.build_version", run_name="__main__")
else:
    from importlib import import_module
    sys.modules[__name__] = import_module("twitterfeel.desktop.build_version")
