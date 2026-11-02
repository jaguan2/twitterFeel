"""Compatibility entry point; implementation: twitterfeel.research.sadness.sadness_report."""
import sys

# Support this historical script path even before an editable install.
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

if __name__ == "__main__":
    from runpy import run_module
    run_module("twitterfeel.research.sadness.sadness_report", run_name="__main__")
else:
    from importlib import import_module
    sys.modules[__name__] = import_module("twitterfeel.research.sadness.sadness_report")
