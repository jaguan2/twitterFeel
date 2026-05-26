"""Filesystem layout for the twitter_feel_plus subproject.

Pulls the source corpus (DistilBERT-labeled tweets) from the main repo's
data/interim/ directory; writes its own results under twitter_feel_plus/results/.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_INTERIM = REPO_ROOT / "data" / "interim"
SOURCE_CSV = DATA_INTERIM / "dataset_godknowswhat.csv"

TFP_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = TFP_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
