"""Preserve historical input and output locations without import-time writes."""
from twitterfeel.common.config import REPO_ROOT, DATA_INTERIM

SOURCE_CSV = DATA_INTERIM / "dataset_godknowswhat.csv"
TFP_ROOT = REPO_ROOT / "twitter_feel_plus"
RESULTS_DIR = TFP_ROOT / "results"
