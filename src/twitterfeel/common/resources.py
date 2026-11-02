"""Read bundled research definitions independently of the writable workspace."""
import hashlib
from importlib.resources import files
import json
from pathlib import Path


def annotation_rubric(root=None):
    """Allow the existing workspace override, otherwise use packaged bytes."""
    override = Path(root) / "src/annotation_rubric.json" if root is not None else None
    raw = (override.read_bytes() if override is not None and override.is_file()
           else files("twitterfeel.labeling").joinpath("annotation_rubric.json").read_bytes())
    return json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()
