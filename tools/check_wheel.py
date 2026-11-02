"""Check a built wheel outside the checkout, without downloading models."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", nargs="?", type=Path)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    if args.wheel is None:
        wheels = list((project / "dist").glob("twitterfeel-*.whl"))
        if len(wheels) != 1:
            parser.error("Supply one wheel path, or keep exactly one TwitterFeel wheel in dist/")
        wheel = wheels[0]
    else:
        wheel = args.wheel.resolve()
    with tempfile.TemporaryDirectory(prefix="twitterfeel-wheel-check-") as temporary:
        check_root = Path(temporary)
        target = check_root / "installed"
        workspace = check_root / "research-workspace"
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-index",
                        "--no-deps", "--target", str(target), str(wheel)], check=True)
        env = {**os.environ, "PYTHONPATH": str(target), "TWITTERFEEL_HOME": str(workspace),
               "TWITTERFEEL_WHEEL_TARGET": str(target)}
        code = '''
import hashlib
from importlib.resources import files
import os
from pathlib import Path
import sys
import twitterfeel
from twitterfeel.common.config import REPO_ROOT, app_release
from twitterfeel.common.resources import annotation_rubric
from twitterfeel.desktop.api import DesktopAPI
target = Path(os.environ["TWITTERFEEL_WHEEL_TARGET"])
assert Path(twitterfeel.__file__).is_relative_to(target), twitterfeel.__file__
assert not REPO_ROOT.exists(), "Imports created workspace storage"
assert not any(n.startswith("twitterfeel.research") for n in sys.modules)
assets = files("twitterfeel.desktop").joinpath("assets")
for name in ("index.html", "app.css", "app.js", "app.ico"):
    assert assets.joinpath(name).read_bytes(), name
rubric, digest = annotation_rubric()
raw = files("twitterfeel.labeling").joinpath("annotation_rubric.json").read_bytes()
assert digest == hashlib.sha256(raw).hexdigest()
api = DesktopAPI(REPO_ROOT)
assert api._annotation()["version"] == rubric["version"]
assert api._app_version() == app_release(REPO_ROOT)
preview = api.call("preview_import", {"name": "Wheel fixture", "format": "txt", "content": "Synthetic post"})
assert preview["ok"], preview
print("Installed wheel: imports, bundled resources, workspace and bridge passed")
'''
        subprocess.run([sys.executable, "-c", code], cwd=check_root, env=env, check=True)
        for module in ("twitterfeel", "twitterfeel.labeling.label_posts",
                       "twitterfeel.inference.setup_models",
                       "twitterfeel.research.forecasting.preprocess_v4"):
            result = subprocess.run([sys.executable, "-m", module, "--help"], cwd=check_root,
                                    env=env, text=True, capture_output=True)
            if result.returncode or "usage:" not in result.stdout:
                raise RuntimeError(f"{module}: {result.stdout}\n{result.stderr}")
        print("Installed wheel: module entry points passed")


if __name__ == "__main__":
    main()
