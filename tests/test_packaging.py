"""Package boundaries, resource loading, workspace isolation and old commands."""
from importlib.metadata import version
from importlib.resources import files
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from twitterfeel.common.config import workspace_root
from twitterfeel.common.resources import annotation_rubric

PROJECT = Path(__file__).resolve().parents[1]


class PackagingTest(unittest.TestCase):
    def test_resources_are_bundled_independently_of_workspace(self):
        rubric, digest = annotation_rubric()
        self.assertEqual(len(digest), 64)
        self.assertIn("version", rubric)
        assets = files("twitterfeel.desktop").joinpath("assets")
        for name in ("index.html", "app.js", "app.css", "app.ico"):
            self.assertGreater(len(assets.joinpath(name).read_bytes()), 0)

    def test_version_manifest_matches_installed_package(self):
        release = json.loads((PROJECT / "app_version.json").read_text(encoding="utf-8"))
        self.assertEqual(version("twitterfeel"), release["version"])

    def test_explicit_workspace_overrides_environment_and_blank_is_rejected(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.dict(os.environ, {"TWITTERFEEL_HOME": str(root / "environment")}):
                self.assertEqual(workspace_root(), root / "environment")
                self.assertEqual(workspace_root(root / "explicit"), root / "explicit")
                with self.assertRaises(ValueError):
                    workspace_root("")
            self.assertFalse((root / "environment").exists())
            self.assertFalse((root / "explicit").exists())

    def test_imports_do_not_initialize_storage_or_load_historical_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary) / "uncreated-workspace"
            env = {**os.environ, "TWITTERFEEL_HOME": str(workspace)}
            code = (
                "from pathlib import Path; import sys; "
                "from twitterfeel.common.config import REPO_ROOT; "
                "from twitterfeel.desktop.api import DesktopAPI; "
                "assert not REPO_ROOT.exists(); "
                "assert not any(n.startswith('twitterfeel.research') for n in sys.modules); "
                "from twitterfeel.research.sadness import paths; "
                "assert not paths.RESULTS_DIR.exists()"
            )
            result = subprocess.run([sys.executable, "-c", code], cwd=temporary, env=env,
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(workspace.exists())

    def test_module_and_legacy_entry_points_work_outside_checkout(self):
        commands = [
            ["-m", "twitterfeel"],
            ["-m", "twitterfeel.labeling.label_posts"],
            ["-m", "twitterfeel.research.forecasting.preprocess_v4"],
            [str(PROJECT / "src/desktop_app.py")],
            [str(PROJECT / "src/label_posts.py")],
        ]
        with tempfile.TemporaryDirectory() as temporary:
            for command in commands:
                with self.subTest(command=command):
                    result = subprocess.run([sys.executable, *command, "--help"], cwd=temporary,
                                            text=True, capture_output=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("usage:", result.stdout)


if __name__ == "__main__":
    unittest.main()
