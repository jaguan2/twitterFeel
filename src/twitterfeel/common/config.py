"""Resolve writable workspaces without creating directories during imports."""
from __future__ import annotations

import os
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def workspace_root(explicit=None):
    """Prefer an explicit path or TWITTERFEEL_HOME; retain checkout defaults."""
    configured = explicit if explicit is not None else os.environ.get("TWITTERFEEL_HOME")
    if configured is not None:
        if not str(configured).strip():
            raise ValueError("Workspace path must not be blank")
        return Path(configured).expanduser().resolve()
    for candidate in (*Path(__file__).resolve().parents, Path.cwd()):
        if (candidate / "app_version.json").is_file() and (candidate / "src/twitterfeel").is_dir():
            return candidate
    local = os.environ.get("LOCALAPPDATA")
    return (Path(local) / "TwitterFeel" if local else Path.home() / ".local/share/twitterfeel").resolve()


REPO_ROOT = workspace_root()
DATA_RAW = REPO_ROOT / "data/raw"
DATA_INTERIM = REPO_ROOT / "data/interim"
MODELS_DIR = REPO_ROOT / "models"
METRICS_DIR = REPO_ROOT / "metrics"


def ensure_research_directories():
    """Initialize legacy output folders when a forecasting command starts."""
    for directory in (DATA_RAW, DATA_INTERIM, MODELS_DIR, METRICS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def app_release(root):
    """Use workspace release metadata, or installed metadata for a fresh workspace."""
    manifest = Path(root) / "app_version.json"
    if manifest.is_file():
        return json.loads(manifest.read_text(encoding="utf-8"))
    try:
        return {"version": version("twitterfeel"), "changes": []}
    except PackageNotFoundError:
        # Historical script commands also work directly from an uninstalled checkout.
        source_manifest = Path(__file__).resolve().parents[3] / "app_version.json"
        return json.loads(source_manifest.read_text(encoding="utf-8"))
