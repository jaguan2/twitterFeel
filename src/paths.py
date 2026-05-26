"""Shared filesystem layout for TwitterFeel scripts.

All scripts import their paths from here so the repo can be reorganized
in one place. Layout:

    REPO_ROOT/
        data/
            raw/        -- source CSVs (Mental-Health-Twitter.xls, ...)
            interim/    -- generated intermediates (embeddings, windows, ...)
        src/            -- this directory
        models/         -- *.keras
        metrics/        -- metrics_*.json
        docs/           -- README, FINDINGS, handoff, notebook
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_INTERIM = REPO_ROOT / "data" / "interim"
MODELS_DIR = REPO_ROOT / "models"
METRICS_DIR = REPO_ROOT / "metrics"

for _d in (DATA_RAW, DATA_INTERIM, MODELS_DIR, METRICS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
