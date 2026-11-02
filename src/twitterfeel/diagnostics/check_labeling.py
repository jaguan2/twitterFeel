"""Run synthetic diagnostics through the CLI and native desktop backend.

This checks execution/parity, not accuracy against human ground truth.
"""
from __future__ import annotations

import argparse
import json
import os
from twitterfeel.common.config import workspace_root
from pathlib import Path
import subprocess
import sys
import tempfile

from twitterfeel.common.provenance import file_digest

ROOT = workspace_root()
CASES = [
    ("mixed_anticipation", "I'm excited about moving, but terrified of leaving everyone behind."),
    ("mixed_farewell", "I'm happy for your promotion, but sad that you are leaving us."),
    ("neutral", "The meeting starts at three in room 204."),
    ("negation", "I'm not angry; I'm disappointed."),
    ("sarcasm", "Great, another cancelled train. Just what I needed."),
    ("quotation", 'She said "I am terrified." The interview transcript is attached.'),
    ("unclear", "Well, that was something."),
    ("blank", ""),
    ("non_english", "Estoy feliz de verte, pero tengo miedo del viaje."),
    ("explicit_joy", "I feel so happy and joyful today!"),
    ("explicit_fear", "I am terrified and afraid for my safety."),
    ("truncation", "I am excited about this. " * 90),
]


def compare_scores(cli_row, app_result, tolerance=1e-5):
    if cli_row["text"] != app_result["text"]:
        raise ValueError("CLI and app scored different text")
    if set(cli_row["scores"]) != set(app_result["scores"]):
        raise ValueError("CLI and app returned different score vocabularies")
    delta = max(abs(cli_row["scores"][label] - app_result["scores"][label])
                for label in cli_row["scores"])
    return {"max_absolute_score_difference": delta, "scores_match": delta <= tolerance}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(ROOT / "runs"):
        parser.error("Diagnostic output must be inside this repository's runs directory")
    if out.exists():
        parser.error("Output already exists; choose a new diagnostic run")
    model_path = ROOT / "runs/model_cache/roberta-go-emotions"
    if not (model_path / "model.safetensors").is_file():
        parser.error("Cached local model not found")
    out.mkdir(parents=True)
    # Bound CPU concurrency so these small diagnostics remain responsive locally.
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    import pandas as pd
    from twitterfeel.inference.engine import AnalysisEngine
    from twitterfeel.desktop.api import DesktopAPI
    from twitterfeel.workspace.workbench_data import discover_runs, load_predictions

    source = out / "synthetic_posts.csv"
    pd.DataFrame([{"post_id": name, "text": text} for name, text in CASES]).to_csv(source, index=False)
    subprocess.run([sys.executable, "-m", "twitterfeel.labeling.label_posts", "label", str(source),
                    "--out", str(out / "cli"), "--model", str(model_path),
                    "--threshold", ".5", "--max-length", "256", "--batch", "1"],
                   check=True, cwd=ROOT)
    predictions = load_predictions(out / "cli/predictions.jsonl")
    runs = [r for r in discover_runs(ROOT) if r["path"].resolve() == out / "cli"]
    if not runs:
        raise ValueError("Diagnostic output must be located beneath the repository runs directory")
    with tempfile.TemporaryDirectory() as temporary:
        # Exercise the shipped bridge/inference path with disposable app state.
        # Native window interactions are checked separately by --smoke-test.
        app = DesktopAPI(Path(temporary))
        app._engine = AnalysisEngine(ROOT)
        findings = []
        for row in predictions.to_dict("records"):
            response = app.call("infer", {"text": row["text"], "threshold": .5})
            result = {"case": row["post_id"], "text": row["text"],
                      "cli_labels": row["labels"], "cli_status": row["status"],
                      "truncated": row["truncated"]}
            if not row["text"].strip():
                result["app_blank_rejected"] = not response["ok"] and "Enter some text first" in response.get("error", "")
            else:
                if not response["ok"]:
                    raise RuntimeError(f"Desktop inference failed for {row['post_id']}: {response.get('error')}")
                trial = response["data"]
                result.update(compare_scores(row, trial))
                result.update({"app_labels": trial["labels"], "app_status": trial["status"],
                               "labels_match": row["labels"] == trial["labels"],
                               "status_matches": row["status"] == trial["status"],
                               "app_truncation_warning": trial["truncated"]})
            findings.append(result)
            print(f"Checked {row['post_id']}", flush=True)
    from twitterfeel.labeling import label_posts
    from twitterfeel.desktop import api as desktop_api
    from twitterfeel.inference import engine as analysis_engine
    report = {"purpose": "Synthetic execution and parity diagnostic; no human gold labels",
              "threshold": .5, "max_length": 256, "batch_size": 1,
              "source_sha256": file_digest(source),
              "labeler_source_sha256": file_digest(label_posts.__file__),
              "app_source_sha256": file_digest(desktop_api.__file__),
              "analysis_engine_sha256": file_digest(analysis_engine.__file__),
              "app_backend": "native_desktop_bridge",
              "model_weights_sha256": file_digest(model_path / "model.safetensors"),
              "model_config_sha256": file_digest(model_path / "config.json"),
              "cli_config": json.loads((out / "cli/config.json").read_text()),
              "findings": findings}
    report["execution_checks_passed"] = all(
        row.get("app_blank_rejected", False) if row["case"] == "blank" else
        row["scores_match"] and row["labels_match"] and row["status_matches"]
        for row in findings)
    (out / "diagnostics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {out / 'diagnostics.json'}")
    return 0 if report["execution_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
