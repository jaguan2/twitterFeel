"""Background, append-only analysis runs with progress and reproducible artifacts."""
from __future__ import annotations

from collections import Counter
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import threading
import uuid

from analysis_engine import AnalysisEngine
from emotion_schema import file_digest
from research_workspace import DatasetLibrary
from label_schema import SCHEMA_VERSION, validate_thresholds


def _write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


class AnalysisJobs:
    def __init__(self, root, engine=None):
        self.root = Path(root).resolve()
        self.engine = engine if engine is not None else AnalysisEngine(self.root)
        self.library = DatasetLibrary(self.root)
        self._lock = threading.Lock()
        self._start_lock = threading.Lock()
        self._jobs = {}

    def start(self, dataset_id, threshold=.5, include_sentiment=True, name="Analysis", calibration=None, engine=None):
        # Serialize launch validation so concurrent bridge calls cannot both
        # reserve an active job and duplicate large dataset/model allocations.
        with self._start_lock:
            with self._lock:
                if any(job["status"] == "running" for job in self._jobs.values()):
                    raise ValueError("An analysis is already running. Wait for it to finish before starting another.")
            return self._start(dataset_id, threshold, include_sentiment, name, calibration, engine or self.engine)

    def _start(self, dataset_id, threshold, include_sentiment, name, calibration=None, engine=None):
        engine = engine or self.engine
        if isinstance(threshold, bool):
            raise ValueError("Threshold must be between zero and one")
        try:
            threshold = float(threshold)
        except (ValueError, TypeError):
            raise ValueError("Threshold must be between zero and one") from None
        if not math.isfinite(threshold) or not 0 < threshold < 1:
            raise ValueError("Threshold must be between zero and one")
        if not isinstance(include_sentiment, bool):
            raise ValueError("include_sentiment must be true or false")
        if not isinstance(name, str) or not 0 < len(name.strip()) <= 120:
            raise ValueError("Analysis name must contain 1 to 120 characters")
        if calibration is not None:
            calibration = copy.deepcopy(calibration)
            calibration["thresholds"] = validate_thresholds(calibration.get("thresholds"))
            current_model = engine.provenance(["emotion"])["models"]["emotion"]
            if calibration.get("model_provenance") != current_model:
                raise ValueError("This threshold profile belongs to different emotion model files; rerun refinement for the installed model")
        records = self.library.load_dataset(dataset_id).to_dict("records")
        dataset = self.library.manifest(dataset_id)
        if not records:
            raise ValueError("Dataset has no rows to analyze")
        tasks = ["emotion", "sentiment"] if include_sentiment else ["emotion"]
        available = engine.available_models()
        for task in tasks:
            if not available.get(task):
                raise ValueError(f"Local {task} model is unavailable. Install it using the model setup instructions first.")
        job_id = uuid.uuid4().hex
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        folder = self.root / "runs/analyses" / (stamp + "_" + job_id[:8])
        folder.mkdir(parents=True, exist_ok=False)
        job = {"id": job_id, "job_id": job_id, "name": name.strip(), "status": "running",
               "processed": 0, "total": len(records), "error": None,
               "run_id": folder.relative_to(self.root).as_posix()}
        version_file = self.root / "app_version.json"
        app_version = json.loads(version_file.read_text(encoding="utf-8"))["version"] if version_file.is_file() else "0.2.0"
        config = {"schema": SCHEMA_VERSION, "status": "running", "name": name.strip(),
                  "created_at": datetime.now(timezone.utc).isoformat(), "rows": len(records),
                  "threshold": threshold, "tasks": tasks, "app_version": app_version,
                  "task": "emotion-and-sentiment" if include_sentiment else "emotion",
                  "dataset_id": dataset_id, "dataset": dataset,
                  "model": getattr(engine, "emotion_model_id", "SamLowe/roberta-base-go_emotions"), "settings": {
                      "threshold": threshold, "include_sentiment": include_sentiment},
                  "note": "Automatic labels; model scores do not establish accuracy or emotional intensity."}
        if calibration is not None:
            config["calibration_profile"] = calibration
            config["settings"]["threshold_mode"] = "per_label_profile"
            config["settings"]["thresholds"] = calibration["thresholds"]
        else:
            config["settings"]["threshold_mode"] = "shared"
        _write_json(folder / "config.json", config)
        with self._lock:
            self._jobs[job_id] = job
        threading.Thread(target=self._run, args=(job_id, folder, records, config, engine),
                         name="analysis-" + job_id[:8], daemon=True).start()
        return self.status(job_id)

    def status(self, job_id):
        with self._lock:
            if not isinstance(job_id, str) or job_id not in self._jobs:
                raise ValueError("Unknown analysis job ID")
            return copy.deepcopy(self._jobs[job_id])

    def _update(self, job_id, **changes):
        with self._lock:
            self._jobs[job_id].update(changes)

    def _run(self, job_id, folder, records, config, engine):
        try:
            config["provenance"] = engine.provenance(config["tasks"])
            _write_json(folder / "config.json", config)
            statuses, labels, sentiments = Counter(), Counter(), Counter()
            invalid = multiple = truncated = 0
            with (folder / "predictions.jsonl").open("x", encoding="utf-8") as output:
                for index, source in enumerate(records, 1):
                    arguments = {"threshold": config["threshold"], "tasks": config["tasks"]}
                    if config.get("calibration_profile"):
                        arguments["thresholds"] = config["calibration_profile"]["thresholds"]
                    result = engine.infer(source["text"], **arguments)
                    row = {**source, **result}
                    # Retain original dataset row identity regardless of a model adapter.
                    row["post_id"] = source["post_id"]
                    row["source_row"] = source["source_row"]
                    output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    output.flush()
                    statuses[row["status"]] += 1
                    labels.update(row["labels"])
                    invalid += row["status"] == "invalid"
                    multiple += sum(label != "neutral" for label in row["labels"]) > 1
                    truncated += bool(row.get("truncated") or (row.get("sentiment") or {}).get("truncated"))
                    sentiment = row.get("sentiment")
                    if sentiment:
                        sentiments[sentiment["label"]] += 1
                    self._update(job_id, processed=index)
            summary = {"rows": len(records), "status_counts": dict(statuses),
                       "label_counts": dict(labels), "sentiment_counts": dict(sentiments),
                       "invalid_rows": invalid, "multiple_emotions": multiple, "truncated": truncated}
            _write_json(folder / "summary.json", summary)
            config.update(status="complete", completed_at=datetime.now(timezone.utc).isoformat(),
                          predictions_sha256=file_digest(folder / "predictions.jsonl"),
                          summary_sha256=file_digest(folder / "summary.json"))
            _write_json(folder / "config.json", config)
            self._update(job_id, status="complete")
        except Exception as exc:
            message = str(exc)
            config.update(status="failed", error=message, failed_at=datetime.now(timezone.utc).isoformat())
            try:
                _write_json(folder / "config.json", config)
            finally:
                self._update(job_id, status="failed", error=message)
