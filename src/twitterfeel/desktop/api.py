"""Allowlisted local JS bridge. No HTTP service or browser dependency."""
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
from types import SimpleNamespace
import uuid

from twitterfeel.common.provenance import file_digest
from twitterfeel.common.resources import annotation_rubric
from twitterfeel.common.config import app_release
from twitterfeel.labeling.label_posts import evaluate, read_posts, sample
from twitterfeel.labeling.label_schema import LABELS, validate_thresholds
from twitterfeel.workspace.research_workspace import DatasetLibrary, FindingsStore, build_findings, MAX_BYTES
from twitterfeel.inference.engine import AnalysisEngine, EMOTION_MODELS
from twitterfeel.workspace.workbench_data import (REPO_ROOT, ReviewerStore, discover_runs, load_predictions,
                            load_review, threshold_frame)


def _records(frame):
    return json.loads(frame.to_json(orient="records", force_ascii=False))


def _summary(frame):
    return {"rows": len(frame), "status_counts": frame.status.value_counts().to_dict(),
            "label_counts": dict(Counter(x for labels in frame.labels for x in labels)),
            "truncated": int(frame.truncated.sum()) if "truncated" in frame else 0,
            "multiple_emotions": int(frame.labels.map(lambda xs: sum(x != "neutral" for x in xs) > 1).sum())}


class DesktopAPI:
    def __init__(self, root=REPO_ROOT):
        self._root = Path(root).resolve()
        self._state = self._root / "runs/workbench"
        self._store = ReviewerStore(self._state / "reviews.sqlite")
        self._lock = threading.Lock()
        self._preference_lock = threading.Lock()
        self._library = DatasetLibrary(self._root)
        self._findings_store = FindingsStore(self._root)
        self._engine = AnalysisEngine(self._root)
        self._candidate_engine = None
        self._jobs = None
        self._imports = {}
        self._import_lock = threading.Lock()
        self._window = None

    def _inside(self, path):
        path = Path(path).resolve()
        if not path.is_relative_to(self._root):
            raise ValueError("Artifact must remain inside this project")
        return path

    def _files(self, kind):
        if kind == "review":
            files = (self._root / "runs").glob("**/review.csv")
        elif kind == "dataset":
            files = list((self._root / "data/raw").glob("*.xls")) + list((self._root / "data/interim").glob("*.csv"))
        elif kind == "evaluation":
            files = (self._state / "evaluations").glob("*/metrics.json")
        else:
            files = (self._root / "runs").glob("labeling_diagnostic_*/diagnostics.json")
        return {p.relative_to(self._root).as_posix(): p for p in sorted(files)
                if p.is_file() and p.resolve().is_relative_to(self._root)
                and not (kind == "review" and p.resolve().is_relative_to(self._root / "runs/public_benchmarks"))}

    def _file(self, kind, identifier):
        choices = self._files(kind)
        if identifier not in choices:
            raise ValueError("Unknown " + kind + " ID; refresh the workspace")
        return self._inside(choices[identifier])

    def _run(self, identifier):
        for run in discover_runs(self._root):
            if run["run_id"] == identifier:
                self._inside(run["predictions_path"])
                for filename, key in (("predictions.jsonl", "predictions_sha256"), ("summary.json", "summary_sha256")):
                    expected = run["config"].get(key)
                    if expected and file_digest(run["path"] / filename) != expected:
                        raise ValueError("Saved analysis artifacts changed since completion")
                return run
        raise ValueError("Unknown labeling run ID; refresh the workspace")

    def _model_path(self):
        local = self._root / "runs/model_cache/roberta-go-emotions"
        if (local / "model.safetensors").is_file():
            return local
        # Reuse an already downloaded Hugging Face snapshot; never download in UI.
        from huggingface_hub import try_to_load_from_cache
        cached = try_to_load_from_cache("SamLowe/roberta-base-go_emotions", "model.safetensors")
        return Path(cached).parent if isinstance(cached, str) else None

    def call(self, action, payload=None):
        try:
            if payload is None:
                payload = {}
            if not isinstance(payload, dict):
                raise ValueError("Payload must be an object")
            handlers = {"bootstrap": self._bootstrap, "load_run": self._load_run,
                        "load_dataset": self._load_dataset, "load_review": self._load_review,
                        "save_review": self._save_review, "export_review": self._export_review,
                        "infer": self._infer, "sandbox": self._sandbox, "evaluate": self._evaluate,
                        "experiments": self._experiments, "load_evaluation": self._load_evaluation,
                        "save_preferences": self._save_preferences,
                        "preview_import": self._preview_import, "pick_dataset": self._pick_dataset,
                        "clipboard": self._clipboard, "import_dataset": self._import_dataset,
                        "library_dataset": self._library_dataset, "start_analysis": self._start_analysis,
                        "analysis_status": self._analysis_status, "findings": self._findings,
                        "save_finding": self._save_finding, "export_findings": self._export_findings}
            handlers["create_review"] = self._create_review
            handlers["prepare_calibration"] = self._prepare_calibration
            handlers["benchmark"] = self._benchmark
            if action not in handlers:
                raise ValueError("Unknown action")
            return {"ok": True, "data": handlers[action](payload)}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _bootstrap(self, payload):
        reviews = []
        for identifier, path in self._files("review").items():
            try:
                rows = len(load_review(path))
                metadata_path = path.parent / "metadata.json"
                metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
                name = (f"Practice · {rows} posts · {metadata.get('rubric_version', 'emotion review')}"
                        if metadata.get("purpose") == "calibration" else path.parent.name)
                reviews.append({"id": identifier, "name": name, "rows": rows,
                                "purpose": metadata.get("purpose", "benchmark")})
            except (ValueError, OSError):
                continue
        runs = [{"id": r["run_id"], "name": r["config"].get("name") or r["path"].name,
                 "rows": r["config"].get("rows", 0), "model": r["config"].get("model", ""),
                 "threshold": r["config"].get("threshold", .5),
                 "threshold_mode": r["config"].get("settings", {}).get("threshold_mode", "shared"),
                 "calibration_profile_name": r["config"].get("calibration_profile", {}).get("name"),
                 "dataset_id": r["config"].get("dataset_id"),
                 "created_at": r["config"].get("created_at", "")}
                for r in discover_runs(self._root) if r["path"].resolve().is_relative_to(self._root)]
        return {"runs": runs, "reviews": reviews, "datasets": [
            {"id": key, "name": path.name} for key, path in self._files("dataset").items()],
            "labels": list(LABELS), "model_available": self._model_path() is not None,
            "model_name": "SamLowe/roberta-base-go_emotions", "preferences": self._preferences(),
            "app": self._app_version(), "library": self._library.list_datasets(),
            "annotation": self._annotation(),
            "models": self._engine.available_models(), "calibration_profiles": self._calibration_profiles(),
            "emotion_models": [{"id": key, "name": spec["name"], "model": spec["id"],
                                "available": self._engine_for(key).available_models()["emotion"]}
                               for key, spec in EMOTION_MODELS.items()]}

    def _engine_for(self, key):
        if key == "roberta":
            return self._engine
        if key != "modernbert":
            raise ValueError("Unknown emotion model; refresh the workspace")
        if self._candidate_engine is None:
            self._candidate_engine = AnalysisEngine(self._root, emotion_model=key)
        return self._candidate_engine

    def _calibration_profiles(self):
        choices = []
        paths = list((self._state / "refinements").glob("*/calibration.json"))
        paths += list((self._root / "runs/model_comparisons").glob("*/modernbert/refinement/calibration.json"))
        for path in sorted(paths, reverse=True):
            if not path.resolve().is_relative_to(self._root):
                continue
            try:
                profile = json.loads(path.read_text(encoding="utf-8"))
                report = path.parent / "report.json"
                if profile.get("schema") != "emotion-calibration-v1" or not report.is_file():
                    continue
                if file_digest(report) != profile.get("source", {}).get("report_sha256"):
                    continue
                profile["thresholds"] = validate_thresholds(profile.get("thresholds"))
                profile.update(id=path.relative_to(self._root).as_posix(), artifact_sha256=file_digest(path))
                created_at = json.loads(report.read_text(encoding="utf-8")).get("created_at", "")
                choices.append((created_at, profile))
            except (OSError, ValueError, TypeError):
                continue
        latest, seen = [], set()
        for _, profile in sorted(choices, key=lambda entry: (entry[0], entry[1]["id"]), reverse=True):
            key = (profile.get("source", {}).get("reference_sha256") or profile["id"],
                   json.dumps(profile.get("model_provenance"), sort_keys=True))
            if key not in seen:
                seen.add(key)
                latest.append(profile)
        return latest

    def _annotation(self):
        data, digest = annotation_rubric(self._root)
        return {**data, "sha256": digest}

    def _prepare_calibration(self, p):
        from twitterfeel.workspace.review_protocol import prepare_calibration
        source = self._file("review", p["review_id"])
        with self._preference_lock:
            return prepare_calibration(self._root, source)

    def _benchmark(self, p):
        from twitterfeel.evaluation.benchmark import benchmark
        from twitterfeel.workspace.review_protocol import enrich_users
        if p.get("split", "development") != "development":
            raise ValueError("Threshold comparisons use development posts only")
        reviewer = p.get("reviewer", "").strip()
        if not reviewer:
            raise ValueError("Enter a reviewer name")
        run = self._run(p["run_id"])
        source = self._file("review", p["review_id"])
        frame = self._store.merged_review(source, reviewer)
        frame = enrich_users(frame, source, self._root)
        folder = self._new_folder("benchmarks")
        snapshot = folder / "human_review.csv"
        frame.to_csv(snapshot, index=False)
        report = benchmark(snapshot, run["predictions_path"])
        report["reference"] = {"kind": "single_reviewer_draft", "reviewer": reviewer,
            "review_id": p["review_id"], "run_id": p["run_id"], "rubric": self._annotation(),
            "user_enrichment": frame.attrs.get("user_enrichment", {})}
        metadata = source.parent / "metadata.json"
        if metadata.exists():
            report["review_protocol"] = json.loads(metadata.read_text(encoding="utf-8"))
            if report["review_protocol"].get("purpose") == "calibration":
                report.setdefault("warnings", []).append("This is a practice round, not the full benchmark. Refine the rubric before making quality claims.")
        output = folder / "benchmark.json"
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        return {"path": str(output), "report": report}

    def _app_version(self):
        return app_release(self._root)

    def _preview_import(self, p):
        name, content, format = p.get("name", "Imported posts"), p.get("content", ""), p.get("format", "csv")
        preview = self._library.preview(name, content, format)
        token = uuid.uuid4().hex
        with self._import_lock:
            while len(self._imports) >= 4:
                self._imports.pop(next(iter(self._imports)))
            self._imports[token] = (content, format)
        return {"token": token, "preview": preview}

    def _pick_dataset(self, p):
        if self._window is None:
            raise ValueError("File picker is available inside the desktop application")
        import webview
        paths = self._window.create_file_dialog(webview.FileDialog.OPEN, allow_multiple=False,
            file_types=("Post datasets (*.csv;*.tsv;*.txt;*.jsonl)",))
        if not paths:
            return {"cancelled": True}
        path = Path(paths[0])
        if path.stat().st_size > MAX_BYTES:
            raise ValueError("Dataset exceeds the 20 MB limit")
        return self._preview_import({"name": path.stem, "content": path.read_bytes(),
                                     "format": path.suffix.lstrip(".").lower()})

    def _clipboard(self, p):
        if self._window is None:
            raise ValueError("Clipboard is available inside the desktop application; you can also paste into the text box")
        # WinForms clipboard access must run on the application's STA UI thread.
        from System import Action
        from System.Windows.Forms import Clipboard
        result = {}
        def read():
            result["text"] = str(Clipboard.GetText()) if Clipboard.ContainsText() else ""
        self._window.native.Invoke(Action(read))
        if len(result.get("text", "").encode("utf-8")) > MAX_BYTES:
            raise ValueError("Clipboard text exceeds the 20 MB limit")
        return result

    def _import_dataset(self, p):
        with self._import_lock:
            draft = self._imports.get(p.get("token"))
            if draft is None:
                raise ValueError("Preview the dataset again before importing")
            content, format = draft
            result = self._library.import_dataset(p.get("name", "Imported posts"), content, format,
                text_column=p.get("text_column"), user_column=p.get("user_column") or None)
            del self._imports[p["token"]]
        return result

    def _library_dataset(self, p):
        identifier = p["dataset_id"]
        frame = self._library.load_dataset(identifier)
        return {"manifest": self._library.manifest(identifier), "rows": len(frame), "preview": _records(frame.head(20))}

    def _create_review(self, p):
        identifier = p["dataset_id"]
        self._library.load_dataset(identifier)  # Verify the immutable source before sampling.
        size = p.get("size", 400)
        if isinstance(size, bool) or not isinstance(size, int) or not 2 <= size <= 2000:
            raise ValueError("Review sample size must be between 2 and 2,000")
        folder = self._state / "review_samples" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8])
        sample(SimpleNamespace(csv=self._inside(self._root / identifier / "dataset.csv"),
                               text_column="text", seed=42, size=size, out=folder))
        return {"review_id": (folder / "review.csv").relative_to(self._root).as_posix(),
                "rows": len(load_review(folder / "review.csv"))}

    def _start_analysis(self, p):
        with self._lock:
            engine = self._engine_for(p.get("emotion_model", "roberta"))
            if self._jobs is None:
                from twitterfeel.inference.jobs import AnalysisJobs
                self._jobs = AnalysisJobs(self._root, engine=self._engine)
            calibration = None
            if p.get("calibration_profile_id"):
                calibration = next((profile for profile in self._calibration_profiles()
                                    if profile["id"] == p["calibration_profile_id"]), None)
                if calibration is None:
                    raise ValueError("Unknown or changed threshold profile; refresh the workspace")
            return self._jobs.start(p["dataset_id"], threshold=calibration.get("shared_threshold", .5) if calibration else p.get("threshold", .5),
                include_sentiment=p.get("include_sentiment", True), name=p.get("name", "Analysis"),
                calibration=calibration, engine=engine)

    def _analysis_status(self, p):
        if self._jobs is None:
            raise ValueError("No analysis has been started in this session")
        return self._jobs.status(p["job_id"])

    def _findings(self, p):
        run = self._run(p["run_id"])
        can_save = run["config"].get("task") in ("emotion", "emotion-and-sentiment")
        return {"report": build_findings(run["path"]),
                "notes": self._findings_store.list(p["run_id"]) if can_save else [], "can_save": can_save}

    def _save_finding(self, p):
        self._run(p["run_id"])
        return self._findings_store.save(p["run_id"], p.get("author", ""), p.get("title", ""), p.get("note", ""))

    def _export_findings(self, p):
        data = self._findings(p)
        folder = self._state / "exports" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8])
        folder.mkdir(parents=True)
        report = data["report"]
        lines = ["# Research findings", "", "Run: " + p["run_id"], "Application: " + self._app_version()["version"], "", str(report.get("overview", "")), ""]
        for item in report.get("observations", []):
            lines.extend(["## " + item["title"], "", item["text"], ""])
        lines.extend(["## Limits", "", *["- " + x for x in report.get("limitations", [])], "", "## Researcher observations", ""])
        for note in data["notes"]:
            lines.extend(["### " + note["title"], "", note["note"], "", "By " + note["author"] + " · " + note["created_at"], ""])
        (folder / "findings.md").write_text("\n".join(lines), encoding="utf-8")
        (folder / "findings.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"path": str(folder / "findings.md")}

    def _preferences(self):
        path = self._state / "preferences.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                return {}
            return {k: v for k, v in value.items() if (k == "theme" and v in ("light", "dark", "system")) or
                    (k == "reviewer" and isinstance(v, str) and 0 < len(v.strip()) <= 80)}
        except (OSError, ValueError):
            return {}

    def _save_preferences(self, p):
        if not p or not set(p).issubset({"theme", "reviewer"}):
            raise ValueError("Only theme and reviewer preferences may be saved")
        if "theme" in p and p["theme"] not in ("light", "dark", "system"):
            raise ValueError("Theme must be light, dark, or system")
        if "reviewer" in p and (not isinstance(p["reviewer"], str) or not 0 < len(p["reviewer"].strip()) <= 80):
            raise ValueError("Reviewer name must contain 1 to 80 characters")
        with self._preference_lock:
            values = self._preferences()
            values.update(p)
            if "reviewer" in values:
                values["reviewer"] = values["reviewer"].strip()
            temporary = self._state / ("preferences-" + uuid.uuid4().hex + ".tmp")
            temporary.write_text(json.dumps(values, indent=2), encoding="utf-8")
            os.replace(temporary, self._state / "preferences.json")
        return {"saved": True, "preferences": values}

    def _load_run(self, p):
        run = self._run(p["run_id"])
        return {"id": run["run_id"], "config": run["config"], "summary": run["summary"],
                "posts": _records(load_predictions(run["predictions_path"]))}

    def _load_dataset(self, p):
        frame = read_posts(self._file("dataset", p["dataset_id"]))
        return {"rows": len(frame), "users": int(frame.user_id.nunique()) if "user_id" in frame else None,
                "repeated_texts": int(frame.text.duplicated().sum()),
                "blank_texts": int(frame.text.str.strip().eq("").sum()),
                "preview": _records(frame[["post_id", "text"]].head(100)),
                "lengths": frame.text.str.len().tolist()}

    def _load_review(self, p):
        path = self._file("review", p["review_id"])
        reviewer = p.get("reviewer", "").strip()
        frame = self._store.merged_review(path, reviewer)
        columns = ["post_id", "source_row", "text_sha256", "text", "split", "human_labels", "review_status", "notes"]
        if "split" not in frame:
            raise ValueError("Review source needs development/test split assignments")
        # Deliberately exclude model columns even if a source export has them.
        return {"rows": _records(frame[columns]), "counts": {
            s: int(frame.review_status.eq(s).sum()) for s in ("pending", "reviewed", "unclear", "invalid")},
            "metadata": json.loads((path.parent / "metadata.json").read_text(encoding="utf-8"))
                if (path.parent / "metadata.json").exists() else {"purpose": "benchmark"},
            "latest": self._store.latest(path, reviewer)}

    def _save_review(self, p):
        path = self._file("review", p["review_id"])
        revision = self._store.save(path, p["post_id"], p["text_sha256"], p["reviewer"],
                                    p["labels"], p["status"], p.get("notes", ""))
        return {"saved": True, "revision": revision}

    def _new_folder(self, category):
        path = self._state / category / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:8])
        path.mkdir(parents=True)
        return path

    def _export_review(self, p):
        path = self._file("review", p["review_id"])
        frame = self._store.merged_review(path, p["reviewer"])
        folder = self._new_folder("exports")
        output = folder / "human_review.csv"
        frame.to_csv(output, index=False)
        return {"path": str(output)}

    def _infer(self, p):
        text = p.get("text", "")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Enter some text first")
        return self._engine.infer(text, threshold=p.get("threshold", .5), tasks=["emotion"])

    def _sandbox(self, p):
        run = self._run(p["run_id"])
        frame = threshold_frame(load_predictions(run["predictions_path"]), float(p["threshold"]))
        return {"posts": _records(frame), "summary": _summary(frame)}

    def _evaluate(self, p):
        run = self._run(p["run_id"])
        source = self._file("review", p["review_id"])
        split = p.get("split", "development")
        if split not in ("development", "test"):
            raise ValueError("Split must be development or test")
        if not p.get("reviewer", "").strip():
            raise ValueError("Enter a reviewer name")
        frame = self._store.merged_review(source, p["reviewer"])
        folder = self._new_folder("evaluations")
        snapshot = folder / "human_review.csv"
        frame.to_csv(snapshot, index=False)
        output = folder / "metrics.json"
        evaluate(SimpleNamespace(review=snapshot, predictions=run["predictions_path"], split=split, out=output))
        metadata = {"run_id": run["run_id"], "reviewer": p["reviewer"],
                    "review_source": source.relative_to(self._root).as_posix(), "split": split,
                    "human_reference": "single reviewer draft; not adjudicated ground truth"}
        (folder / "evaluation.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return {"path": str(output), "metrics": json.loads(output.read_text(encoding="utf-8"))}

    def _experiments(self, p):
        evaluations, diagnostics, published, refinements, comparisons = [], [], [], [], []
        for key, path in self._files("evaluation").items():
            meta = path.parent / "evaluation.json"
            evaluations.append({"id": key, "metrics": json.loads(path.read_text(encoding="utf-8")),
                                "metadata": json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}})
        for key, path in sorted(self._files("diagnostic").items(), key=lambda item: item[1].stat().st_mtime):
            diagnostics.append({"id": key, "report": json.loads(path.read_text(encoding="utf-8"))})
        for path in sorted((self._root / "runs/public_benchmarks").glob("**/benchmark.json")):
            if not path.resolve().is_relative_to(self._root):
                continue
            try:
                report = json.loads(path.read_text(encoding="utf-8"))
                hashes = report.get("hashes", {})
                intact = all((path.parent / filename).is_file() and
                             file_digest(path.parent / filename) == hashes.get(key)
                             for filename, key in (("predictions.jsonl", "predictions_sha256"),
                                                   ("config.json", "prediction_config_sha256")))
                if report.get("ready") and intact and report.get("reference_origin") == "published_human_annotations":
                    published.append({"id": path.relative_to(self._root).as_posix(), "report": report})
            except (OSError, ValueError):
                continue
        for profile in self._calibration_profiles():
            if not profile["id"].startswith("runs/workbench/refinements/"):
                continue  # Candidate refinements are shown in their paired model comparison.
            path = self._inside(self._root / profile["id"]).parent / "report.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            refinements.append({"id": path.relative_to(self._root).as_posix(), "report": report,
                                "profile_id": profile["id"], "profile_name": profile.get("name", "Research profile")})
        baseline_reports = {file_digest(path) for path in (self._state / "refinements").glob("*/report.json")
                            if path.resolve().is_relative_to(self._root)}
        for path in sorted((self._root / "runs/model_comparisons").glob("*/comparison.json"), reverse=True):
            if not path.resolve().is_relative_to(self._root):
                continue
            try:
                report = json.loads(path.read_text(encoding="utf-8"))
                candidate = path.parent / "modernbert/refinement/report.json"
                hashes = report.get("hashes", {})
                if (report.get("schema") == "emotion-model-comparison-v1" and candidate.is_file()
                        and file_digest(candidate) == hashes.get("candidate_report_sha256")
                        and hashes.get("baseline_report_sha256") in baseline_reports):
                    comparisons.append({"id": path.relative_to(self._root).as_posix(), "report": report})
            except (OSError, ValueError, TypeError):
                continue
        return {"evaluations": list(reversed(evaluations)), "diagnostics": list(reversed(diagnostics)),
                "published": list(reversed(published)), "refinements": refinements, "model_comparisons": comparisons}

    def _load_evaluation(self, p):
        path = self._file("evaluation", p["id"])
        metrics = json.loads(path.read_text(encoding="utf-8"))
        metadata = json.loads((path.parent / "evaluation.json").read_text(encoding="utf-8"))
        run = self._run(metadata["run_id"])
        snapshot = path.parent / "human_review.csv"
        if file_digest(snapshot) != metrics["review_sha256"] or file_digest(run["predictions_path"]) != metrics["predictions_sha256"]:
            raise ValueError("Evaluation artifacts changed since scoring; cannot show reliable errors")
        reference = load_review(snapshot)
        guesses = {r["post_id"]: r for r in _records(load_predictions(run["predictions_path"]))}
        errors = []
        for row in _records(reference):
            if row["review_status"] != "reviewed" or row["split"] != metrics["split"]:
                continue
            predicted = guesses[row["post_id"]]
            expected, actual = set(row["human_labels"].split(";")), set(predicted["labels"])
            errors.append({"post_id": row["post_id"], "text": row["text"],
                           "human_labels": sorted(expected), "model_labels": sorted(actual),
                           "missed": sorted(expected-actual), "extra": sorted(actual-expected),
                           "exact_match": expected == actual})
        return {"metrics": metrics, "metadata": metadata, "errors": errors}


# Short alias for the embedded entry point and external integration probes.
API = DesktopAPI
