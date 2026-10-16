"""Completed runs are discoverable, failures stay explicit, and inputs persist."""
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from analysis_jobs import AnalysisJobs
from emotion_schema import file_digest
from label_schema import LABELS, interpret, text_hash
from research_workspace import DatasetLibrary
from workbench_data import discover_runs, load_predictions


class FakeEngine:
    def __init__(self, failure=False):
        self.failure = failure
        self.tasks = []

    def available_models(self):
        return {"emotion": True, "sentiment": True}

    def provenance(self, tasks):
        return {"models": {"emotion": {"id": "fixture", "revision": "pinned"}}, "tasks": tasks}

    def infer(self, text, threshold, tasks, thresholds=None):
        self.tasks.append(tasks)
        if self.failure:
            raise RuntimeError("deliberate model failure")
        scores = {label: .9 if label in ("joy", "excitement") else .01 for label in LABELS}
        result = {"text": text, "text_sha256": text_hash(text), "scores": scores,
                  "truncated": False, "sentiment": None, **interpret(scores, threshold, thresholds)}
        if not text.strip():
            result.update(status="invalid", scores={}, labels=[], review_reason="blank_text", related_groups=[])
        elif "sentiment" in tasks:
            result["sentiment"] = {"label": "positive", "scores": {"negative": .01, "neutral": .01, "positive": .98}}
        return result


class AnalysisJobsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "app_version.json").write_text('{"version":"0.2.0"}')
        self.library = DatasetLibrary(self.root)
        self.dataset = self.library.import_dataset("Study", "hello\n\nworld", "txt")

    def wait(self, jobs, job):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            status = jobs.status(job["id"])
            if status["status"] in ("complete", "failed"):
                return status
            threading.Event().wait(.01)
        self.fail("Job did not finish")

    def test_complete_run_preserves_blanks_identity_and_provenance(self):
        jobs = AnalysisJobs(self.root, FakeEngine())
        job = self.wait(jobs, jobs.start(self.dataset["id"], name="Study run"))
        self.assertEqual((job["status"], job["processed"], job["total"]), ("complete", 3, 3))
        folder = self.root / job["run_id"]
        config = json.loads((folder / "config.json").read_text())
        summary = json.loads((folder / "summary.json").read_text())
        self.assertEqual(config["provenance"]["models"]["emotion"]["revision"], "pinned")
        self.assertEqual(config["dataset"]["dataset_sha256"], self.dataset["dataset_sha256"])
        self.assertEqual(config["app_version"], "0.2.0")
        self.assertEqual(config["task"], "emotion-and-sentiment")
        self.assertEqual(config["predictions_sha256"], file_digest(folder / "predictions.jsonl"))
        self.assertEqual(config["summary_sha256"], file_digest(folder / "summary.json"))
        self.assertEqual(summary["sentiment_counts"], {"positive": 2})
        self.assertEqual((summary["invalid_rows"], summary["multiple_emotions"]), (1, 2))
        frame = load_predictions(folder / "predictions.jsonl")
        self.assertEqual(frame.text.tolist(), ["hello", "", "world"])
        self.assertEqual(frame.post_id.tolist(), self.library.load_dataset(self.dataset["id"]).post_id.tolist())
        self.assertEqual(len(discover_runs(self.root)), 1)
        job["status"] = "changed outside"
        self.assertEqual(jobs.status(job["id"])["status"], "complete")

    def test_failed_run_is_recorded_and_never_discovered(self):
        jobs = AnalysisJobs(self.root, FakeEngine(failure=True))
        job = self.wait(jobs, jobs.start(self.dataset["id"]))
        self.assertEqual(job["status"], "failed")
        self.assertIn("deliberate model failure", job["error"])
        self.assertEqual(discover_runs(self.root), [])
        config = json.loads((self.root / job["run_id"] / "config.json").read_text())
        self.assertEqual(config["status"], "failed")

    def test_repeated_runs_do_not_overwrite_and_emotion_only_is_explicit(self):
        engine = FakeEngine()
        jobs = AnalysisJobs(self.root, engine)
        first = self.wait(jobs, jobs.start(self.dataset["id"], include_sentiment=False))
        path = self.root / first["run_id"] / "predictions.jsonl"
        original = path.read_bytes()
        second = self.wait(jobs, jobs.start(self.dataset["id"]))
        self.assertNotEqual(first["run_id"], second["run_id"])
        self.assertEqual(path.read_bytes(), original)
        config = json.loads((path.parent / "config.json").read_text())
        self.assertEqual(config["tasks"], ["emotion"])
        self.assertEqual(config["task"], "emotion")

    def test_only_one_active_job_is_allowed(self):
        entered, release = threading.Event(), threading.Event()
        engine = FakeEngine()
        original = engine.provenance

        def blocking_provenance(tasks):
            entered.set()
            release.wait(5)
            return original(tasks)

        engine.provenance = blocking_provenance
        jobs = AnalysisJobs(self.root, engine)
        job = jobs.start(self.dataset["id"])
        self.assertTrue(entered.wait(2))
        try:
            with self.assertRaisesRegex(ValueError, "already running"):
                jobs.start(self.dataset["id"])
        finally:
            release.set()
        self.assertEqual(self.wait(jobs, job)["status"], "complete")

    def test_invalid_settings_unknown_datasets_and_missing_models_do_not_start(self):
        jobs = AnalysisJobs(self.root, FakeEngine())
        for threshold in (float("nan"), 0, 1, True, "bad"):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                jobs.start(self.dataset["id"], threshold=threshold)
        with self.assertRaises(ValueError):
            jobs.start("../../private")
        jobs.engine.available_models = lambda: {"emotion": True, "sentiment": False}
        with self.assertRaisesRegex(ValueError, "sentiment model is unavailable"):
            jobs.start(self.dataset["id"])
        self.assertEqual(jobs._jobs, {})
        with self.assertRaises(ValueError):
            jobs.status("unknown")

    def test_profile_is_applied_saved_and_rejects_another_model(self):
        jobs = AnalysisJobs(self.root, FakeEngine())
        profile = {"name": "Fixture profile", "thresholds": dict.fromkeys(LABELS, .95),
                   "model_provenance": jobs.engine.provenance(["emotion"])["models"]["emotion"]}
        job = self.wait(jobs, jobs.start(self.dataset["id"], calibration=profile))
        self.assertEqual(job["status"], "complete", job)
        folder = self.root / job["run_id"]
        rows = load_predictions(folder / "predictions.jsonl")
        self.assertEqual(rows.iloc[0].labels, [])
        self.assertEqual(rows.iloc[0].sentiment["label"], "positive")
        config = json.loads((folder / "config.json").read_text())
        self.assertEqual(config["calibration_profile"]["name"], "Fixture profile")
        self.assertEqual(config["settings"]["threshold_mode"], "per_label_profile")
        profile["model_provenance"] = {"id": "different"}
        with self.assertRaisesRegex(ValueError, "different emotion model"):
            jobs.start(self.dataset["id"], calibration=profile)

    def test_selected_engine_is_frozen_for_the_run_and_does_not_replace_default(self):
        class AlternateEngine(FakeEngine):
            emotion_model_id = "fixture/alternate"

            def infer(self, text, threshold, tasks, thresholds=None):
                result = super().infer(text, threshold, tasks, thresholds)
                if text.strip():
                    result["scores"] = {label: .9 if label == "fear" else .01 for label in LABELS}
                    result.update(interpret(result["scores"], threshold, thresholds))
                return result

        jobs = AnalysisJobs(self.root, FakeEngine())
        alternate = self.wait(jobs, jobs.start(self.dataset["id"], engine=AlternateEngine()))
        folder = self.root / alternate["run_id"]
        self.assertEqual(load_predictions(folder / "predictions.jsonl").iloc[0].labels, ["fear"])
        self.assertEqual(json.loads((folder / "config.json").read_text())["model"], "fixture/alternate")
        default = self.wait(jobs, jobs.start(self.dataset["id"]))
        self.assertEqual(set(load_predictions(self.root / default["run_id"] / "predictions.jsonl").iloc[0].labels), {"joy", "excitement"})


if __name__ == "__main__":
    unittest.main()
