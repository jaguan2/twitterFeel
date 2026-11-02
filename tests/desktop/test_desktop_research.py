"""Exercise the native research workflow without reading the real clipboard."""
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

from twitterfeel.desktop.api import DesktopAPI
from twitterfeel.common.provenance import file_digest
from twitterfeel.labeling.label_schema import LABELS, interpret, text_hash


class FakeEngine:
    def available_models(self):
        return {"emotion": True, "sentiment": True}

    def provenance(self, tasks):
        return {"models": {task: {"id": "test-" + task} for task in tasks}}

    def infer(self, text, threshold=.5, tasks=None, thresholds=None):
        scores = {label: .8 if label == "joy" else .01 for label in LABELS}
        return {"text": text, "text_sha256": text_hash(text), "scores": scores,
                "truncated": False, **interpret(scores, threshold, thresholds),
                "sentiment": {"label": "positive", "scores": {"positive": .9, "negative": .05, "neutral": .05}}}


class ResearchBridgeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "app_version.json").write_text('{"version":"0.2.0","changes":[]}')
        self.api = DesktopAPI(self.root)
        self.api._engine = FakeEngine()

    def call(self, action, **payload):
        reply = self.api.call(action, payload)
        self.assertTrue(reply["ok"], reply)
        json.dumps(reply, allow_nan=False)
        return reply["data"]

    def test_import_analyze_review_observe_export(self):
        draft = self.call("preview_import", name="My posts", content="body,person\nSo happy!,one\nExcited today,two\n", format="csv")
        self.assertEqual(draft["preview"]["columns"], ["body", "person"])
        self.assertFalse(self.api.call("import_dataset", {"token": draft["token"], "name": "My posts"})["ok"])
        dataset = self.call("import_dataset", token=draft["token"], name="My posts", text_column="body", user_column="person")
        self.assertEqual(dataset["version"], 1)
        self.assertFalse(self.api.call("import_dataset", {"token": draft["token"]})["ok"])
        loaded = self.call("library_dataset", dataset_id=dataset["id"])
        self.assertEqual(loaded["preview"][0]["text"], "So happy!")
        job = self.call("start_analysis", dataset_id=dataset["id"], include_sentiment=True)
        for _ in range(200):
            job = self.call("analysis_status", job_id=job["id"])
            if job["status"] != "running":
                break
            time.sleep(.01)
        self.assertEqual(job["status"], "complete", job)
        run = self.call("load_run", run_id=job["run_id"])
        self.assertEqual(run["summary"]["sentiment_counts"], {"positive": 2})
        finding = self.call("findings", run_id=job["run_id"])
        self.assertTrue(finding["can_save"])
        self.call("save_finding", run_id=job["run_id"], author="Tester", title="Check mixed emotions", note="Inspect examples before accepting these labels.")
        finding = self.call("findings", run_id=job["run_id"])
        self.assertEqual(len(finding["notes"]), 1)
        exported = self.call("export_findings", run_id=job["run_id"])
        self.assertIn("Inspect examples", Path(exported["path"]).read_text(encoding="utf-8"))
        review = self.call("create_review", dataset_id=dataset["id"], size=2)
        self.assertEqual(review["rows"], 2)
        rows = self.call("load_review", review_id=review["review_id"], reviewer="Tester")["rows"]
        self.assertTrue(all(row["review_status"] == "pending" and not row["human_labels"] for row in rows))
        predictions = self.root / job["run_id"] / "predictions.jsonl"
        predictions.write_text(predictions.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        self.assertFalse(self.api.call("load_run", {"run_id": job["run_id"]})["ok"])
        self.assertFalse(self.api.call("findings", {"run_id": job["run_id"]})["ok"])

    def test_invalid_drafts_and_native_only_actions(self):
        for action, payload in [("clipboard", {}), ("pick_dataset", {}),
                                ("library_dataset", {"dataset_id": "../../outside"}),
                                ("preview_import", {"content": "", "name": "Empty"})]:
            self.assertFalse(self.api.call(action, payload)["ok"])

    def test_explicit_research_profile_applies_and_changed_report_is_rejected(self):
        folder = self.root / "runs/workbench/refinements/fixture"
        folder.mkdir(parents=True)
        report_path = folder / "report.json"
        report_path.write_text('{"kind":"development_refinement"}')
        profile = {"schema": "emotion-calibration-v1", "name": "Research cutoffs",
                   "method": "regularized_per_label", "shared_threshold": .3,
                   "provisional": True, "thresholds": dict.fromkeys(LABELS, .95),
                   "model_provenance": self.api._engine.provenance(["emotion"])["models"]["emotion"],
                   "source": {"report_sha256": file_digest(report_path)}}
        (folder / "calibration.json").write_text(json.dumps(profile))
        boot = self.call("bootstrap")
        self.assertEqual(len(boot["calibration_profiles"]), 1)
        identifier = boot["calibration_profiles"][0]["id"]
        draft = self.call("preview_import", name="Profile check", content="Happy!", format="txt")
        dataset = self.call("import_dataset", token=draft["token"], name="Profile check", text_column="text")
        job = self.call("start_analysis", dataset_id=dataset["id"], calibration_profile_id=identifier)
        for _ in range(200):
            job = self.call("analysis_status", job_id=job["id"])
            if job["status"] != "running":
                break
            time.sleep(.01)
        self.assertEqual(job["status"], "complete", job)
        run = self.call("load_run", run_id=job["run_id"])
        self.assertEqual(run["posts"][0]["labels"], [])
        self.assertEqual(run["config"]["threshold"], .3)
        self.assertEqual(run["config"]["calibration_profile"]["artifact_sha256"], file_digest(folder / "calibration.json"))
        report_path.write_text('{"kind":"changed"}')
        self.assertEqual(self.call("bootstrap")["calibration_profiles"], [])
        self.assertFalse(self.api.call("start_analysis", {"dataset_id": dataset["id"], "calibration_profile_id": identifier})["ok"])

    def test_model_comparison_requires_both_recorded_source_reports(self):
        baseline = self.root / "runs/workbench/refinements/baseline/report.json"
        baseline.parent.mkdir(parents=True)
        baseline.write_text('{"methods":{}}')
        folder = self.root / "runs/model_comparisons/check"
        candidate = folder / "modernbert/refinement/report.json"
        candidate.parent.mkdir(parents=True)
        candidate.write_text('{"methods":{"candidate":{}}}')
        report = {"schema": "emotion-model-comparison-v1", "hashes": {
            "baseline_report_sha256": file_digest(baseline), "candidate_report_sha256": file_digest(candidate)}}
        (folder / "comparison.json").write_text(json.dumps(report))
        self.assertEqual(len(self.call("experiments")["model_comparisons"]), 1)
        candidate.write_text('{"methods":{"changed":{}}}')
        self.assertEqual(self.call("experiments")["model_comparisons"], [])


if __name__ == "__main__":
    unittest.main()
