"""Native bridge action, identity, review isolation, and artifact integrity checks."""
import json
from pathlib import Path
from importlib.resources import files
import sys
import tempfile
import unittest

import pandas as pd

from twitterfeel.desktop.api import DesktopAPI
from twitterfeel.labeling.label_schema import LABELS, SCHEMA_VERSION, interpret, text_hash


class DesktopAPITest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        run = self.root / "runs/test"
        run.mkdir(parents=True)
        self.scores = {label: .8 if label == "joy" else .01 for label in LABELS}
        self.record = {"post_id": "001", "source_row": 9, "text": "Happy today!",
                       "text_sha256": text_hash("Happy today!"), "scores": self.scores,
                       "truncated": False, **interpret(self.scores)}
        self.predictions = run / "predictions.jsonl"
        self.predictions.write_text(json.dumps(self.record) + "\n")
        (run / "config.json").write_text(json.dumps({"schema": SCHEMA_VERSION,
            "status": "complete", "rows": 1, "model": "test-model", "threshold": .5}))
        (run / "summary.json").write_text(json.dumps({"rows": 1}))
        review = self.root / "runs/review"
        review.mkdir()
        self.review_path = review / "review.csv"
        pd.DataFrame([{"post_id": "001", "source_row": 9, "text": "Happy today!",
                       "text_sha256": text_hash("Happy today!"), "split": "development",
                       "human_labels": "anger", "review_status": "reviewed", "notes": "old",
                       "emotion": "joy", "scores": "must stay hidden"}]).to_csv(self.review_path, index=False)
        raw = self.root / "data/raw"
        raw.mkdir(parents=True)
        pd.DataFrame([{"post_id": "raw1", "user_id": "person", "post_text": "Original!",
                       "cleaned_text": "origin"}]).to_csv(raw / "sample.xls", index=False)
        self.api = DesktopAPI(self.root)
        self.api._model_path = lambda: None

    def test_calibration_is_blind_and_unreviewed_benchmark_has_no_scores(self):
        rubric = self.root / "src/annotation_rubric.json"
        rubric.parent.mkdir()
        rubric.write_bytes(files("twitterfeel.labeling").joinpath("annotation_rubric.json").read_bytes())
        practice = self.call("prepare_calibration", review_id="runs/review/review.csv")
        self.assertEqual(practice["rows"], 1)
        repeat = self.call("prepare_calibration", review_id="runs/review/review.csv")
        self.assertTrue(repeat["existing"])
        self.assertEqual(practice["review_id"], repeat["review_id"])
        data = self.call("load_review", review_id=practice["review_id"], reviewer="Alice")
        self.assertEqual(data["metadata"]["purpose"], "calibration")
        self.assertFalse(data["rows"][0]["human_labels"])
        report = self.call("benchmark", run_id="runs/test", review_id="runs/review/review.csv", reviewer="Alice")["report"]
        self.assertFalse(report["ready"])
        self.assertEqual(report["results"], [])
        self.assertFalse(self.api.call("benchmark", {"run_id": "runs/test", "review_id": "runs/review/review.csv", "reviewer": "Alice", "split": "test"})["ok"])

    def call(self, action, **payload):
        result = self.api.call(action, payload)
        self.assertTrue(result["ok"], result)
        # Ensure every response is suitable for the native JSON bridge.
        json.dumps(result, allow_nan=False)
        return result["data"]

    def save(self, reviewer="Alice", labels=None):
        return self.call("save_review", review_id="runs/review/review.csv", reviewer=reviewer,
                         post_id="001", text_sha256=text_hash("Happy today!"),
                         labels=labels or ["joy"], status="reviewed", notes="clear")

    def test_bootstrap_dataset_and_no_arbitrary_paths_or_actions(self):
        boot = self.call("bootstrap")
        self.assertEqual(boot["runs"][0]["id"], "runs/test")
        self.assertEqual(boot["reviews"][0]["rows"], 1)
        self.assertFalse(boot["model_available"])
        data = self.call("load_dataset", dataset_id="data/raw/sample.xls")
        self.assertEqual(data["preview"][0]["text"], "Original!")
        self.assertEqual(data["users"], 1)
        for action, payload in [("__dict__", {}), ("load_run", {"run_id": "../../private"}),
                                ("load_dataset", {"dataset_id": str(self.root / "data/raw/sample.xls")}),
                                ("load_review", {"review_id": "../review.csv"})]:
            with self.subTest(action=action, payload=payload):
                self.assertFalse(self.api.call(action, payload)["ok"])

    def test_blind_review_save_history_and_identity_preserving_export(self):
        args = {"review_id": "runs/review/review.csv", "reviewer": "Alice"}
        initial = self.call("load_review", **args)
        self.assertEqual(initial["rows"][0]["human_labels"], "")
        self.assertEqual(initial["rows"][0]["review_status"], "pending")
        self.assertNotIn("scores", initial["rows"][0])
        self.assertNotIn("emotion", initial["rows"][0])
        first = self.save()
        second = self.save(labels=["joy", "excitement"])
        self.assertGreater(second["revision"], first["revision"])
        loaded = self.call("load_review", **args)
        self.assertEqual(loaded["counts"]["reviewed"], 1)
        other = self.call("load_review", review_id=args["review_id"], reviewer="Bob")
        self.assertEqual(other["counts"]["pending"], 1)
        exported = Path(self.call("export_review", **args)["path"])
        self.assertTrue(exported.is_relative_to(self.root / "runs/workbench/exports"))
        data = pd.read_csv(exported, dtype={"post_id": str})
        self.assertEqual(data.post_id.iloc[0], "001")
        self.assertEqual(data.text_sha256.iloc[0], text_hash("Happy today!"))

    def test_review_rejects_stale_hash_and_neutral_combo(self):
        payload = {"review_id": "runs/review/review.csv", "reviewer": "Alice", "post_id": "001",
                   "text_sha256": text_hash("wrong"), "labels": ["joy"], "status": "reviewed", "notes": ""}
        self.assertFalse(self.api.call("save_review", payload)["ok"])
        payload.update(text_sha256=text_hash("Happy today!"), labels=["neutral", "joy"])
        self.assertFalse(self.api.call("save_review", payload)["ok"])

    def test_threshold_sandbox_preserves_saved_run(self):
        before = self.predictions.read_bytes()
        data = self.call("sandbox", run_id="runs/test", threshold=.9)
        self.assertEqual(data["posts"][0]["status"], "uncertain")
        self.assertEqual(data["summary"]["status_counts"], {"uncertain": 1})
        self.assertEqual(self.predictions.read_bytes(), before)

    def test_evaluation_snapshot_errors_and_tamper_detection(self):
        self.save(labels=["fear"])
        result = self.call("evaluate", run_id="runs/test", review_id="runs/review/review.csv",
                           reviewer="Alice", split="development")
        self.assertEqual(result["metrics"]["micro_f1"], 0)
        history = self.call("experiments")
        self.assertEqual(len(history["evaluations"]), 1)
        identifier = history["evaluations"][0]["id"]
        detail = self.call("load_evaluation", id=identifier)
        self.assertEqual(detail["errors"][0]["missed"], ["fear"])
        self.assertEqual(detail["errors"][0]["extra"], ["joy"])
        self.predictions.write_text(self.predictions.read_text() + "\n")
        self.assertFalse(self.api.call("load_evaluation", {"id": identifier})["ok"])

    def test_invalid_inference_does_not_load_model(self):
        self.assertFalse(self.api.call("infer", {"text": ""})["ok"])
        self.assertFalse(self.api.call("infer", {"text": "hello", "threshold": 1})["ok"])
        self.assertEqual(self.api._engine._models, {})

    def test_preferences_are_validated_and_survive_new_api_instance(self):
        self.call("save_preferences", theme="dark", reviewer="  Alice  ")
        reopened = DesktopAPI(self.root)
        reopened._model_path = lambda: None
        self.assertEqual(reopened.call("bootstrap")["data"]["preferences"],
                         {"theme": "dark", "reviewer": "Alice"})
        self.call("save_preferences", theme="system")
        self.assertEqual(DesktopAPI(self.root)._preferences(),
                         {"theme": "system", "reviewer": "Alice"})
        for payload in ({"path": "../outside"}, {"theme": "purple"}, {"reviewer": ""}, {"reviewer": "x" * 81}):
            with self.subTest(payload=payload):
                self.assertFalse(self.api.call("save_preferences", payload)["ok"])


if __name__ == "__main__":
    unittest.main()
