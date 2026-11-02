"""Local workbench integrity, threshold exploration, and blind-review checks."""
import copy
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

import pandas as pd

from twitterfeel.labeling.label_schema import LABELS, SCHEMA_VERSION, interpret, text_hash
from twitterfeel.workspace.workbench_data import (ReviewerStore, discover_runs, load_predictions,
                            load_review, threshold_frame)


def prediction(identifier="p1", text="Excited and scared", **values):
    scores = {label: values.get(label, .01) for label in LABELS}
    return {"post_id": identifier, "source_row": 0, "text": text,
            "text_sha256": text_hash(text), "scores": scores,
            "truncated": False, **interpret(scores)}


class WorkbenchTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def review(self):
        path = self.root / "review.csv"
        pd.DataFrame([
            {"post_id": "001", "source_row": 7, "text": "A good day!",
             "text_sha256": text_hash("A good day!"), "split": "development",
             "human_labels": "anger", "review_status": "reviewed", "notes": "another review"},
            {"post_id": "002", "source_row": 8, "text": "The bus is here.",
             "text_sha256": text_hash("The bus is here."), "split": "test",
             "human_labels": "", "review_status": "pending", "notes": ""},
        ]).to_csv(path, index=False)
        return path

    def test_append_history_and_independent_reviewers(self):
        path = self.review()
        store = ReviewerStore(self.root / "reviews.sqlite3")
        first = store.save(path, "001", text_hash("A good day!"), "Alice", ["joy"], "reviewed", "first")
        second = store.save(path, "001", text_hash("A good day!"), "Alice", ["excitement", "joy"], "reviewed", "revised")
        store.save(path, "001", text_hash("A good day!"), "Bob", [], "unclear", "context?")
        self.assertGreater(second, first)
        alice = store.latest(path, "Alice")["001"]
        self.assertEqual(set(alice["labels"]), {"joy", "excitement"})
        self.assertEqual(alice["notes"], "revised")
        self.assertEqual(store.latest(path, "Bob")["001"]["status"], "unclear")
        with closing(sqlite3.connect(store.db_path)) as db:
            history = db.execute("SELECT notes FROM review_revisions ORDER BY revision").fetchall()
        self.assertEqual(history, [("first",), ("revised",), ("context?",)])

    def test_invalid_annotation_and_stale_hash_rejected(self):
        path = self.review()
        store = ReviewerStore(self.root / "reviews.sqlite3")
        invalid = [(["neutral", "joy"], "reviewed"), ([], "reviewed"),
                   (["made-up"], "reviewed"), (["joy"], "pending"),
                   (["joy"], "unclear"), (["fear"], "invalid")]
        for labels, status in invalid:
            with self.subTest(labels=labels, status=status), self.assertRaises(ValueError):
                store.save(path, "001", text_hash("A good day!"), "Alice", labels, status, "")
        with self.assertRaisesRegex(ValueError, "stale"):
            store.save(path, "001", text_hash("changed"), "Alice", ["joy"], "reviewed", "")
        self.assertEqual(store.latest(path, "Alice"), {})

    def test_export_preserves_identity_and_never_inherits_another_review(self):
        path = self.review()
        store = ReviewerStore(self.root / "reviews.sqlite3")
        blank = store.merged_review(path, "New reviewer")
        self.assertTrue(blank.review_status.eq("pending").all())
        self.assertTrue(blank.human_labels.eq("").all())
        store.save(path, "001", text_hash("A good day!"), "Alice", ["joy"], "reviewed", "clear")
        merged = store.merged_review(path, "Alice")
        original = load_review(path)
        for column in ("post_id", "source_row", "text_sha256", "text", "split"):
            self.assertEqual(merged[column].tolist(), original[column].tolist())
        self.assertEqual(merged.human_labels.tolist(), ["joy", ""])
        exported = self.root / "export.csv"
        merged.to_csv(exported, index=False)
        reopened = load_review(exported)
        self.assertEqual(reopened.post_id.tolist(), ["001", "002"])
        self.assertEqual(reopened.text_sha256.tolist(), original.text_sha256.tolist())

    def test_changed_source_text_cannot_inherit_old_labels(self):
        path = self.review()
        store = ReviewerStore(self.root / "reviews.sqlite3")
        store.save(path, "001", text_hash("A good day!"), "Alice", ["joy"], "reviewed", "")
        source = load_review(path)
        source.loc[0, "text"] = "Changed text"
        source.loc[0, "text_sha256"] = text_hash("Changed text")
        source.to_csv(path, index=False)
        self.assertEqual(store.latest(path, "Alice"), {})
        self.assertTrue(store.merged_review(path, "Alice").review_status.eq("pending").all())

    def test_threshold_exploration_preserves_original_scores_and_truncation(self):
        records = [prediction(excitement=.8, fear=.6), prediction("p2", "Long text", joy=.6)]
        records[1]["truncated"] = True
        records.append({"post_id": "empty", "source_row": 2, "text": "", "text_sha256": text_hash(""),
                        "scores": {}, "labels": [], "status": "invalid", "truncated": False,
                        "review_reason": "empty_text", "related_groups": []})
        df = pd.DataFrame(records)
        before = copy.deepcopy(df.to_dict("records"))
        adjusted = threshold_frame(df, .7)
        self.assertEqual(adjusted.iloc[0].labels, ["excitement"])
        self.assertEqual(adjusted.iloc[1].status, "review_needed")
        self.assertIn("text_truncated", adjusted.iloc[1].review_reason)
        self.assertEqual(adjusted.iloc[2].status, "invalid")
        adjusted.iloc[0].scores["fear"] = .99
        self.assertEqual(df.to_dict("records"), before)

    def test_prediction_loader_rejects_duplicate_identity_and_mutated_text(self):
        path = self.root / "predictions.jsonl"
        record = prediction(joy=.8)
        path.write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            load_predictions(path)
        record["text"] = "Edited"
        path.write_text(json.dumps(record) + "\n")
        with self.assertRaisesRegex(ValueError, "hash"):
            load_predictions(path)

    def test_discovery_separates_forecasting_incomplete_and_labeling_runs(self):
        for name, config in [
            ("forecast", {"status": "complete", "task": "binary"}),
            ("label", {"status": "complete", "schema": SCHEMA_VERSION}),
            ("unfinished", {"status": "running", "schema": SCHEMA_VERSION}),
        ]:
            directory = self.root / "runs" / name
            directory.mkdir(parents=True)
            (directory / "config.json").write_text(json.dumps(config))
            (directory / "summary.json").write_text(json.dumps({"rows": 1}))
            (directory / "predictions.jsonl").write_text(json.dumps(prediction()) + "\n")
        damaged = self.root / "runs" / "broken"
        damaged.mkdir()
        (damaged / "config.json").write_text("broken json")
        found = discover_runs(self.root)
        self.assertEqual([item["run_id"] for item in found], ["runs/label"])
        self.assertEqual(len(load_predictions(found[0]["predictions_path"])), 1)


if __name__ == "__main__":
    unittest.main()
