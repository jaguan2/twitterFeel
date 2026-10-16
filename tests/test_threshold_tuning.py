"""Grouped OOF isolation, training support gates, held-out labels, artifact integrity."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from emotion_schema import file_digest
from label_schema import LABELS, interpret, text_hash
from threshold_tuning import apply_calibration, fit_macro_per_label, fit_per_label, grouped_folds, refine


class ThresholdTuningTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.reference = self.root / "reference.csv"
        self.predictions = self.root / "predictions.jsonl"
        self.rows, predictions = [], []
        for index in range(160):
            text = "duplicated" if index < 2 else f"fixture {index}"
            label = "joy" if index < 2 or index % 2 == 0 else "fear"
            row = {"post_id": str(index), "text": text, "text_sha256": text_hash(text),
                   "human_labels": label, "review_status": "reviewed", "split": "development"}
            self.rows.append(row)
            scores = {name: .01 for name in LABELS}
            scores["joy"] = .4 if label == "joy" else .2
            scores["fear"] = .8 if label == "fear" else .1
            predictions.append({"post_id": str(index), "text": text, "text_sha256": text_hash(text),
                                "scores": scores, "truncated": index == 0, **interpret(scores)})
        self.rows.append({"post_id": "heldout", "text": "reserved", "text_sha256": text_hash("reserved"),
                          "human_labels": "invalid test label never parsed", "review_status": "reviewed", "split": "test"})
        pd.DataFrame(self.rows).to_csv(self.reference, index=False)
        self.predictions.write_text("\n".join(map(json.dumps, predictions)), encoding="utf-8")
        (self.root / "config.json").write_text(json.dumps({"predictions_sha256": file_digest(self.predictions),
            "provenance": {"models": {"emotion": {"id": "fixture", "files_sha256": {"weights": "pinned"}}}}}))

    def test_duplicate_texts_share_folds(self):
        hashes = [row["text_sha256"] for row in self.rows[:-1]]
        assignment = grouped_folds(hashes)
        self.assertEqual(assignment[0], assignment[1])
        np.testing.assert_array_equal(assignment, grouped_folds(hashes))
        for fold in range(5):
            training = {h for h, f in zip(hashes, assignment) if f != fold}
            validation = {h for h, f in zip(hashes, assignment) if f == fold}
            self.assertFalse(training & validation)

    def test_sparse_labels_fall_back_to_training_shared_threshold(self):
        truth = np.zeros((100, len(LABELS)), dtype=bool)
        scores = np.full(truth.shape, .1)
        truth[:24, 0] = True
        scores[:24, 0] = .3
        truth[:50, 1] = True
        scores[:50, 1] = .8
        thresholds, details = fit_per_label(scores, truth, shared=.4)
        self.assertEqual(thresholds[0], .4)
        self.assertIn(LABELS[0], details["fallback_labels"])
        self.assertNotIn(LABELS[1], details["fallback_labels"])
        truth[:50, 2] = True
        scores[:50, 2] = .3
        macro = fit_macro_per_label(scores, truth, shared=.4)
        self.assertEqual(macro[0], .4)
        self.assertEqual(macro[2], .3)

    def test_test_labels_are_unused_and_artifact_hashes_bound(self):
        original_config = (self.root / "config.json").read_bytes()
        report = refine(self.reference, self.predictions, self.root / "run1")
        self.assertEqual(report["rows"], 160)
        self.assertFalse(report["test_used"])
        self.rows[-1]["human_labels"] = "changed to any arbitrary value"
        pd.DataFrame(self.rows).to_csv(self.reference, index=False)
        second = refine(self.reference, self.predictions, self.root / "run2")
        self.assertEqual(report["methods"], second["methods"])
        self.assertEqual(report["folds"], second["folds"])
        self.assertEqual(original_config, (self.root / "config.json").read_bytes())
        calibration = json.loads((self.root / "run1/calibration.json").read_text())
        self.assertEqual(calibration["source"]["report_sha256"], file_digest(self.root / "run1/report.json"))
        self.assertEqual(report["oof_predictions_sha256"], file_digest(self.root / "run1/oof_predictions.jsonl"))
        self.assertEqual(set(calibration["thresholds"]), set(LABELS))
        self.assertTrue(calibration["provisional"])
        oof = [json.loads(line) for line in (self.root / "run1/oof_predictions.jsonl").read_text().splitlines()]
        self.assertNotIn("heldout", [row["post_id"] for row in oof])
        self.assertEqual(oof[0]["fold"], oof[1]["fold"])
        self.assertEqual(report["methods"]["fixed_0.5"]["status_counts"]["review_needed"], 1)
        with self.assertRaisesRegex(ValueError, "fresh output"):
            refine(self.reference, self.predictions, self.root / "run1")

    def test_neutral_conflicts_remain_review_needed(self):
        scores = {label: .01 for label in LABELS}
        scores.update(joy=.4, neutral=.4)
        result = apply_calibration(scores, {label: .3 for label in LABELS})
        self.assertEqual(result["status"], "review_needed")
        self.assertEqual(result["review_reason"], "neutral_emotion_conflict")

    def test_modified_predictions_are_rejected(self):
        self.predictions.write_text(self.predictions.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "recorded configuration hash"):
            refine(self.reference, self.predictions, self.root / "invalid")


if __name__ == "__main__":
    unittest.main()
