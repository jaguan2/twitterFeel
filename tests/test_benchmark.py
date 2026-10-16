"""Human-only development selection, paired identities, mixed emotions, uncertainty."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from benchmark import benchmark
from label_schema import LABELS, interpret, text_hash


class BenchmarkTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.review = self.root / "review.csv"
        self.predictions = self.root / "predictions.jsonl"
        self.rows = [
            {"post_id": "01", "text": "Mixed fixture", "user_id": "a", "split": "development", "human_labels": "joy;fear", "review_status": "reviewed"},
            {"post_id": "02", "text": "Joy fixture", "user_id": "a", "split": "development", "human_labels": "joy", "review_status": "reviewed"},
            {"post_id": "03", "text": "Pending fixture", "user_id": "b", "split": "development", "human_labels": "", "review_status": "pending"},
            {"post_id": "04", "text": "Held out fixture", "user_id": "c", "split": "test", "human_labels": "not inspected", "review_status": "reviewed"},
        ]
        self.records = []
        for i, row in enumerate(self.rows):
            row["text_sha256"] = text_hash(row["text"])
            scores = {label: .01 for label in LABELS}
            scores["joy"] = .8
            if i == 0:
                scores["fear"] = .4
            self.records.append({"post_id": row["post_id"], "text": row["text"],
                                 "text_sha256": row["text_sha256"], "scores": scores,
                                 "truncated": i == 0, **interpret(scores)})
        self.write()

    def write(self):
        pd.DataFrame(self.rows).to_csv(self.review, index=False)
        self.predictions.write_text("\n".join(json.dumps(row) for row in self.records), encoding="utf-8")

    def test_shared_development_sweep_recovers_mixed_emotions_and_is_deterministic(self):
        report = benchmark(self.review, self.predictions)
        self.assertTrue(report["ready"])
        self.assertEqual(report["recommended_threshold"], .4)
        self.assertEqual(report["counts"]["reviewed_rows"], 2)
        self.assertEqual(report["counts"]["excluded_review_status_counts"], {"pending": 1})
        self.assertEqual(report["bootstrap_unit"], "user")
        self.assertEqual(report, benchmark(self.review, self.predictions))
        best = next(row for row in report["results"] if row["threshold"] == .4)
        baseline = next(row for row in report["results"] if row["threshold"] == .5)
        self.assertEqual(best["micro_f1"], 1.)
        self.assertEqual(best["exact_match"], 1.)
        self.assertEqual(best["mixed_emotion"]["full_label_recovery"], 1.)
        self.assertEqual(baseline["mixed_emotion"]["label_recall"], .5)
        self.assertEqual(baseline["mixed_emotion"]["full_label_recovery"], 0.)
        self.assertEqual(best["status_counts"]["review_needed"], 1)
        self.assertEqual(best["per_label"]["fear"]["support"], 1)
        self.assertEqual(best["micro_f1_interval"]["clusters"], 1)
        json.dumps(report, allow_nan=False)

    def test_no_human_review_returns_readiness_instead_of_accuracy(self):
        for row in self.rows:
            if row["split"] == "development":
                row["review_status"] = "pending"
        self.write()
        report = benchmark(self.review, self.predictions)
        self.assertFalse(report["ready"])
        self.assertEqual(report["results"], [])
        self.assertIsNone(report["recommended_threshold"])

    def test_cannot_tune_test_or_nonfinite_thresholds(self):
        with self.assertRaisesRegex(ValueError, "restricted to development"):
            benchmark(self.review, self.predictions, split="test")
        for thresholds in ([], [.5, .5], [float("nan")], [0], [True]):
            with self.subTest(thresholds=thresholds), self.assertRaises(ValueError):
                benchmark(self.review, self.predictions, thresholds)

    def test_missing_mismatched_duplicate_predictions_are_rejected(self):
        self.records[0]["post_id"] = "missing"
        self.write()
        with self.assertRaisesRegex(ValueError, "Missing or text-mismatched"):
            benchmark(self.review, self.predictions)
        self.records[0]["post_id"] = "01"
        self.records[0]["text"] = "changed"
        self.write()
        with self.assertRaisesRegex(ValueError, "recorded hash"):
            benchmark(self.review, self.predictions)
        self.records[0]["text"] = "Mixed fixture"
        self.records.append(self.records[0])
        self.write()
        with self.assertRaisesRegex(ValueError, "Duplicate prediction"):
            benchmark(self.review, self.predictions)

    def test_cross_split_text_leakage_and_invalid_human_labels_rejected(self):
        self.rows[3]["text"] = self.rows[0]["text"]
        self.rows[3]["text_sha256"] = self.rows[0]["text_sha256"]
        self.write()
        with self.assertRaisesRegex(ValueError, "crosses development/test"):
            benchmark(self.review, self.predictions)
        self.rows.pop()
        self.rows[0]["human_labels"] = "neutral;joy"
        self.write()
        with self.assertRaisesRegex(ValueError, "Invalid human labels"):
            benchmark(self.review, self.predictions)

    def test_invalid_predictions_count_as_misses_and_missing_users_fallback(self):
        self.records[0].update(status="invalid", scores={}, labels=[])
        for row in self.rows:
            row.pop("user_id")
        self.write()
        report = benchmark(self.review, self.predictions)
        self.assertEqual(report["counts"]["invalid_predictions"], 1)
        self.assertEqual(report["bootstrap_unit"], "post")
        self.assertEqual(report["recommended_threshold"], .5)
        self.assertAlmostEqual(report["results"][0]["micro_f1"], .5)
        self.assertTrue(any("cannot account for dependence" in warning for warning in report["warnings"]))

    def test_published_policy_preserves_original_neutral_overlaps_without_weakening_default(self):
        for row in self.rows:
            row["reference_origin"] = "published_human_annotations"
        self.rows[0]["human_labels"] = "neutral;joy"
        self.write()
        with self.assertRaisesRegex(ValueError, "Invalid human labels"):
            benchmark(self.review, self.predictions)
        report = benchmark(self.review, self.predictions, published_reference=True)
        self.assertTrue(report["ready"])
        self.assertEqual(report["counts"]["per_label_support"]["neutral"], 1)
        self.assertIn("Preserve official", report["reference_policy"])


if __name__ == "__main__":
    unittest.main()
