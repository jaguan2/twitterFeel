"""Same-reference/fold model pairing and strict pinned candidate architecture."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import pandas as pd

from twitterfeel.evaluation.compare_models import compare_oof, run_comparison, validate_model_config
from twitterfeel.common.provenance import file_digest
from twitterfeel.labeling.label_schema import LABELS, interpret, text_hash
from twitterfeel.evaluation.threshold_tuning import refine


class FixtureCandidate:
    def __init__(self):
        self.seen = []

    def provenance(self):
        return {"models": {"emotion": {"id": "fixture-candidate", "revision": "pinned"}}}

    def infer_batch(self, texts):
        self.seen.extend(texts)
        output = []
        for text in texts:
            scores = {label: .9 if label == "joy" else .01 for label in LABELS}
            output.append({"scores": scores, "truncated": False, **interpret(scores)})
        return output


class CompareModelsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.reference = self.root / "reference.csv"
        self.predictions = self.root / "predictions.jsonl"
        rows, predictions = [], []
        for index in range(20):
            text = f"human reference {index}"
            rows.append({"post_id": str(index), "text": text, "text_sha256": text_hash(text),
                         "human_labels": "joy", "review_status": "reviewed", "split": "development"})
            scores = {label: .4 if label == "joy" else .01 for label in LABELS}
            predictions.append({"post_id": str(index), "text": text, "text_sha256": text_hash(text),
                                "scores": scores, "truncated": False, **interpret(scores)})
        rows.append({"post_id": "test", "text": "NEVER SCORE THIS", "text_sha256": text_hash("NEVER SCORE THIS"),
                     "human_labels": "not parsed", "review_status": "reviewed", "split": "test"})
        pd.DataFrame(rows).to_csv(self.reference, index=False)
        self.predictions.write_text("\n".join(map(json.dumps, predictions)), encoding="utf-8")
        (self.root / "config.json").write_text(json.dumps({"predictions_sha256": file_digest(self.predictions),
            "provenance": {"models": {"emotion": {"id": "fixture-baseline"}}}}))
        self.baseline = self.root / "baseline"
        refine(self.reference, self.predictions, self.baseline)

    def test_candidate_must_have_exact_multilabel_vocabulary_and_architecture(self):
        config = SimpleNamespace(id2label=dict(enumerate(LABELS)), problem_type="multi_label_classification", model_type="modernbert")
        self.assertEqual(validate_model_config(config), dict(enumerate(LABELS)))
        config.problem_type = "single_label_classification"
        with self.assertRaisesRegex(ValueError, "multi-label"):
            validate_model_config(config)
        config.problem_type = "multi_label_classification"
        config.model_type = "unknown-custom"
        with self.assertRaisesRegex(ValueError, "architecture"):
            validate_model_config(config)

    def test_actual_protocol_pairing_never_scores_test_and_preserves_inputs(self):
        original = self.reference.read_bytes()
        candidate = FixtureCandidate()
        result = run_comparison(self.root, self.reference, self.baseline, candidate=candidate)
        self.assertEqual(len(candidate.seen), 20)
        self.assertNotIn("NEVER SCORE THIS", candidate.seen)
        self.assertTrue(result["folds_verified_identical"])
        self.assertFalse(result["test_scored"])
        self.assertEqual(result["methods"]["fixed_0.5"]["micro_f1_difference"], 1.)
        self.assertEqual(result["methods"]["fixed_0.3"]["micro_f1_difference"], 0.)
        self.assertEqual(self.reference.read_bytes(), original)
        folder = self.root / result["run_id"]
        config = json.loads((folder / "config.json").read_text())
        self.assertEqual(config["status"], "complete")
        self.assertEqual(config["predictions_sha256"], file_digest(folder / "predictions.jsonl"))
        self.assertTrue((folder.parent / "comparison.json").is_file())
        candidate_report = json.loads((folder / "refinement/report.json").read_text())
        self.assertEqual(candidate_report["hashes"]["prediction_config_sha256"], file_digest(folder / "config.json"))
        self.assertEqual(list((self.root / "runs/workbench/refinements").glob("**/calibration.json")), [])

    def test_modified_or_unpaired_oof_data_is_rejected(self):
        candidate = FixtureCandidate()
        result = run_comparison(self.root, self.reference, self.baseline, candidate=candidate)
        folder = self.root / result["run_id"] / "refinement"
        path = folder / "oof_predictions.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["fold"] = (rows[0]["fold"] + 1) % 5
        path.write_text("\n".join(map(json.dumps, rows)))
        with self.assertRaisesRegex(ValueError, "recorded hash"):
            compare_oof(self.baseline, folder)
        report_path = folder / "report.json"
        report = json.loads(report_path.read_text())
        report["oof_predictions_sha256"] = file_digest(path)
        report_path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "paired posts, labels, and folds"):
            compare_oof(self.baseline, folder)


if __name__ == "__main__":
    unittest.main()
