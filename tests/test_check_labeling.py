"""Parity checks reject mismatched inputs and changed model output."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from check_labeling import CASES, compare_scores


class DiagnosticTest(unittest.TestCase):
    def test_same_labels_do_not_hide_different_scores(self):
        cli = {"text": "Synthetic post", "scores": {"joy": .8, "fear": .7}}
        app = {"text": "Synthetic post", "scores": {"joy": .8, "fear": .6}}
        result = compare_scores(cli, app)
        self.assertFalse(result["scores_match"])
        self.assertAlmostEqual(result["max_absolute_score_difference"], .1)

    def test_numerical_tolerance_is_explicit(self):
        cli = {"text": "Synthetic post", "scores": {"joy": .8}}
        app = {"text": "Synthetic post", "scores": {"joy": .800001}}
        self.assertTrue(compare_scores(cli, app)["scores_match"])
        self.assertFalse(compare_scores(cli, app, tolerance=1e-8)["scores_match"])

    def test_parity_requires_same_text_and_score_vocabulary(self):
        cli = {"text": "Original", "scores": {"joy": .8}}
        with self.assertRaisesRegex(ValueError, "different text"):
            compare_scores(cli, {"text": "Edited", "scores": {"joy": .8}})
        with self.assertRaisesRegex(ValueError, "vocabularies"):
            compare_scores(cli, {"text": "Original", "scores": {"fear": .8}})

    def test_diagnostic_corpus_keeps_invalid_and_unsupported_cases(self):
        cases = dict(CASES)
        self.assertEqual(cases["blank"], "")
        self.assertIn("non_english", cases)
        self.assertIn("sarcasm", cases)
        self.assertIn("quotation", cases)
        self.assertIn("negation", cases)
        self.assertGreater(len(cases["truncation"].split()), 256)
        self.assertEqual(len(cases), len(CASES))


if __name__ == "__main__":
    unittest.main()
