"""Meaningful post-labeling checks without downloading a model."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import pandas as pd

from twitterfeel.labeling.label_schema import LABELS, interpret, text_hash, validate_thresholds
from twitterfeel.labeling.label_posts import evaluate, new_directory, read_posts, sample


class InterpretationTest(unittest.TestCase):
    def scores(self, **selected):
        return {label: selected.get(label, .01) for label in LABELS}

    def test_mixed_emotions_survive_without_forced_winner(self):
        result = interpret(self.scores(excitement=.8, fear=.7, love=.9))
        self.assertEqual(set(result["labels"]), {"excitement", "fear", "love"})
        self.assertEqual(result["status"], "labeled")

    def test_neutral_is_not_a_fallback_for_low_scores(self):
        self.assertEqual(interpret(self.scores())["status"], "uncertain")
        self.assertEqual(interpret(self.scores())["labels"], [])
        self.assertEqual(interpret(self.scores(neutral=.8))["status"], "neutral")

    def test_neutral_conflict_retains_candidates_for_review(self):
        result = interpret(self.scores(neutral=.8, anger=.7))
        self.assertEqual(result["status"], "review_needed")
        self.assertEqual(set(result["labels"]), {"neutral", "anger"})
        self.assertIn("neutral_emotion_conflict", result["review_reason"])

    def test_invalid_scores_cannot_look_like_uncertainty(self):
        for bad in (float("nan"), float("inf"), -.1, 1.1):
            with self.subTest(score=bad), self.assertRaises(ValueError):
                interpret(self.scores(anger=bad))
        with self.assertRaises(ValueError):
            interpret({"anger": .5})

    def test_per_emotion_thresholds_preserve_independent_labels_and_conflicts(self):
        thresholds = dict.fromkeys(LABELS, .5)
        thresholds.update(excitement=.2, fear=.4, neutral=.7)
        result = interpret(self.scores(excitement=.25, fear=.45, neutral=.6), thresholds=thresholds)
        self.assertEqual(result["labels"], ["excitement", "fear"])
        self.assertEqual(result["status"], "labeled")
        result = interpret(self.scores(excitement=.25, neutral=.8), thresholds=thresholds)
        self.assertEqual(result["status"], "review_needed")
        self.assertIn("neutral_emotion_conflict", result["review_reason"])
        for invalid in ({"joy": .3}, {**thresholds, "joy": float("nan")},
                        {**thresholds, "joy": True}, {**thresholds, "joy": "0.3"}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_thresholds(invalid)


class WorkflowTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def csv(self, name, rows):
        path = self.root / name
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def test_numeric_looking_text_keeps_its_original_characters(self):
        for column in ("text", "post_text", "custom_body"):
            path = self.csv(column + ".csv", [{"post_id": "001", column: "000123"}])
            frame = read_posts(path, text_column=column)
            self.assertEqual(frame.iloc[0].text, "000123")
            self.assertEqual(frame.iloc[0].text_sha256, text_hash("000123"))

    def quiet(self, func, args):
        with contextlib.redirect_stdout(io.StringIO()):
            return func(args)

    def test_hash_detects_edited_text_and_duplicate_ids(self):
        blank = self.csv("blank_id.csv", [{"post_id": " ", "text": "valid text"}])
        with self.assertRaisesRegex(ValueError, "blank"):
            read_posts(blank)
        path = self.csv("posts.csv", [{"post_id": "a", "cleaned_text": "edited",
                                       "text_sha256": text_hash("original")}])
        with self.assertRaisesRegex(ValueError, "hash"):
            read_posts(path)
        path = self.csv("duplicates.csv", [
            {"post_id": "a", "cleaned_text": "first"},
            {"post_id": "a", "cleaned_text": "different"}])
        with self.assertRaises(ValueError):
            read_posts(path)

    def test_review_roundtrip_preserves_original_source_id(self):
        path = self.csv("source_id.csv", [{"post_id": "row_1", "source_post_id": "00123456789012345678",
                                           "text": "A post", "text_sha256": text_hash("A post")}])
        result = read_posts(path)
        self.assertEqual(result.iloc[0].source_post_id, "00123456789012345678")

    def test_raw_text_is_preferred_and_source_identifiers_stay_exact(self):
        path = self.csv("raw.csv", [
            {"post_id": "00123", "user_id": "0007", "post_text": "I'm EXCITED! :)",
             "cleaned_text": "excit"},
            {"post_id": "00123", "user_id": "0007", "post_text": "I'm EXCITED! :)",
             "cleaned_text": "excit"}])
        posts = read_posts(path)
        self.assertEqual(posts.text.tolist(), ["I'm EXCITED! :)"] * 2)
        self.assertEqual(posts.source_post_id.tolist(), ["00123"] * 2)
        self.assertEqual(posts.user_id.tolist(), ["0007"] * 2)
        self.assertEqual(posts.post_id.nunique(), 2)
        # A canonical review export must retain the same IDs and text hashes.
        review = self.root / "canonical.csv"
        posts.to_csv(review, index=False)
        reopened = read_posts(review)
        self.assertEqual(posts.post_id.tolist(), reopened.post_id.tolist())
        self.assertEqual(posts.text_sha256.tolist(), reopened.text_sha256.tolist())

    def test_explicit_text_override_is_hashed_instead_of_default(self):
        path = self.csv("fields.csv", [{"post_text": "Original!", "cleaned_text": "clean"}])
        posts = read_posts(path, "cleaned_text")
        self.assertEqual(posts.text.iloc[0], "clean")
        self.assertEqual(posts.text_sha256.iloc[0], text_hash("clean"))

    def test_blind_sample_is_disjoint_by_user_and_exact_text(self):
        rows = [{"post_id": f"p{i}", "user_id": f"u{i//3}",
                 "cleaned_text": f"unique text {i}", "emotion": "hidden",
                 "emotion_prob_joy": .9} for i in range(24)]
        rows.append({"post_id": "duplicate", "user_id": "another-user",
                     "cleaned_text": "unique text 0", "emotion": "hidden"})
        path = self.csv("posts.csv", rows)
        out = self.root / "review"
        args = SimpleNamespace(csv=path, out=out, size=24, seed=42, text_column=None)
        self.quiet(sample, args)
        review = pd.read_csv(out / "review.csv", keep_default_na=False)
        self.assertFalse(review.text_sha256.duplicated().any())
        self.assertTrue(review.human_labels.eq("").all())
        self.assertNotIn("emotion", review)
        self.assertNotIn("emotion_prob_joy", review)
        original_users = pd.DataFrame(rows).set_index("post_id").user_id
        self.assertTrue(review.user_id.eq(review.post_id.map(original_users)).all())
        dev = review[review.split == "development"]
        test = review[review.split == "test"]
        self.assertTrue(len(dev) and len(test))
        self.assertFalse(set(dev.user_id) & set(test.user_id))
        self.assertFalse(set(dev.text_sha256) & set(test.text_sha256))
        with self.assertRaises(ValueError):
            self.quiet(sample, args)

    def evaluation_fixture(self):
        review = self.csv("review.csv", [
            {"post_id": "a", "cleaned_text": "first", "split": "development",
             "human_labels": "joy;fear", "review_status": "reviewed"},
            {"post_id": "b", "cleaned_text": "second", "split": "development",
             "human_labels": "joy", "review_status": "reviewed"},
            {"post_id": "c", "cleaned_text": "third", "split": "development",
             "human_labels": "", "review_status": "unclear"},
        ])
        predictions = self.root / "predictions.jsonl"
        records = [
            {"post_id": "a", "text_sha256": text_hash("first"),
             "labels": ["joy", "fear"], "status": "labeled"},
            {"post_id": "b", "text_sha256": text_hash("second"),
             "labels": [], "status": "uncertain"},
        ]
        predictions.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
        return SimpleNamespace(review=review, predictions=predictions,
                               split="development", out=self.root / "evaluation.json")

    def test_uncertain_predictions_count_as_misses_not_removed_rows(self):
        args = self.evaluation_fixture()
        self.quiet(evaluate, args)
        result = json.loads(args.out.read_text(encoding="utf-8"))
        self.assertEqual(result["reviewed_rows"], 2)
        self.assertEqual(result["excluded_review_status_counts"], {"unclear": 1})
        self.assertEqual(result["per_label"]["joy"]["support"], 2)
        self.assertEqual(result["per_label"]["joy"]["recall"], .5)
        self.assertAlmostEqual(result["micro_f1"], .8)
        self.assertEqual(result["exact_match"], .5)
        self.assertEqual(result["automatic_acceptance_fraction"], .5)
        before = args.out.read_bytes()
        with self.assertRaises(ValueError):
            self.quiet(evaluate, args)
        self.assertEqual(before, args.out.read_bytes())

    def test_evaluation_refuses_prediction_text_mismatch(self):
        args = self.evaluation_fixture()
        records = [json.loads(line) for line in args.predictions.read_text().splitlines()]
        records[0]["text_sha256"] = text_hash("wrong text")
        args.predictions.write_text("".join(json.dumps(r) + "\n" for r in records))
        with self.assertRaisesRegex(ValueError, "mismatched"):
            self.quiet(evaluate, args)

    def test_output_directory_is_never_reused(self):
        path = self.root / "existing"
        path.mkdir()
        sentinel = path / "keep.txt"
        sentinel.write_text("keep")
        with self.assertRaises(ValueError):
            new_directory(path)
        self.assertEqual(sentinel.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
