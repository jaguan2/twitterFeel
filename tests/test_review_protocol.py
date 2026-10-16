"""Calibration stays blind and development-only; user recovery needs provenance."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from emotion_schema import file_digest
from label_posts import read_posts
from label_schema import text_hash
from review_protocol import enrich_users, prepare_calibration


class ReviewProtocolTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "src").mkdir()
        self.rubric = self.root / "src/annotation_rubric.json"
        self.rubric.write_text(json.dumps({"version": "test-v1", "labels": ["joy", "fear", "neutral"]}))
        folder = self.root / "runs/review"
        folder.mkdir(parents=True)
        self.review_path = folder / "review.csv"

    def review(self, development=40, test=8):
        records = [{"post_id": f"p{i}", "source_row": i, "text": f"Synthetic post {i}",
                    "text_sha256": text_hash(f"Synthetic post {i}"),
                    "split": "development" if i < development else "test",
                    "user_id": f"u{i//2}", "human_labels": "anger", "review_status": "reviewed",
                    "notes": "must not be copied", "scores": "must stay hidden", "emotion": "anger"}
                   for i in range(development+test)]
        pd.DataFrame(records).to_csv(self.review_path, index=False)
        return read_posts(self.review_path)

    def legacy_source(self):
        source = self.root / "data/raw/source.xls"
        source.parent.mkdir(parents=True)
        pd.DataFrame([
            {"post_id": "0001", "post_text": "A first post", "user_id": "0007"},
            {"post_id": "0001", "post_text": "A second post", "user_id": "0008"},
            {"post_id": "0003", "post_text": "A third post", "user_id": ""},
        ]).to_csv(source, index=False)
        data = read_posts(source)
        data["split"] = "development"
        data.drop(columns="user_id").to_csv(self.review_path, index=False)
        (self.review_path.parent / "sample.json").write_text(json.dumps({
            "source_sha256": file_digest(source), "text_column": "post_text"}))
        return source, read_posts(self.review_path)

    def test_calibration_is_30_unique_blind_development_posts_and_reusable(self):
        original = self.review()
        # Include duplicate text under another identity and blank text.
        extra = original.iloc[[0, 1]].copy()
        extra["post_id"] = ["duplicate", "blank"]
        extra.loc[extra.post_id == "blank", ["text", "text_sha256"]] = ["", text_hash("")]
        pd.concat([original, extra]).to_csv(self.review_path, index=False)
        before = self.review_path.read_bytes()
        first = prepare_calibration(self.root, self.review_path)
        repeated = prepare_calibration(self.root, self.review_path)
        self.assertFalse(first["existing"])
        self.assertTrue(repeated["existing"])
        self.assertEqual(first["review_id"], repeated["review_id"])
        path = self.root / first["review_id"]
        practice = read_posts(path)
        self.assertEqual(len(practice), 30)
        self.assertTrue(practice.text_sha256.is_unique)
        self.assertTrue(practice.split.eq("development").all())
        self.assertTrue(practice.review_status.eq("pending").all())
        self.assertTrue(practice.human_labels.eq("").all())
        self.assertTrue(practice.notes.eq("").all())
        self.assertNotIn("scores", practice)
        self.assertNotIn("emotion", practice)
        self.assertEqual(before, self.review_path.read_bytes())
        metadata = json.loads((path.parent / "metadata.json").read_text())
        self.assertEqual(metadata["test_rows_selected"], 0)
        self.assertEqual(metadata["source_sha256"], file_digest(self.review_path))
        self.assertEqual(metadata["rubric_sha256"], file_digest(self.rubric))
        self.assertEqual(metadata["rubric_version"], "test-v1")
        nested = prepare_calibration(self.root, path)
        self.assertEqual(nested["review_id"], first["review_id"])
        self.assertTrue(nested["existing"])

    def test_small_sample_warns_and_rubric_change_creates_separate_practice(self):
        self.review(development=3, test=2)
        first = prepare_calibration(self.root, self.review_path)
        self.assertEqual(first["rows"], 3)
        self.assertIn("target is 30", first["warning"])
        original = (self.root / first["review_id"]).read_bytes()
        self.rubric.write_text(json.dumps({"version": "test-v2"}))
        second = prepare_calibration(self.root, self.review_path)
        self.assertNotEqual(first["review_id"], second["review_id"])
        self.assertEqual((self.root / first["review_id"]).read_bytes(), original)

    def test_test_only_samples_cannot_become_calibration(self):
        self.review(development=0, test=3)
        with self.assertRaisesRegex(ValueError, "No unique nonblank development"):
            prepare_calibration(self.root, self.review_path)
        with self.assertRaises(ValueError):
            prepare_calibration(self.root, self.root / "outside/review.csv")

    def test_user_recovery_requires_exact_all_row_alignment_and_preserves_blanks(self):
        _, frame = self.legacy_source()
        before = frame.copy(deep=True)
        enriched = enrich_users(frame, self.review_path, self.root)
        self.assertNotIn("user_id", frame)
        self.assertEqual(frame.to_dict("records"), before.to_dict("records"))
        self.assertEqual(enriched.user_id.tolist(), ["0007", "0008", ""])
        self.assertEqual(enriched.attrs["user_enrichment"]["status"], "partial")
        self.assertEqual(enriched.attrs["user_enrichment"]["recovered_rows"], 2)
        self.assertTrue(enriched.attrs["user_enrichment"]["warnings"])
        changed = frame.copy()
        changed.loc[0, "source_row"] = 999
        fallback = enrich_users(changed, self.review_path, self.root)
        self.assertNotIn("user_id", fallback)
        self.assertEqual(fallback.attrs["user_enrichment"]["status"], "unavailable")

    def test_source_hash_mismatch_never_guesses_users(self):
        source, frame = self.legacy_source()
        source.write_text(source.read_text() + "\n")
        result = enrich_users(frame, self.review_path, self.root)
        self.assertNotIn("user_id", result)
        self.assertEqual(result.attrs["user_enrichment"]["recovered_rows"], 0)
        self.assertIn("fingerprint", result.attrs["user_enrichment"]["warnings"][1])

    def test_existing_users_are_untouched_and_calibration_tampering_detected(self):
        frame = self.review(development=4, test=2)
        result = enrich_users(frame, self.review_path, self.root)
        self.assertEqual(frame.user_id.tolist(), result.user_id.tolist())
        self.assertEqual(result.attrs["user_enrichment"]["status"], "already_present")
        calibration = prepare_calibration(self.root, self.review_path)
        path = self.root / calibration["review_id"]
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "changed"):
            prepare_calibration(self.root, self.review_path)


if __name__ == "__main__":
    unittest.main()
