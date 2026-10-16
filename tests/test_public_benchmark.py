"""Official labels stay human sourced, split isolated, and reproducibly pinned."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from emotion_schema import file_digest
from label_schema import LABELS, interpret, text_hash
from public_benchmark import prepare_goemotions, score_goemotions_development

REVISION = "a" * 40


class FixtureEngine:
    def __init__(self):
        self.seen = []

    def provenance(self, tasks):
        return {"models": {"emotion": {"id": "fixture", "revision": "pinned"}}}

    def infer(self, text, threshold, tasks):
        self.seen.append(text)
        scores = {label: .9 if label == "joy" else .01 for label in LABELS}
        return {"text": text, "text_sha256": text_hash(text), "scores": scores,
                "truncated": False, **interpret(scores, threshold)}


class PublicBenchmarkTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dev = "Development\t17\td1\nDuplicate\t17\td2\nDuplicate\t17\td3\nConflict\t17\td4\nConflict\t2\td5\nOverlap\t17\td6\nNeutral conflict\t17,27\td7\n123\t17\td8\n"
        self.test = "Overlap\t17\tt1\nReserved only\t2\tt2\n"

    def fetch(self, url):
        if "api.github.com" in url:
            return json.dumps([{"sha": REVISION}]).encode()
        filename = url.rsplit("/", 1)[-1]
        return {"dev.tsv": self.dev.encode(), "test.tsv": self.test.encode(),
                "emotions.txt": "\n".join(LABELS).encode(), "LICENSE": b"Apache License Version 2.0",
                "README.md": b"Official human annotations"}[filename]

    def test_preparation_provenance_duplicates_conflicts_and_test_reservation(self):
        with patch("public_benchmark._fetch", side_effect=self.fetch):
            manifest = prepare_goemotions(self.root)
        self.assertEqual(manifest["revision"], REVISION)
        self.assertEqual((manifest["development_rows"], manifest["test_rows"]), (8, 2))
        self.assertEqual(manifest["deduplication"]["development"]["conflicting_text_rows_retained"], 2)
        self.assertEqual(manifest["deduplication"]["development"]["consistent_duplicate_rows_retained"], 1)
        self.assertEqual(manifest["deduplication"]["development"]["neutral_emotion_overlap_rows_retained"], 1)
        self.assertEqual(manifest["deduplication"]["development_test_overlap_rows_retained_official_splits"], 1)
        reference = pd.read_csv(self.root / manifest["review_path"])
        self.assertTrue(reference.reference_origin.eq("published_human_annotations").all())
        self.assertTrue(reference.post_id.str.startswith("goemotions:").all())
        self.assertEqual(Path(manifest["review_path"]).name, "reference.csv")
        with patch("public_benchmark._fetch", side_effect=AssertionError("cache must avoid network")):
            self.assertEqual(prepare_goemotions(self.root, REVISION), manifest)

    def test_actual_scoring_uses_only_development_and_report_config_hash_stays_valid(self):
        engine = FixtureEngine()
        with patch("public_benchmark._fetch", side_effect=self.fetch):
            result = score_goemotions_development(self.root, revision=REVISION, engine=engine)
        self.assertEqual(engine.seen, ["Development", "Duplicate", "Duplicate", "Conflict", "Conflict", "Overlap", "Neutral conflict", "123"])
        self.assertFalse(result["report"]["test_scored"])
        folder = self.root / result["run_id"]
        self.assertEqual(result["report"]["hashes"]["prediction_config_sha256"], file_digest(folder / "config.json"))
        self.assertEqual(json.loads((folder / "config.json").read_text())["status"], "complete")
        self.assertTrue((folder / "benchmark.json").is_file())

    def test_cached_source_tampering_is_rejected(self):
        with patch("public_benchmark._fetch", side_effect=self.fetch):
            manifest = prepare_goemotions(self.root, REVISION)
        (self.root / manifest["id"] / "dev.tsv").write_text("tampered")
        with self.assertRaisesRegex(ValueError, "recorded hash"):
            prepare_goemotions(self.root, REVISION)

    def test_reuse_is_bound_to_pinned_model_and_original_predictions(self):
        with patch("public_benchmark._fetch", side_effect=self.fetch):
            first = score_goemotions_development(self.root, revision=REVISION, engine=FixtureEngine())
        cached = self.root / first["run_id"] / "predictions.jsonl"
        engine = FixtureEngine()
        second = score_goemotions_development(self.root, revision=REVISION, engine=engine, reuse_predictions=cached)
        self.assertEqual(engine.seen, [])
        config = json.loads((self.root / second["run_id"] / "config.json").read_text())
        self.assertEqual((config["reused_rows"], config["newly_scored_rows"]), (8, 0))
        self.assertNotEqual(first["run_id"], second["run_id"])
        cached.write_text(cached.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "recorded hash"):
            score_goemotions_development(self.root, revision=REVISION, engine=FixtureEngine(), reuse_predictions=cached)


if __name__ == "__main__":
    unittest.main()
