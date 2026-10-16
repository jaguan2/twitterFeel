"""User imports remain immutable, explicit, and separate from model findings."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from research_workspace import DatasetLibrary, FindingsStore, build_findings


class WorkspaceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "app_version.json").write_text(json.dumps({"version": "test-version"}))
        self.library = DatasetLibrary(self.root)

    def test_preview_keeps_identifiers_and_requires_explicit_mapping(self):
        content = 'post_id,body,author,emotion,scores\n00123,"Hello, world!",0007,anger,0.9\n'
        preview = self.library.preview("Survey", content)
        self.assertEqual(preview["preview"][0]["post_id"], "00123")
        self.assertTrue(preview["requires_text_mapping"])
        self.assertEqual(self.library.list_datasets(), [])
        with self.assertRaisesRegex(ValueError, "Choose the column"):
            self.library.import_dataset("Survey", content)
        manifest = self.library.import_dataset("Survey", content, text_column="body", user_column="author")
        frame = self.library.load_dataset(manifest["id"])
        self.assertEqual(frame.post_id.tolist(), ["00123"])
        self.assertEqual(frame.user_id.tolist(), ["0007"])
        self.assertEqual(frame.text.tolist(), ["Hello, world!"])
        self.assertNotIn("emotion", frame)
        self.assertNotIn("scores", frame)
        self.assertEqual(manifest["app_version"], "test-version")

    def test_repeat_import_increments_immutable_version_and_retains_exact_bytes(self):
        content = b'\xef\xbb\xbftext,post_id\r\n"caf\xc3\xa9!",0001\r\n'
        first = self.library.import_dataset("  My Study  ", content, text_column="text")
        first_path = self.root / first["id"] / "dataset.csv"
        original = first_path.read_bytes()
        second = DatasetLibrary(self.root).import_dataset("my   study", content, text_column="text")
        self.assertEqual((first["version"], second["version"]), (1, 2))
        self.assertNotEqual(first["id"], second["id"])
        self.assertEqual(first_path.read_bytes(), original)
        self.assertEqual((self.root / first["id"] / "source.csv").read_bytes(), content)
        self.assertEqual(first["content_sha256"], hashlib.sha256(content).hexdigest())
        self.assertEqual(first["dataset_sha256"], hashlib.sha256(original).hexdigest())
        self.assertEqual(len(self.library.list_datasets()), 2)

    def test_formats_blank_rows_duplicate_source_ids_and_source_mapping(self):
        tsv = self.library.import_dataset("TSV", "post_id\tbody\n0001\thello\n0001\t\n", "tsv", "body")
        frame = self.library.load_dataset(tsv["id"])
        self.assertTrue(frame.post_id.is_unique)
        self.assertEqual(frame.source_post_id.tolist(), ["0001", "0001"])
        self.assertEqual(frame.text.tolist(), ["hello", ""])
        txt = self.library.import_dataset("Paste", "First line\n\nThird line", "txt")
        self.assertEqual(self.library.load_dataset(txt["id"]).text.tolist(), ["First line", "", "Third line"])
        jsonl = self.library.import_dataset("JSON", '{"post_id":"0003","message":"Hi"}\n{"post_id":9007199254740993,"message":"Again"}', "jsonl", "message")
        self.assertEqual(self.library.load_dataset(jsonl["id"]).source_post_id.tolist(), ["0003", "9007199254740993"])
        numeric_text = self.library.import_dataset("Codes", "message\n000123\n000456", text_column="message")
        self.assertEqual(self.library.load_dataset(numeric_text["id"]).text.tolist(), ["000123", "000456"])

    def test_limits_malformed_data_arbitrary_paths_and_hash_tampering(self):
        for data in ('text,text\na,b', 'text\na,b', 'text\n"unfinished', '\x00binary', ''):
            with self.subTest(data=data), self.assertRaises(ValueError):
                self.library.preview("Bad", data)
        with patch("research_workspace.MAX_BYTES", 3), self.assertRaisesRegex(ValueError, "20 MB"):
            self.library.preview("Big", "four")
        with patch("research_workspace.MAX_ROWS", 1), self.assertRaisesRegex(ValueError, "100,000"):
            self.library.preview("Rows", "text\na\nb")
        with self.assertRaises(ValueError):
            self.library.load_dataset("../../outside")
        item = self.library.import_dataset("Integrity", "text\noriginal", text_column="text")
        (self.root / item["id"] / "dataset.csv").write_text("text\nchanged")
        with self.assertRaisesRegex(ValueError, "changed"):
            self.library.load_dataset(item["id"])

    def analysis_run(self):
        folder = self.root / "runs/analyses/example"
        folder.mkdir(parents=True)
        (folder / "config.json").write_text(json.dumps({"task": "emotion-and-sentiment", "status": "complete", "dataset_id": "test", "app_version": "test-version"}))
        records = [
            {"post_id": "a", "text": "Mixed", "labels": ["joy", "fear"], "status": "labeled", "sentiment": {"label": "negative"}},
            {"post_id": "b", "text": "Joy", "labels": ["joy"], "status": "labeled", "sentiment": {"label": "positive"}},
            {"post_id": "c", "text": "", "labels": [], "status": "invalid", "sentiment": None},
        ]
        (folder / "predictions.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
        (folder / "summary.json").write_text(json.dumps({"rows": 3}))
        return folder

    def test_authored_observations_are_append_only_and_validated(self):
        self.analysis_run()
        store = FindingsStore(self.root)
        first = store.save("runs/analyses/example", "Alice", "Mixed language", "Inspect mixed expressions.")
        second = store.save("runs/analyses/example", "Alice", "Follow up", "Review again.")
        self.assertNotEqual(first["id"], second["id"])
        self.assertEqual(len(store.list("runs/analyses/example")), 2)
        self.assertTrue(all(row["kind"] == "authored" for row in store.list()))
        self.assertEqual(first["app_version"], "test-version")
        for run in ("../outside", str(self.root / "runs/analyses/example"), "runs/analyses/missing"):
            with self.subTest(run=run), self.assertRaises(ValueError):
                store.save(run, "Alice", "Title", "Note")
        with self.assertRaises(ValueError):
            store.save("runs/analyses/example", "", "Title", "Note")

    def test_automatic_findings_use_observed_counts_not_accuracy_claims(self):
        folder = self.analysis_run()
        one, two = build_findings(folder), build_findings(folder)
        self.assertEqual(one, two)
        self.assertEqual(one["kind"], "automatic")
        by_title = {row["title"]: row for row in one["observations"]}
        self.assertEqual(by_title["Most frequently selected emotions"]["evidence"]["top_labels"], {"joy": 2, "fear": 1})
        self.assertEqual(by_title["Multiple emotions"]["evidence"]["posts"], 1)
        self.assertEqual(by_title["Sentiment polarity"]["evidence"]["sentiment_counts"], {"negative": 1, "positive": 1})
        self.assertIn("separately", by_title["Sentiment polarity"]["text"])
        self.assertTrue(any("do not measure accuracy" in s for s in one["limitations"]))
        self.assertIn("overlap", by_title["Most frequently selected emotions"]["text"])

    def test_no_sentiment_is_reported_as_unavailable_not_inferred(self):
        folder = self.analysis_run()
        (folder / "predictions.jsonl").write_text(json.dumps({"post_id": "a", "labels": ["fear"], "status": "labeled"}) + "\n")
        result = build_findings(folder)
        self.assertTrue(any(r["title"] == "Sentiment was not available" for r in result["observations"]))
        self.assertTrue(any("row count differs" in text for text in result["limitations"]))


if __name__ == "__main__":
    unittest.main()
