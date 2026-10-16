import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from semeval_data import DEV_MEMBERS, NATIVE_LABELS, SHARED_LABELS, parse_development, prepare_development


def sample(rows):
    lines = ["\t".join(["ID", "Tweet", *NATIVE_LABELS])]
    for identifier, text, labels in rows:
        lines.append("\t".join([identifier, text, *[str(int(label in labels)) for label in NATIVE_LABELS]]))
    return ("\n".join(lines) + "\n").encode("utf-8")


class SemEvalDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def archive(self, data, extra=None):
        path = self.root / "source.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(DEV_MEMBERS[0], data)
            for name, content in (extra or {}).items():
                archive.writestr(name, content)
        return path

    def test_preserves_native_and_overlapping_shared_labels(self):
        text = "I’m hopeful but afraid. #newstart"
        records = parse_development(sample([("dev1", text, ["fear", "optimism", "trust"])]))
        self.assertEqual(records[0]["text"], text)
        self.assertEqual(records[0]["native_labels"], ["fear", "optimism", "trust"])
        self.assertEqual(records[0]["shared_labels"], ["fear", "optimism"])
        self.assertEqual(records[0]["unrepresented_labels"], ["trust"])
        self.assertEqual(records[0]["text_sha256"], hashlib.sha256(text.encode()).hexdigest())
        self.assertEqual(len(records[0]["native_binary_labels"]), 11)

    def test_unsupported_emotion_never_becomes_neutral(self):
        rows = parse_development(sample([("1", "Waiting for tomorrow", ["anticipation"]), ("2", "The post arrived", [])]))
        self.assertEqual(rows[0]["shared_labels"], [])
        self.assertFalse(rows[0]["native_all_negative"])
        self.assertTrue(rows[0]["shared_all_negative"])
        self.assertTrue(rows[1]["native_all_negative"])
        self.assertNotIn("neutral", rows[0]["native_labels"])
        self.assertNotIn("human_labels", rows[0])

    def test_only_known_development_and_terms_members_are_opened(self):
        data = sample([("1", "Happy and afraid", ["joy", "fear"])])
        archive = self.archive(data, {"SemEval2018-Task1-all-data/English/E-c/2018-E-c-En-test-gold.txt": "DO NOT OPEN", "../../escape.txt": "malicious", "README.txt": "Research terms"})
        opened, original = [], zipfile.ZipFile.read
        def read(instance, name, *args, **kwargs):
            opened.append(name.filename if isinstance(name, zipfile.ZipInfo) else name)
            return original(instance, name, *args, **kwargs)
        with patch.object(zipfile.ZipFile, "read", read):
            manifest = prepare_development(archive, self.root / "v1")
        self.assertEqual(set(opened), {DEV_MEMBERS[0], "README.txt"})
        self.assertFalse(manifest["test_labels_read"])
        self.assertFalse(manifest["model_scored"])
        self.assertEqual(manifest["shared_mapping"], {label: label for label in SHARED_LABELS})
        self.assertEqual((self.root / "v1/source.zip").read_bytes(), archive.read_bytes())
        canonical = (self.root / "v1/reference.jsonl").read_bytes()
        self.assertEqual(manifest["canonical_sha256"], hashlib.sha256(canonical).hexdigest())
        self.assertEqual(json.loads(canonical)["split"], "development")
        self.assertFalse((self.root / "escape.txt").exists())

    def test_wrong_hash_and_existing_output_do_not_mutate_reference(self):
        archive = self.archive(sample([("1", "Good", ["joy"])]))
        out = self.root / "v1"
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            prepare_development(archive, out, expected_sha256="0" * 64)
        self.assertFalse(out.exists())
        prepare_development(archive, out)
        previous = (out / "manifest.json").read_bytes()
        with self.assertRaises(FileExistsError):
            prepare_development(archive, out)
        self.assertEqual((out / "manifest.json").read_bytes(), previous)

    def test_rejects_missing_duplicate_or_invalid_labels(self):
        for data in [b"ID\tTweet\n1\tHi\n", sample([("1", "A", []), ("1", "B", [])]), sample([("1", "A", [])]).replace(b"\t0", b"\t-1", 1)]:
            with self.assertRaises(ValueError):
                parse_development(data)
        archive = self.archive(sample([("1", "Hi", [])]), {DEV_MEMBERS[1]: sample([("2", "Other", [])])})
        with self.assertRaisesRegex(ValueError, "exactly one"):
            prepare_development(archive, self.root / "v1")


if __name__ == "__main__":
    unittest.main()
