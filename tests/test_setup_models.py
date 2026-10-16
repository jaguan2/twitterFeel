"""Pinned cache reuse and restricted checkpoint conversion, without networking."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from label_schema import LABELS
from setup_models import PINS, TOKEN_FILES, setup_model


class SetupModelsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def cache(self, task):
        spec = PINS[task]
        folder = self.root / "runs/model_cache" / spec["directory"]
        metadata = folder / ".cache/huggingface/download"
        metadata.mkdir(parents=True)
        config = {"id2label": dict(enumerate(LABELS)), "problem_type": "multi_label_classification"} if task == "emotion" else {"id2label": {0: "negative", 1: "neutral", 2: "positive"}}
        for name in (*TOKEN_FILES, spec["weight"]):
            raw = json.dumps(config).encode() if name == "config.json" else b"synthetic cached file"
            (folder / name).write_bytes(raw)
            (metadata / (name + ".metadata")).write_text(spec["revision"] + "\n" + hashlib.sha256(raw).hexdigest() + "\n0")
        return folder

    def test_complete_pinned_cache_never_calls_downloader(self):
        self.cache("emotion")
        download = Mock(side_effect=AssertionError("Network must not be used"))
        result = setup_model(self.root, "emotion", download)
        self.assertTrue(result["reused_download"])
        download.assert_not_called()
        manifest = json.loads(Path(result["manifest"]).read_text())
        self.assertEqual(manifest["revision"], PINS["emotion"]["revision"])
        self.assertIn("model.safetensors", manifest["files_sha256"])

    def test_mismatched_revision_or_corrupted_file_is_not_silently_reused(self):
        folder = self.cache("emotion")
        (folder / "model.safetensors").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            setup_model(self.root, "emotion", Mock())
        metadata = folder / ".cache/huggingface/download/config.json.metadata"
        metadata.write_text("another-revision\nignored\n0")
        with self.assertRaisesRegex(ValueError, "another model revision"):
            setup_model(self.root, "emotion", Mock())

    def test_sentiment_conversion_uses_weights_only_and_records_source(self):
        import torch
        from safetensors.torch import load_file
        folder = self.cache("sentiment")
        with patch("torch.load", return_value={"weight": torch.tensor([1., 2.])}) as loader:
            result = setup_model(self.root, "sentiment", Mock())
        loader.assert_called_once_with(folder / "pytorch_model.bin", map_location="cpu", weights_only=True)
        self.assertEqual(load_file(str(folder / "model.safetensors"))["weight"].tolist(), [1., 2.])
        manifest = json.loads(Path(result["manifest"]).read_text())
        self.assertEqual(manifest["conversion"]["source_sha256"], hashlib.sha256((folder / "pytorch_model.bin").read_bytes()).hexdigest())
        self.assertTrue(manifest["conversion"]["performed_this_setup"])
        (folder / "model.safetensors").write_bytes(b"changed derived weights")
        with self.assertRaisesRegex(ValueError, "Runtime weights changed"):
            setup_model(self.root, "sentiment", Mock())


if __name__ == "__main__":
    unittest.main()
