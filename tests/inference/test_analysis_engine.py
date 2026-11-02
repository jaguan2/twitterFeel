"""Independent sentiment scoring, preprocessing, and auditable local inference."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from twitterfeel.inference.engine import AnalysisEngine, EMOTION_MODELS, SENTIMENT_LABELS, sentiment_text
from twitterfeel.labeling.label_schema import LABELS, text_hash


class TinyTokenizer:
    def __init__(self):
        self.seen = []

    def __call__(self, text, **kwargs):
        self.seen.append(text)
        return {"input_ids": list(range(len(text.split()) + 2))}


class TinyModel:
    def __init__(self, values):
        self.values = values

    def __call__(self, **kwargs):
        import torch
        return SimpleNamespace(logits=torch.tensor([self.values]))


class AnalysisEngineTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.engine = AnalysisEngine(self.root)
        self.emotion_tokenizer, self.sentiment_tokenizer = TinyTokenizer(), TinyTokenizer()
        # Deliberately conflicting classifiers prove sentiment isn't inferred
        # from the emotion label or its threshold.
        values = [4.0 if label == "joy" else -8.0 for label in LABELS]
        self.engine._models = {
            "emotion": (self.emotion_tokenizer, TinyModel(values), dict(enumerate(LABELS))),
            "sentiment": (self.sentiment_tokenizer, TinyModel([4., 1., -2.]), dict(enumerate(SENTIMENT_LABELS))),
        }

    def test_sentiment_is_independent_softmax_and_original_text_is_preserved(self):
        original = "@Someone I love it! https://example.com/test"
        result = self.engine.infer(original)
        self.assertEqual(result["labels"], ["joy"])
        self.assertEqual(result["sentiment"]["label"], "negative")
        self.assertAlmostEqual(sum(result["sentiment"]["scores"].values()), 1, places=6)
        self.assertEqual(result["text"], original)
        self.assertEqual(result["text_sha256"], text_hash(original))
        self.assertEqual(result["sentiment"]["scored_text"], "@user I love it! http")
        self.assertEqual(self.emotion_tokenizer.seen[0], original)
        self.assertEqual(self.sentiment_tokenizer.seen[0], "@user I love it! http")
        result_high = self.engine.infer(original, threshold=.9999)
        self.assertEqual(result_high["labels"], [])
        self.assertEqual(result_high["sentiment"], result["sentiment"])
        json.dumps(result, allow_nan=False)

    def test_blank_and_bad_arguments_never_load_models(self):
        with patch.object(self.engine, "_load", side_effect=AssertionError("must not load")):
            self.assertEqual(self.engine.infer("  ")["status"], "invalid")
            for threshold in (float("nan"), float("inf"), 0, 1, True, None, "bad"):
                with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                    self.engine.infer("hello", threshold)
            for tasks in ([], ["sentiment"], ["emotion", "unknown"], ["emotion", "emotion"], "emotion"):
                with self.subTest(tasks=tasks), self.assertRaises(ValueError):
                    self.engine.infer("hello", tasks=tasks)

    def test_omitted_sentiment_is_not_scored_and_truncation_is_visible(self):
        text = " ".join(["hello"] * 300)
        result = self.engine.infer(text, tasks=["emotion"])
        self.assertTrue(result["truncated"])
        self.assertEqual(result["status"], "review_needed")
        self.assertIn("text_truncated", result["review_reason"])
        self.assertIsNone(result["sentiment"])
        self.assertEqual(self.sentiment_tokenizer.seen, [])
        result = self.engine.infer(text)
        self.assertTrue(result["sentiment"]["truncated"])

    def test_placeholder_preprocessing_follows_model_card(self):
        self.assertEqual(sentiment_text("Hi  @alice https://a.test @"), "Hi  @user http @")

    def test_model_availability_and_provenance_detect_weight_changes(self):
        cache = self.root / "runs/model_cache/twitter-sentiment"
        cache.mkdir(parents=True)
        (cache / "model.safetensors").write_bytes(b"first-weights")
        (cache / "config.json").write_text("{}", encoding="utf-8")
        self.assertEqual(self.engine.available_models()["sentiment"], True)
        with patch.object(self.engine, "_path", return_value=cache):
            first = self.engine.provenance()
            self.assertEqual(first["task_configs"]["sentiment"]["activation"], "softmax")
            (cache / "model.safetensors").write_bytes(b"new-longer-weights")
            second = self.engine.provenance()
        self.assertNotEqual(first["models"]["sentiment"]["files_sha256"]["model.safetensors"],
                            second["models"]["sentiment"]["files_sha256"]["model.safetensors"])

    def test_missing_model_has_actionable_error(self):
        self.engine._models.clear()
        with patch.object(self.engine, "_path", return_value=None):
            with self.assertRaisesRegex(ValueError, "Local emotion model missing.*Install"):
                self.engine.infer("hello")

    def test_profile_changes_emotion_selection_without_changing_sentiment(self):
        ordinary = self.engine.infer("hello")
        thresholds = dict.fromkeys(LABELS, .5)
        thresholds["joy"] = .9999
        profiled = self.engine.infer("hello", thresholds=thresholds)
        self.assertEqual(ordinary["labels"], ["joy"])
        self.assertEqual(profiled["labels"], [])
        self.assertEqual(profiled["sentiment"], ordinary["sentiment"])
        with patch.object(self.engine, "_load", side_effect=AssertionError("must validate first")):
            with self.assertRaises(ValueError):
                self.engine.infer("", thresholds={"joy": .3})

    def test_candidate_availability_requires_its_pinned_local_revision(self):
        engine = AnalysisEngine(self.root, emotion_model="modernbert")
        cache = self.root / "runs/model_cache/modernbert-go-emotions"
        cache.mkdir(parents=True)
        (cache / "model.safetensors").write_bytes(b"fixture")
        (cache / "config.json").write_text("{}")
        self.assertFalse(engine.available_models()["emotion"])
        metadata = cache / ".cache/huggingface/download/config.json.metadata"
        metadata.parent.mkdir(parents=True)
        metadata.write_text(EMOTION_MODELS["modernbert"]["revision"] + "\n")
        self.assertTrue(engine.available_models()["emotion"])
        self.assertEqual(engine.provenance(["emotion"])["models"]["emotion"]["id"], EMOTION_MODELS["modernbert"]["id"])
        metadata.write_text("another revision\n")
        self.assertFalse(engine.available_models()["emotion"])
        with self.assertRaises(ValueError):
            AnalysisEngine(self.root, emotion_model="unknown")


if __name__ == "__main__":
    unittest.main()
