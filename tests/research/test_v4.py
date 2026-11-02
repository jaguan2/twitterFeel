"""Regression checks for temporal boundaries, grouped evaluation and ingestion."""
import json
import contextlib
import io
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from twitterfeel.research.forecasting.emotion_schema import PROBABILITY_COLUMNS, text_digest
from twitterfeel.research.forecasting.evaluation import make_splits, clustered_accuracy_interval, scores
from twitterfeel.research.forecasting.preprocess_v4 import build_windows
from twitterfeel.research.forecasting.train_v4 import prepare_task, fit_predict
from twitterfeel.research.forecasting.train_v4 import main as train_main
from twitterfeel.research.forecasting.preprocess_v4 import main as preprocess_main
from twitterfeel.research.forecasting.prepare_reddit import collect, sample_authors, main as reddit_main
from twitterfeel.research.forecasting.prepare_sentiment140 import main as prepare_s140


def fixture():
    df = pd.DataFrame({"user_id": ["a"] * 5,
                       "post_created": pd.date_range("2020-01-01", periods=5, freq="h"),
                       "cleaned_text": [f"text {i}" for i in range(5)],
                       "emotion_encoded": [0, 1, 2, 3, 4]})
    embeddings = np.repeat(np.arange(5, dtype=np.float32)[:, None], 384, axis=1)
    return df, embeddings


class WindowsTest(unittest.TestCase):
    def test_sorted_alignment_and_no_target_features(self):
        df, embeddings = fixture()
        order = [3, 1, 4, 0, 2]
        data, audit = build_windows(df.iloc[order], embeddings[order], 2, 1, 100)
        np.testing.assert_array_equal(data["X"][:, :, 0], [[0, 1], [1, 2], [2, 3]])
        np.testing.assert_array_equal(data["y"], [2, 3, 4])
        np.testing.assert_array_equal(data["emotion_history"].argmax(axis=-1), [[0, 1], [1, 2], [2, 3]])
        self.assertTrue((data["X"][:, 0, -1] == 0).all())
        self.assertEqual(audit["emotion_history"], "one_hot_fallback")

    def test_timestamp_ties_never_create_implied_order(self):
        df, embeddings = fixture()
        df.loc[1, "post_created"] = df.loc[0, "post_created"]
        data, audit = build_windows(df, embeddings, 2, 1, 100)
        self.assertEqual(audit["timestamp_tie_windows_skipped"], 1)
        self.assertEqual(data["y"].tolist(), [3, 4])

    def test_exact_duplicates_and_empty_text_removed_with_alignment(self):
        df, embeddings = fixture()
        df.loc[1] = df.loc[0]
        df.loc[2, "cleaned_text"] = " "
        data, audit = build_windows(df, embeddings, 1, 1, 100)
        self.assertEqual(audit["exact_duplicates_removed"], 1)
        self.assertEqual(data["source_rows"].tolist(), [[0, 3], [3, 4]])

    def test_probability_validation(self):
        df, embeddings = fixture()
        df[PROBABILITY_COLUMNS] = np.eye(6)[df.emotion_encoded]
        data, audit = build_windows(df, embeddings, 2, 1, 100)
        self.assertEqual(audit["emotion_history"], "probabilities")
        df.loc[0, PROBABILITY_COLUMNS[0]] = .5
        with self.assertRaisesRegex(ValueError, "distributions"):
            build_windows(df, embeddings, 2, 1, 100)

    def test_hash_is_order_sensitive(self):
        self.assertNotEqual(text_digest(["ab", "c"]), text_digest(["a", "bc"]))
        self.assertNotEqual(text_digest(["a", "b"]), text_digest(["b", "a"]))


class EvaluationTest(unittest.TestCase):
    def test_no_users_cross_folds_or_holdout(self):
        users = np.repeat(np.arange(20), 5)
        dev, test, folds = make_splits(users, 42, 3)
        self.assertFalse(set(users[dev]) & set(users[test]))
        np.testing.assert_array_equal(np.sort(np.concatenate([va for _, va in folds])), dev)
        for train, val in folds:
            self.assertFalse(set(users[train]) & set(users[val]))
            self.assertFalse(set(users[train]) & set(users[test]))
        np.testing.assert_array_equal(make_splits(users, 42, 3)[1], test)

    def test_paired_interval_identical_predictions_is_zero(self):
        y = np.array([0, 1, 1, 0])
        p = np.array([[.8, .2], [.3, .7], [.2, .8], [.4, .6]])
        interval = clustered_accuracy_interval(y, p, np.array([0, 0, 1, 1]), p, draws=100)
        self.assertEqual([interval[k] for k in ["estimate", "lower", "upper"]], [0, 0, 0])

    def test_fixed_label_macro_and_single_class_auc(self):
        result = scores(np.array([0, 0]), np.array([[1., 0.], [1., 0.]]), np.array([0, 1]))
        self.assertEqual(result["macro_f1"], .5)
        self.assertIsNone(result["roc_auc"])

    def test_four_class_unknown_history_is_valid_distribution(self):
        d = {"y": np.array([0, 2, 3]), "emotion_history": np.eye(6)[[2, 5, 1]][:, None, :]}
        y, h, keep, _ = prepare_task(d, "4cls")
        self.assertEqual(keep.tolist(), [True, False, True])
        np.testing.assert_allclose(h.sum(axis=-1), 1)
        np.testing.assert_allclose(h[0], [[.25]*4])

    def test_transition_fits_training_targets_only(self):
        from argparse import Namespace
        y = np.array([0, 1, 0, 1])
        h = np.tile([[[1., 0.]]], (4, 1, 1))
        args = Namespace(features="text-time", class_weight="none", seed=42)
        p1, _ = fit_predict("transition", None, h, y, None, np.array([0, 1]), np.array([2, 3]), args, 2)
        y[2:] = 1-y[2:]
        p2, _ = fit_predict("transition", None, h, y, None, np.array([0, 1]), np.array([2, 3]), args, 2)
        np.testing.assert_array_equal(p1, p2)


class IngestTest(unittest.TestCase):
    def test_reddit_cli_closes_temporary_database(self):
        with tempfile.TemporaryDirectory() as temp:
            raw, out = Path(temp)/"raw.jsonl", Path(temp)/"out.csv"
            raw.write_text("\n".join(json.dumps({"author": "person", "body": f"hello world {i}",
                            "created_utc": 100+i, "subreddit": "test"}) for i in range(20)), encoding="utf-8")
            with patch.object(sys, "argv", ["prepare_reddit", str(raw), str(out)]):
                reddit_main()
            self.assertEqual(len(pd.read_csv(out)), 20)

    def test_reddit_filters_and_excludes_over_cap_authors(self):
        with tempfile.TemporaryDirectory() as temp:
            raw = Path(temp) / "input.jsonl"
            records = []
            for author, count in [("person", 3), ("prolific", 5), ("somebot", 3)]:
                records.extend({"author": author, "body": f"hello world {i}", "created_utc": 100+i,
                                "subreddit": "test"} for i in range(count))
            raw.write_text("null\n{invalid\n" + "\n".join(json.dumps(r) for r in records), encoding="utf-8")
            with sqlite3.connect(":memory:") as con, patch("twitterfeel.research.forecasting.prepare_reddit.MIN_USER_POSTS", 2), patch("twitterfeel.research.forecasting.prepare_reddit.MAX_USER_POSTS", 4):
                collect(raw, 3, 60, None, set(), con)
                rows = sample_authors(con, 10)
                self.assertEqual(len(rows), 3)
                self.assertEqual({r[0] for r in rows}, {"person"})

    def test_sentiment_spam_ratio_computed_before_dedup(self):
        with tempfile.TemporaryDirectory() as temp:
            raw, out = Path(temp)/"raw.csv", Path(temp)/"out.csv"
            rows = []
            for author, count in [("spam", 80), ("person", 20)]:
                for i in range(count):
                    rows.append([0, len(rows), "Mon Apr 06 22:19:45 PDT 2009", "NO_QUERY", author,
                                 f"{author} message {i % 20}"])
            pd.DataFrame(rows).to_csv(raw, header=False, index=False)
            prepare_s140(raw, out)
            result = pd.read_csv(out)
            self.assertEqual(len(result), 20)
            self.assertTrue(result.cleaned_text.str.startswith("person").all())


class WorkflowTest(unittest.TestCase):
    def make_data(self, root):
        rows = [(str(u), f"2020-01-01 {i:02d}:00:00", f"post {u} {i}", (u+i)%6)
                for u in range(12) for i in range(8)]
        df = pd.DataFrame(rows, columns=["user_id", "post_created", "cleaned_text", "emotion_encoded"])
        emb = np.random.default_rng(42).normal(size=(len(df), 384)).astype(np.float32)
        arrays, _ = build_windows(df, emb, history=2, min_posts=3)
        path = root / "windows.npz"
        np.savez_compressed(path, **arrays)
        return path, arrays

    def test_development_run_does_not_score_test_or_overwrite(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            source, _ = self.make_data(root)
            out = root / "run"
            argv = ["train_v4", str(source), "--out", str(out), "--models", "frequency", "--folds", "2"]
            with patch.object(sys, "argv", argv):
                train_main()
            result = json.loads((out/"metrics.json").read_text())
            self.assertNotIn("test", result["frequency"])
            self.assertFalse((out/"frequency_test.npz").exists())
            with np.load(out/"splits.npz") as split, np.load(out/"frequency_development.npz") as pred:
                test_users = set(split["user_ids"][split["assignment"] == "test"])
                self.assertFalse(test_users & set(pred["user_ids"]))
                self.assertTrue(np.isfinite(pred["probability"]).all())
            with patch.object(sys, "argv", argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    train_main()

    def test_four_class_final_model_roundtrip(self):
        import joblib
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            source, arrays = self.make_data(root)
            out = root / "final"
            argv = ["train_v4", str(source), "--out", str(out), "--models", "logistic",
                    "--folds", "2", "--task", "4cls", "--evaluate-test"]
            with patch.object(sys, "argv", argv):
                train_main()
            _, h, _, _ = prepare_task(arrays, "4cls")
            inputs = np.concatenate((arrays["X"], h), axis=-1)
            flat = np.concatenate((inputs[:, -1], inputs.mean(axis=1)), axis=1)
            with np.load(out/"logistic_test.npz") as pred:
                expected = joblib.load(out/"model.joblib").predict_proba(flat[pred["row_index"]])
                np.testing.assert_allclose(pred["probability"], expected)

    def test_preprocess_rejects_misaligned_embedding_manifest(self):
        from twitterfeel.common.provenance import file_digest
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            df, embeddings = fixture()
            csv, emb, out = root/"data.csv", root/"emb.npy", root/"out.npz"
            df.to_csv(csv, index=False)
            np.save(emb, embeddings)
            emb.with_suffix(".json").write_text(json.dumps({
                "text_sha256": text_digest(list(reversed(df.cleaned_text))),
                "embedding_sha256": file_digest(emb)}))
            argv = ["preprocess_v4", str(csv), str(emb), str(out), "--min-posts", "1", "--history", "2"]
            with patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(ValueError, "manifest"):
                    preprocess_main()
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
