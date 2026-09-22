"""Audited next-retained-post windows; keeps row identity and emotion history."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from emotion_schema import PROBABILITY_COLUMNS, text_digest, file_digest


def build_windows(df, embeddings, history=9, min_posts=20, max_posts=1000):
    if history < 1 or min_posts < 1 or max_posts < min_posts:
        raise ValueError("Require history >= 1 and 1 <= min_posts <= max_posts")
    required = {"user_id", "post_created", "cleaned_text", "emotion_encoded"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(df.columns))}")
    if embeddings.shape != (len(df), 384) or not np.isfinite(embeddings).all():
        raise ValueError("Expected finite, row-aligned (N, 384) embeddings")
    df = df.copy()
    df["source_row"] = np.arange(len(df))
    df["post_created"] = pd.to_datetime(df["post_created"], errors="coerce", utc=True)
    audit = {
        "input_rows": len(df),
        "missing_text": int(df.cleaned_text.fillna("").str.strip().eq("").sum()),
        "invalid_time": int(df.post_created.isna().sum()),
        "missing_user": int(df.user_id.isna().sum()),
        "duplicate_user_time": int(df.duplicated(["user_id", "post_created"]).sum()),
        "duplicate_user_text": int(df.duplicated(["user_id", "cleaned_text"]).sum()),
    }
    if not df.emotion_encoded.isin(range(6)).all():
        raise ValueError("emotion_encoded must contain integer labels 0..5")
    has_probs = [c in df for c in PROBABILITY_COLUMNS]
    if any(has_probs) and not all(has_probs):
        raise ValueError("Incomplete emotion probability columns")
    if all(has_probs):
        probs = df[PROBABILITY_COLUMNS].to_numpy(dtype=np.float32)
        if (not np.isfinite(probs).all() or (probs < 0).any()
                or not np.allclose(probs.sum(axis=1), 1, atol=1e-5)):
            raise ValueError("Invalid emotion probability distributions")
        if not np.array_equal(probs.argmax(axis=1), df.emotion_encoded.to_numpy()):
            raise ValueError("Probability argmax disagrees with emotion_encoded")
        audit["emotion_history"] = "probabilities"
    else:
        probs = np.eye(6, dtype=np.float32)[df.emotion_encoded.to_numpy(dtype=int)]
        audit["emotion_history"] = "one_hot_fallback"

    valid = (df.user_id.notna() & df.post_created.notna()
             & df.cleaned_text.fillna("").str.strip().ne(""))
    df = df[valid].copy()
    # Keep repeated text at different times; drop only identical user/time/text records.
    before = len(df)
    df = df.drop_duplicates(["user_id", "post_created", "cleaned_text"])
    audit["exact_duplicates_removed"] = before - len(df)
    df["user_id"] = df.user_id.astype(str)
    df = df.sort_values(["user_id", "post_created", "source_row"], kind="stable")
    chunks = {k: [] for k in ("X", "emotion_history", "y", "user_ids", "source_rows", "target_time")}
    tied = 0
    for uid, group in df.groupby("user_id", sort=False):
        if not min_posts <= len(group) <= max_posts or len(group) <= history:
            continue
        rows = group.source_row.to_numpy()
        time = group.post_created.astype("int64").to_numpy() // 10**9
        hour = group.post_created.dt.hour.to_numpy()
        dow = group.post_created.dt.dayofweek.to_numpy()
        gap = np.log1p(np.r_[0, np.diff(time)] / 60)
        extra = np.column_stack((dow >= 5, (hour < 5) | (hour == 23),
                                 np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24),
                                 np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7), gap))
        features = np.concatenate((embeddings[rows], extra), axis=1).astype(np.float32)
        labels = group.emotion_encoded.to_numpy(dtype=np.int64)
        for end in range(history, len(group)):
            start = end - history
            # No inferred order inside timestamp ties: require strict chronology throughout.
            if (np.diff(time[start:end+1]) <= 0).any():
                tied += 1
                continue
            x = features[start:end].copy()
            x[0, -1] = 0
            chunks["X"].append(x)
            chunks["emotion_history"].append(probs[rows[start:end]])
            chunks["y"].append(labels[end])
            chunks["user_ids"].append(uid)
            chunks["source_rows"].append(rows[start:end+1])
            chunks["target_time"].append(time[end])
    if not chunks["y"]:
        raise ValueError("No windows remain after data-quality and user filters")
    audit.update(windows=len(chunks["y"]), timestamp_tie_windows_skipped=tied,
                 retained_users=len(set(chunks["user_ids"])), history=history,
                 min_posts=min_posts, max_posts=max_posts,
                 task="next retained post", time_basis="UTC")
    return {k: np.asarray(v) for k, v in chunks.items()}, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("embeddings", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--history", type=int, default=9)
    parser.add_argument("--min-posts", type=int, default=20)
    parser.add_argument("--max-posts", type=int, default=1000)
    parser.add_argument("--allow-unverified-embeddings", action="store_true",
                        help="Opt into legacy embeddings without a text-order manifest")
    args = parser.parse_args()
    if args.out.suffix != ".npz":
        parser.error("Output must end in .npz")
    if args.out.exists() or args.out.with_suffix(".json").exists():
        parser.error("Output exists; choose a new artifact name")
    df = pd.read_csv(args.csv, dtype={"user_id": str})
    manifest_path = args.embeddings.with_suffix(".json")
    verified = False
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        verified = (manifest["text_sha256"] == text_digest(df.cleaned_text.fillna("").astype(str))
                    and manifest["embedding_sha256"] == file_digest(args.embeddings))
        if not verified:
            raise ValueError("Embedding manifest does not match CSV text order or embedding file")
    elif not args.allow_unverified_embeddings:
        parser.error("Missing embedding manifest; regenerate or use --allow-unverified-embeddings")
    arrays, audit = build_windows(df, np.load(args.embeddings, allow_pickle=False),
                                  args.history, args.min_posts, args.max_posts)
    audit.update(csv_sha256=file_digest(args.csv), embeddings_sha256=file_digest(args.embeddings),
                 embedding_alignment_verified=verified)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    audit["npz_sha256"] = file_digest(args.out)
    args.out.with_suffix(".json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
