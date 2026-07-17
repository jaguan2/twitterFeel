"""V2 preprocessing for the emotion-prediction LSTM.

Improvements vs. the original notebook pipeline:
  - Per-step features are 384-dim sentence embeddings + weekend + overnight
    (= 386 per step), not just emotion_encoded + 2 flags.
  - Fixed slicing for trailing partial batches (the v1 split_group bug
    silently dropped them).
  - Stride-1 sliding window per user -- ~9x more training data than v1's
    non-overlapping windows.

Output:
    windows_v2.npz  containing arrays:
        X             (N, 9, 386)  float32
        y             (N,)          int64
        user_ids      (N,)          int64
        last_emotion  (N,)          int64  -- 9th tweet's emotion, for persistence baseline
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from paths import DATA_INTERIM

INPUT_CSV = DATA_INTERIM / "dataset_godknowswhat.csv"
INPUT_EMB = DATA_INTERIM / "tweet_embeddings.npy"
OUTPUT_NPZ = DATA_INTERIM / "windows_v2.npz"

WINDOW = 10  # 9 history + 1 target
HISTORY = WINDOW - 1
EMBED_DIM = 384
MIN_USER_TWEETS = 20
MAX_USER_TWEETS = 1000


def overnight_tweet(hour: int) -> int:
    return 1 if hour in range(5) or hour == 23 else 0


def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    embeddings = np.load(INPUT_EMB)
    assert len(df) == len(embeddings), (
        f"row count mismatch: csv={len(df)} emb={len(embeddings)}"
    )
    print(f"Loaded {len(df):,} rows + embeddings {embeddings.shape}")

    df["post_created"] = pd.to_datetime(df["post_created"])
    df["hour_of_day"] = df["post_created"].dt.hour
    df["overnight"] = df["hour_of_day"].apply(overnight_tweet).astype(np.float32)
    df["weekend"] = (df["post_created"].dt.weekday >= 5).astype(np.float32)
    df["_orig_idx"] = np.arange(len(df))  # for joining back to embeddings

    counts = df.groupby("user_id").size()
    keep_users = counts[(counts >= MIN_USER_TWEETS) & (counts <= MAX_USER_TWEETS)].index
    df = df[df["user_id"].isin(keep_users)].copy()
    df = df.sort_values(["user_id", "post_created"]).reset_index(drop=True)
    print(f"After {MIN_USER_TWEETS}-{MAX_USER_TWEETS} tweet filter: "
          f"{len(df):,} rows, {df['user_id'].nunique()} users")

    X_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []
    uid_chunks: list[np.ndarray] = []
    last_emo_chunks: list[np.ndarray] = []

    for user_id, group in df.groupby("user_id", sort=False):
        n = len(group)
        if n < WINDOW:
            continue
        orig_idx = group["_orig_idx"].to_numpy()
        emo = group["emotion_encoded"].to_numpy(dtype=np.int64)
        flags = group[["weekend", "overnight"]].to_numpy(dtype=np.float32)
        embs = embeddings[orig_idx]  # (n, 384)

        per_step = np.concatenate([embs, flags], axis=1)  # (n, 386)

        # Stride-1 sliding window: window[i] = rows i..i+WINDOW-1
        num_windows = n - WINDOW + 1
        win_X = np.zeros((num_windows, HISTORY, EMBED_DIM + 2), dtype=np.float32)
        win_y = np.zeros(num_windows, dtype=np.int64)
        win_last_emo = np.zeros(num_windows, dtype=np.int64)
        for i in range(num_windows):
            win_X[i] = per_step[i : i + HISTORY]
            win_y[i] = emo[i + HISTORY]
            win_last_emo[i] = emo[i + HISTORY - 1]
        X_chunks.append(win_X)
        y_chunks.append(win_y)
        uid_chunks.append(np.full(num_windows, user_id, dtype=np.int64))
        last_emo_chunks.append(win_last_emo)

    X = np.concatenate(X_chunks, axis=0)
    y = np.concatenate(y_chunks, axis=0)
    user_ids = np.concatenate(uid_chunks, axis=0)
    last_emotion = np.concatenate(last_emo_chunks, axis=0)

    np.savez(
        OUTPUT_NPZ, X=X, y=y, user_ids=user_ids, last_emotion=last_emotion
    )
    print(f"\nWrote {OUTPUT_NPZ.name}: X={X.shape}, y={y.shape}, "
          f"users={len(np.unique(user_ids))}")

    print("\nTarget distribution:")
    for cls in range(6):
        print(f"  class {cls}: {int((y == cls).sum()):>5}")


if __name__ == "__main__":
    main()
