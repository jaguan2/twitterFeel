"""V3 preprocessing: V2 features + richer time signal + configurable window size.

Per-step features per tweet (vs V2's 386):
    384  text embedding (MiniLM)
      1  weekend flag
      1  overnight flag
      1  sin(2 pi * hour / 24)
      1  cos(2 pi * hour / 24)
      1  sin(2 pi * day_of_week / 7)
      1  cos(2 pi * day_of_week / 7)
      1  log1p(minutes_since_prev_tweet)  (0 for the first tweet in a window)
    ----
    391  total

Usage:
    python preprocess_v3.py [WINDOW_SIZE]

WINDOW_SIZE defaults to 10 (9 history + 1 target). Try 16 or 21 for longer
context. Outputs windows_v3_w{N}.npz.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from paths import DATA_INTERIM

INPUT_CSV = DATA_INTERIM / "dataset_godknowswhat.csv"
INPUT_EMB = DATA_INTERIM / "tweet_embeddings.npy"

EMBED_DIM = 384
EXTRA_PER_STEP = 7
PER_STEP = EMBED_DIM + EXTRA_PER_STEP
MIN_USER_TWEETS = 20
MAX_USER_TWEETS = 1000


def overnight_tweet(hour: int) -> int:
    return 1 if hour in range(5) or hour == 23 else 0


def main(window: int) -> None:
    history = window - 1
    out_path = DATA_INTERIM / f"windows_v3_w{window}.npz"

    df = pd.read_csv(INPUT_CSV)
    embeddings = np.load(INPUT_EMB)
    assert len(df) == len(embeddings)
    print(f"Loaded {len(df):,} rows + embeddings {embeddings.shape}")

    df["post_created"] = pd.to_datetime(df["post_created"])
    df["hour"] = df["post_created"].dt.hour
    df["dow"] = df["post_created"].dt.weekday
    df["overnight"] = df["hour"].apply(overnight_tweet).astype(np.float32)
    df["weekend"] = (df["dow"] >= 5).astype(np.float32)
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24).astype(np.float32)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24).astype(np.float32)
    df["sin_dow"] = np.sin(2 * np.pi * df["dow"] / 7).astype(np.float32)
    df["cos_dow"] = np.cos(2 * np.pi * df["dow"] / 7).astype(np.float32)
    df["_orig_idx"] = np.arange(len(df))

    counts = df.groupby("user_id").size()
    keep = counts[(counts >= MIN_USER_TWEETS) & (counts <= MAX_USER_TWEETS)].index
    df = df[df["user_id"].isin(keep)].copy()
    df = df.sort_values(["user_id", "post_created"]).reset_index(drop=True)
    print(
        f"After {MIN_USER_TWEETS}-{MAX_USER_TWEETS} tweet filter: "
        f"{len(df):,} rows, {df['user_id'].nunique()} users, window={window}"
    )

    feature_cols = [
        "weekend",
        "overnight",
        "sin_hour",
        "cos_hour",
        "sin_dow",
        "cos_dow",
    ]

    X_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []
    uid_chunks: list[np.ndarray] = []
    last_emo_chunks: list[np.ndarray] = []

    for user_id, group in df.groupby("user_id", sort=False):
        n = len(group)
        if n < window:
            continue
        orig_idx = group["_orig_idx"].to_numpy()
        embs = embeddings[orig_idx]  # (n, 384)
        feats = group[feature_cols].to_numpy(dtype=np.float32)  # (n, 6)
        emo = group["emotion_encoded"].to_numpy(dtype=np.int64)

        # minutes_since_prev_tweet (log1p)
        ts = group["post_created"].to_numpy().astype("datetime64[s]")
        gaps_seconds = np.diff(ts).astype(np.int64)
        gaps_minutes = np.concatenate([[0], gaps_seconds]).astype(np.float32) / 60.0
        gaps_minutes = np.clip(gaps_minutes, 0, None)
        log_gap = np.log1p(gaps_minutes).reshape(-1, 1)  # (n, 1)

        per_step = np.concatenate([embs, feats, log_gap], axis=1)  # (n, 391)
        assert per_step.shape[1] == PER_STEP

        num_windows = n - window + 1
        win_X = np.zeros((num_windows, history, PER_STEP), dtype=np.float32)
        win_y = np.zeros(num_windows, dtype=np.int64)
        win_last_emo = np.zeros(num_windows, dtype=np.int64)
        for i in range(num_windows):
            win_X[i] = per_step[i : i + history]
            win_y[i] = emo[i + history]
            win_last_emo[i] = emo[i + history - 1]
            # Zero the log_gap for the first step of each window (no "previous")
            win_X[i, 0, -1] = 0.0

        X_chunks.append(win_X)
        y_chunks.append(win_y)
        uid_chunks.append(np.full(num_windows, user_id, dtype=np.int64))
        last_emo_chunks.append(win_last_emo)

    X = np.concatenate(X_chunks, axis=0)
    y = np.concatenate(y_chunks, axis=0)
    user_ids = np.concatenate(uid_chunks, axis=0)
    last_emotion = np.concatenate(last_emo_chunks, axis=0)

    np.savez(out_path, X=X, y=y, user_ids=user_ids, last_emotion=last_emotion)
    print(f"\nWrote {out_path.name}: X={X.shape}, y={y.shape}, "
          f"users={len(np.unique(user_ids))}")

    print("\nTarget distribution:")
    for cls in range(6):
        print(f"  class {cls}: {int((y == cls).sum()):>5}")


if __name__ == "__main__":
    window = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    main(window)
