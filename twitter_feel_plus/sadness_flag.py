"""Per-user rolling sadness signal + sustained-flag detection.

For each user in the corpus, computes a rolling fraction of tweets whose
DistilBERT-predicted emotion is `sadness`, both:

  - over a fixed window of N consecutive tweets (tweet-window), and
  - over a fixed window of N days (time-window).

A window is "flagged" if its sadness fraction crosses BOTH:
  (a) an absolute threshold (default 0.40), AND
  (b) the user's personal long-run baseline by at least DEV_THRESHOLD (default +0.15).

Consecutive flagged windows of length >= MIN_SUSTAINED form a "flag period."
Outputs (under twitter_feel_plus/results/):

  per_user_sadness.csv  -- one row per user with summary stats
  flag_windows.csv      -- one row per flagged period (anonymous_id, start, end, ...)

User IDs are replaced with sequential pseudonyms (user_001, user_002, ...) so
no real Twitter handle is exposed in the outputs. The mapping is NOT saved.

This is a signal-detection method. It is NOT a diagnostic tool. See the
twitter_feel_plus/README.md for the full list of caveats.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from paths import RESULTS_DIR, SOURCE_CSV

# ---- configurable parameters --------------------------------------------------
SADNESS_LABEL = "sadness"
TWEET_WINDOW = 20             # rolling window size in number of tweets
DAY_WINDOW = 14               # rolling window size in days
ABS_THRESHOLD = 0.40          # window flagged if sadness fraction > this
DEV_THRESHOLD = 0.15          # AND if it exceeds user's personal baseline by this much
MIN_SUSTAINED = 3             # min consecutive flagged windows to form a flag period


def load_corpus() -> pd.DataFrame:
    df = pd.read_csv(SOURCE_CSV)
    required = {"user_id", "post_created", "emotion"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"source csv is missing columns: {missing}")
    df["post_created"] = pd.to_datetime(df["post_created"], utc=True)
    df = df.sort_values(["user_id", "post_created"]).reset_index(drop=True)
    df["is_sad"] = (df["emotion"] == SADNESS_LABEL).astype(np.int8)
    return df


def runs_of_true(mask: np.ndarray, min_length: int) -> list[tuple[int, int]]:
    """Return [(start_idx, end_idx_inclusive), ...] for runs of True at least
    min_length long. Empty list if no qualifying runs."""
    runs = []
    i = 0
    n = len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        if j - i >= min_length:
            runs.append((i, j - 1))
        i = j
    return runs


def time_rolling_frac(group: pd.DataFrame, window_days: int) -> pd.Series:
    """Rolling sadness fraction over a calendar-day window, indexed by tweet."""
    s = pd.Series(
        group["is_sad"].to_numpy(),
        index=group["post_created"].to_numpy(),
    )
    # rolling on a DatetimeIndex needs a sorted index and a Timedelta-like window
    window = pd.Timedelta(days=window_days)
    return s.rolling(window, min_periods=1).mean().reset_index(drop=True)


def analyze_user(anon_id: str, group: pd.DataFrame) -> tuple[dict, list[dict]]:
    """Compute per-user summary + list of flag periods for one user."""
    group = group.reset_index(drop=True)
    n = len(group)
    baseline = float(group["is_sad"].mean())

    # tweet-count rolling
    tw_roll = (
        group["is_sad"]
        .rolling(window=TWEET_WINDOW, min_periods=TWEET_WINDOW)
        .mean()
    )
    # day-window rolling
    day_roll = time_rolling_frac(group, DAY_WINDOW)

    # Flag mask (BOTH conditions, applied to tweet-count rolling for indexing)
    flagged = (
        (tw_roll > ABS_THRESHOLD)
        & (tw_roll > baseline + DEV_THRESHOLD)
    ).fillna(False).to_numpy()

    runs = runs_of_true(flagged, MIN_SUSTAINED)

    flag_records = []
    for start_idx, end_idx in runs:
        # Window at position k covers tweets [k - TWEET_WINDOW + 1 .. k], so
        # the "true span" of the flag period in tweet space is:
        span_start_idx = max(0, start_idx - TWEET_WINDOW + 1)
        span_end_idx = end_idx
        peak_frac = float(tw_roll.iloc[start_idx : end_idx + 1].max())
        start_ts = group["post_created"].iloc[span_start_idx]
        end_ts = group["post_created"].iloc[span_end_idx]
        duration_days = (end_ts - start_ts).total_seconds() / 86400.0
        flag_records.append(
            {
                "anonymous_id": anon_id,
                "flag_start_ts": start_ts.isoformat(),
                "flag_end_ts": end_ts.isoformat(),
                "duration_days": round(duration_days, 2),
                "n_tweets_in_span": int(span_end_idx - span_start_idx + 1),
                "peak_window_sad_frac": round(peak_frac, 4),
                "user_baseline_sad_frac": round(baseline, 4),
                "excess_over_baseline": round(peak_frac - baseline, 4),
            }
        )

    per_user = {
        "anonymous_id": anon_id,
        "total_tweets": n,
        "baseline_sad_frac": round(baseline, 4),
        "max_tweet_window_sad_frac": (
            None if tw_roll.dropna().empty else round(float(tw_roll.max()), 4)
        ),
        "max_day_window_sad_frac": round(float(day_roll.max()), 4),
        "num_flag_periods": len(flag_records),
        "total_flagged_days": (
            round(sum(r["duration_days"] for r in flag_records), 2)
            if flag_records
            else 0.0
        ),
        "first_tweet_ts": group["post_created"].iloc[0].isoformat(),
        "last_tweet_ts": group["post_created"].iloc[-1].isoformat(),
    }
    return per_user, flag_records


def main() -> None:
    df = load_corpus()
    print(
        f"Loaded {len(df):,} tweets across {df['user_id'].nunique()} users; "
        f"corpus sadness rate = {df['is_sad'].mean():.3f}"
    )

    # Assign deterministic anonymous IDs in order of first appearance.
    user_order = (
        df.sort_values("post_created")
        .groupby("user_id", sort=False)
        .head(1)["user_id"]
        .tolist()
    )
    anon_map = {uid: f"user_{i + 1:03d}" for i, uid in enumerate(user_order)}

    per_user_records: list[dict] = []
    flag_records: list[dict] = []

    for user_id, group in df.groupby("user_id", sort=False):
        anon_id = anon_map[user_id]
        per_user, flags = analyze_user(anon_id, group)
        per_user_records.append(per_user)
        flag_records.extend(flags)

    per_user_df = pd.DataFrame(per_user_records).sort_values("anonymous_id")
    flag_df = pd.DataFrame(flag_records)
    if not flag_df.empty:
        flag_df = flag_df.sort_values(["anonymous_id", "flag_start_ts"])

    per_user_out = RESULTS_DIR / "per_user_sadness.csv"
    flag_out = RESULTS_DIR / "flag_windows.csv"
    per_user_df.to_csv(per_user_out, index=False)
    flag_df.to_csv(flag_out, index=False)

    # Also write a small params record so the outputs are self-describing.
    params_out = RESULTS_DIR / "params.json"
    params_out.write_text(
        json.dumps(
            {
                "source_csv": str(SOURCE_CSV.relative_to(SOURCE_CSV.parents[2])),
                "sadness_label": SADNESS_LABEL,
                "tweet_window": TWEET_WINDOW,
                "day_window_days": DAY_WINDOW,
                "abs_threshold": ABS_THRESHOLD,
                "dev_threshold": DEV_THRESHOLD,
                "min_sustained": MIN_SUSTAINED,
                "n_users": int(per_user_df.shape[0]),
                "n_flag_periods": int(flag_df.shape[0]),
                "users_with_at_least_one_flag": int(
                    (per_user_df["num_flag_periods"] > 0).sum()
                ),
            },
            indent=2,
        )
    )

    print(f"Wrote {per_user_out.name}  ({len(per_user_df)} users)")
    print(f"Wrote {flag_out.name}      ({len(flag_df)} flag periods)")
    print(f"Wrote {params_out.name}")
    flagged_users = int((per_user_df["num_flag_periods"] > 0).sum())
    print(
        f"\nSummary: {flagged_users}/{len(per_user_df)} users had at least one "
        f"flagged period under params "
        f"(tweet_window={TWEET_WINDOW}, abs>{ABS_THRESHOLD}, "
        f"excess_over_baseline>{DEV_THRESHOLD}, "
        f"min_sustained={MIN_SUSTAINED} windows)."
    )


if __name__ == "__main__":
    main()
