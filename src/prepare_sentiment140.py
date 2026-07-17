"""Prepare a Sentiment140 subsample in the schema the pipeline expects.

Sentiment140 (Go, Bhayani & Huang 2009) is a 1.6M-tweet corpus with user
handles and timestamps — ~6,200 users have the >=20 tweets our windowing
needs, vs 54 users in the Mental-Health-Twitter corpus. FINDINGS.md ranks
"more users" as the top-impact fix, so this script adapts that corpus to
the existing pipeline.

Reads the raw training CSV (no header: target, tweet_id, date, flag, user,
text), cleans text (HTML entities, URLs, @mentions), drops repetitive bot
accounts (e.g. `lost_dog` posted the same plea 549 times), filters to
users with MIN-MAX usable tweets, subsamples users to a total-tweet budget
(CPU inference cost), and writes a CSV with the columns downstream scripts
use: user_id (int), post_created, cleaned_text.

Note: Sentiment140 was collected by emoticon search, so each user's rows
are a *sample* of their timeline, not consecutive tweets. The V3 log
time-gap feature at least exposes the spacing to the model.

Usage:
    python prepare_sentiment140.py RAW_CSV [OUT_CSV]

OUT_CSV defaults to data/interim/dataset_s140_unlabeled.csv; run
generate_dataset.py and generate_embeddings.py on it next.
"""
from __future__ import annotations

import html
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from paths import DATA_INTERIM

DEFAULT_OUT = DATA_INTERIM / "dataset_s140_unlabeled.csv"
MIN_USER_TWEETS = 20
MAX_USER_TWEETS = 1000
MIN_UNIQUE_RATIO = 0.5  # below this share of distinct texts, treat as bot
TWEET_BUDGET = 50_000
RANDOM_SEED = 42

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = URL_RE.sub("", text)
    text = MENTION_RE.sub("", text)
    return WS_RE.sub(" ", text).strip()


def main(raw_csv: Path, out_csv: Path) -> None:
    df = pd.read_csv(
        raw_csv,
        encoding="latin-1",
        header=None,
        names=["target", "tweet_id", "date", "flag", "user", "text"],
    )
    print(f"Loaded {len(df):,} raw rows")

    df = df.drop_duplicates("tweet_id")
    df["cleaned_text"] = df["text"].astype(str).map(clean_text)
    df = df[df["cleaned_text"].str.len() >= 3]
    df = df.drop_duplicates(["user", "cleaned_text"])
    print(f"After cleaning + dedup: {len(df):,} rows")

    stats = df.groupby("user")["cleaned_text"].agg(["size", "nunique"])
    eligible = stats[
        (stats["size"] >= MIN_USER_TWEETS)
        & (stats["size"] <= MAX_USER_TWEETS)
        & (stats["nunique"] / stats["size"] >= MIN_UNIQUE_RATIO)
    ]
    print(f"Eligible users ({MIN_USER_TWEETS}-{MAX_USER_TWEETS} tweets, "
          f"unique-ratio >= {MIN_UNIQUE_RATIO}): {len(eligible):,} "
          f"({int(eligible['size'].sum()):,} tweets)")

    rng = np.random.default_rng(RANDOM_SEED)
    users = eligible.index.to_numpy()
    rng.shuffle(users)
    counts = eligible.loc[users, "size"].to_numpy()
    n_keep = int(np.searchsorted(np.cumsum(counts), TWEET_BUDGET)) + 1
    keep_users = set(users[: min(n_keep, len(users))])

    df = df[df["user"].isin(keep_users)].copy()
    # "Mon Apr 06 22:19:45 PDT 2009" — corpus is entirely PDT
    df["post_created"] = pd.to_datetime(
        df["date"].str.replace(" PDT ", " ", regex=False),
        format="%a %b %d %H:%M:%S %Y",
    )
    df["user_id"] = pd.factorize(df["user"])[0]
    df = df.sort_values(["user_id", "post_created"]).reset_index(drop=True)

    out = df[["user_id", "post_created", "cleaned_text"]]
    out.to_csv(out_csv, index=False)
    print(f"Wrote {len(out):,} tweets, {out['user_id'].nunique():,} users "
          f"-> {out_csv.name}")


if __name__ == "__main__":
    raw = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUT
    main(raw, out)
