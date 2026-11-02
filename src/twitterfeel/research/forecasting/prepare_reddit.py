"""Prepare a Reddit dump subsample in the schema the pipeline expects.

Reads a Pushshift-style newline-delimited JSON dump (one comment or
submission per line) and writes the three columns every downstream script
needs -- user_id, post_created, cleaned_text -- plus `subreddit` for the
confound analysis described below. From there the corpus flows through the
existing pipeline unchanged, via the DATASET_TAG mechanism:

    python prepare_reddit.py RC_2019-01.zst
    python generate_dataset.py    data/interim/dataset_reddit_unlabeled.csv \\
                                  data/interim/dataset_reddit.csv
    python generate_embeddings.py data/interim/dataset_reddit_unlabeled.csv \\
                                  data/interim/tweet_embeddings_reddit.npy
    python preprocess_v3.py 10 reddit
    python train_v3.py windows_reddit_w10.npz --binary

Dumps are streamed, never loaded whole -- a month of comments is tens of
GB compressed. Accepts .zst (the Pushshift default), .bz2, .gz, or plain
text.

Why Reddit is worth the trouble: unlike the Sentiment140 subsample, whose
rows are an emoticon-search *sample* of each timeline, a Reddit dump gives
denser per-author history. Length, content and subreddit filters still remove
intervening posts: the task is "next retained post", not necessarily next post.

Three caveats this script only partly mitigates, all worth reading before
trusting any result built on its output:

1. LABELER DOMAIN SHIFT. generate_dataset.py runs a DistilBERT trained on
   `dair-ai/emotion` (short, tweet-shaped text) truncated at 128 tokens.
   Reddit comments are longer and differently registered, so labels are
   noisier here than on Twitter. --min-words/--max-words keeps comments
   tweet-shaped to narrow the gap; it does not close it.
2. SUBREDDIT CONFOUND. A comment in r/aww is labeled joy and one in
   r/depression sadness largely regardless of the author's trajectory, so
   the model can learn "which subreddit" instead of "this user's emotional
   dynamics" -- a shortcut with the same flavor as the row-level leak ruled
   out in EXPERIMENTS.md E4. `subreddit` is emitted so this can be tested
   both ways; --exclude-subreddits restricts collection up front.
3. UTC TIMESTAMPS. Reddit stores created_utc, and a dump carries no user
   timezone. The `overnight` and cyclical-hour features in preprocess_v3.py
   assume *local* time, so on a global userbase they are substantially
   weaker signal here than on the (entirely PDT) Sentiment140 corpus.
"""
from __future__ import annotations

import argparse
import bz2
import gzip
import html
import io
import json
import re
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

from twitterfeel.common.config import ensure_research_directories, DATA_INTERIM

DEFAULT_OUT = DATA_INTERIM / "dataset_reddit_unlabeled.csv"
MIN_USER_POSTS = 20      # matches preprocess_v3.MIN_USER_TWEETS
MAX_USER_POSTS = 1000    # matches preprocess_v3.MAX_USER_TWEETS
MIN_UNIQUE_RATIO = 0.5   # below this share of distinct texts, treat as a bot
RANDOM_SEED = 42

# Authors that are structurally not people.
SKIP_AUTHORS = {"[deleted]", "[removed]", "AutoModerator", "None", ""}
BOT_NAME_RE = re.compile(r"(?:bot|automoderator|transcriber)s?$", re.I)
DELETED_BODIES = {"[deleted]", "[removed]", ""}

CODE_FENCE_RE = re.compile(r"```.*?```", re.S)
INLINE_CODE_RE = re.compile(r"`[^`]*`")
QUOTE_LINE_RE = re.compile(r"^\s*>.*$", re.M)
MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
USER_MENTION_RE = re.compile(r"/?u/[A-Za-z0-9_-]+")
WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Strip Reddit markup down to the author's own prose.

    Order matters: unescape first, because quoted lines arrive as `&gt;`,
    and quotes must go before anything else reads the text as prose --
    they are *other people's words* and would otherwise be fed to the
    emotion labeler as if the author had written them.
    """
    text = html.unescape(text)
    text = text.replace("​", " ")  # zero-width space, endemic in dumps
    text = CODE_FENCE_RE.sub(" ", text)
    text = INLINE_CODE_RE.sub(" ", text)
    text = QUOTE_LINE_RE.sub(" ", text)
    text = MD_LINK_RE.sub(r"\1", text)
    text = URL_RE.sub("", text)
    text = USER_MENTION_RE.sub("", text)
    text = text.replace("*", "").replace("^", "").replace("~", "")
    return WS_RE.sub(" ", text).strip()


def extract_text(rec: dict) -> str:
    """Comments carry `body`; submissions carry `title` + `selftext`."""
    body = rec.get("body")
    if body is not None:
        return str(body)
    title = str(rec.get("title") or "")
    selftext = str(rec.get("selftext") or "")
    return f"{title}. {selftext}".strip() if selftext else title


def _parse_lines(stream):
    """Yield JSON objects, tolerating the occasional corrupt line."""
    for line in stream:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
            if isinstance(record, dict):
                yield record
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue


def iter_records(path: Path):
    """Stream records from a dump, transparently handling compression."""
    suffix = path.suffix.lower()
    if suffix == ".zst":
        try:
            import zstandard
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise SystemExit(
                "Reading .zst dumps needs the `zstandard` package:\n"
                "    pip install zstandard"
            ) from exc
        # Pushshift dumps are written with a large compression window;
        # the default decompressor rejects them without max_window_size.
        dctx = zstandard.ZstdDecompressor(max_window_size=2**31)
        with path.open("rb") as fh, dctx.stream_reader(fh) as reader:
            yield from _parse_lines(
                io.TextIOWrapper(reader, encoding="utf-8", errors="ignore")
            )
    elif suffix == ".bz2":
        with bz2.open(path, "rt", encoding="utf-8", errors="ignore") as fh:
            yield from _parse_lines(fh)
    elif suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as fh:
            yield from _parse_lines(fh)
    else:
        with path.open("rt", encoding="utf-8", errors="ignore") as fh:
            yield from _parse_lines(fh)


def collect(
    raw: Path,
    min_words: int,
    max_words: int,
    max_records: int | None,
    excluded: set[str],
    connection: sqlite3.Connection,
) -> None:
    """Stream into a disk-backed staging table; RAM does not grow with authors.

    Counts include posts past the per-author storage cap, so overactive users
    are excluded instead of appearing to have exactly MAX_USER_POSTS posts.
    Temporary disk usage grows with retained text; --budget limits final output.
    """
    connection.executescript("""
        CREATE TABLE authors (author TEXT PRIMARY KEY, count INTEGER NOT NULL);
        CREATE TABLE posts (author TEXT, created INTEGER, text TEXT, subreddit TEXT);
        CREATE INDEX posts_author ON posts(author);
    """)
    seen = kept = 0

    for rec in iter_records(raw):
        seen += 1
        if max_records is not None and seen > max_records:
            print(f"  reached --max-records={max_records:,}, stopping read")
            break
        if seen % 1_000_000 == 0:
            connection.commit()
            print(f"  read {seen:,} records, staged {kept:,}")

        author = str(rec.get("author") or "")
        if author in SKIP_AUTHORS or BOT_NAME_RE.search(author):
            continue

        subreddit = str(rec.get("subreddit") or "")
        if subreddit.lower() in excluded:
            continue

        raw_text = extract_text(rec)
        if raw_text in DELETED_BODIES:
            continue
        text = clean_text(raw_text)
        n_words = text.count(" ") + 1 if text else 0
        if not (min_words <= n_words <= max_words):
            continue

        created = rec.get("created_utc")
        if created is None:
            continue
        try:
            created = int(created)
        except (TypeError, ValueError):
            continue

        # Reject timestamps pandas cannot represent before writing the final CSV.
        if not -9223372036 <= created <= 9223372036:
            continue
        connection.execute("INSERT INTO authors VALUES (?, 1) ON CONFLICT(author) "
                           "DO UPDATE SET count=count+1", (author,))
        count = connection.execute("SELECT count FROM authors WHERE author=?", (author,)).fetchone()[0]
        if count > MAX_USER_POSTS:
            continue
        connection.execute("INSERT INTO posts VALUES (?, ?, ?, ?)",
                           (author, created, text, subreddit))
        kept += 1

    connection.commit()
    print(f"Streamed {seen:,} records -> staged {kept:,} posts on disk")


def sample_authors(connection, budget):
    """Shuffle eligible author IDs only, then load selected histories."""
    eligible = connection.execute("""
        SELECT a.author, a.count FROM authors a JOIN posts p ON a.author=p.author
        WHERE a.count BETWEEN ? AND ? GROUP BY a.author
        HAVING CAST(COUNT(DISTINCT p.text) AS REAL)/COUNT(*) >= ?
        ORDER BY a.author
    """, (MIN_USER_POSTS, MAX_USER_POSTS, MIN_UNIQUE_RATIO)).fetchall()
    if not eligible:
        raise ValueError("No eligible authors; increase the dump slice")
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(eligible)
    rows = []
    for author, _ in eligible:
        rows.extend(connection.execute(
            "SELECT author, created, text, subreddit FROM posts WHERE author=? ORDER BY created, rowid",
            (author,)).fetchall())
        if len(rows) >= budget:
            break
    print(f"Eligible authors: {len(eligible):,}; sampled {len(rows):,} posts")
    return rows


def main() -> None:
    ensure_research_directories()
    parser = argparse.ArgumentParser(
        description="Prepare a Reddit dump for the TwitterFeel pipeline."
    )
    parser.add_argument("raw", type=Path, help="Dump file (.zst/.bz2/.gz/.jsonl)")
    parser.add_argument("out", type=Path, nargs="?", default=DEFAULT_OUT)
    parser.add_argument(
        "--budget", type=int, default=50_000,
        help="Target post count (default 50000, matching the s140 subsample)",
    )
    parser.add_argument(
        "--min-words", type=int, default=3,
        help="Drop comments shorter than this after cleaning (default 3)",
    )
    parser.add_argument(
        "--max-words", type=int, default=60,
        help="Drop comments longer than this, to stay near the tweet-shaped "
             "distribution the emotion labeler was trained on (default 60)",
    )
    parser.add_argument(
        "--max-records", type=int, default=None,
        help="Stop after reading N raw records. Dumps are roughly "
             "chronological, so this takes a coherent time slice off the "
             "front rather than a random sample.",
    )
    parser.add_argument(
        "--exclude-subreddits", type=str, default="",
        help="Comma-separated subreddits to skip (case-insensitive). "
             "Consider excluding mental-health support subs, where the "
             "label reflects the venue more than the author.",
    )
    args = parser.parse_args()
    if (args.budget < 1 or args.min_words < 1 or args.max_words < args.min_words
            or (args.max_records is not None and args.max_records < 1)):
        parser.error("Require positive budget/record cap and 1 <= min-words <= max-words")

    excluded = {
        s.strip().lower().removeprefix("r/")
        for s in args.exclude_subreddits.split(",")
        if s.strip()
    }
    if excluded:
        print(f"Excluding subreddits: {sorted(excluded)}")

    with tempfile.TemporaryDirectory(prefix="twitterfeel_reddit_") as staging:
        with closing(sqlite3.connect(str(Path(staging) / "posts.sqlite"))) as connection:
            collect(args.raw, args.min_words, args.max_words, args.max_records, excluded, connection)
            rows = sample_authors(connection, args.budget)
    df = pd.DataFrame(rows, columns=["author", "created_utc", "cleaned_text", "subreddit"])
    df["post_created"] = pd.to_datetime(df["created_utc"], unit="s")  # tz-naive UTC
    df["user_id"] = pd.factorize(df["author"])[0]
    df = df.sort_values(["user_id", "post_created"]).reset_index(drop=True)

    out = df[["user_id", "post_created", "cleaned_text", "subreddit"]]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(
        f"Wrote {len(out):,} posts, {out['user_id'].nunique():,} users "
        f"-> {args.out.name}"
    )
    print(f"Median posts/user: {int(out.groupby('user_id').size().median())}")
    print(f"Distinct subreddits: {out['subreddit'].nunique():,}")


if __name__ == "__main__":
    main()
