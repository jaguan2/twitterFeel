# Historical Reddit importer and forecasting workflow

This reference documents the earlier corpus importer and next-post experiments.
For current existing-post labeling, start with the [README](../README.md) and
[annotation guide](EMOTION_LABELING.md). The importer remains available, but
the commands below use the historical six-class labeling and forecasting path.

Extending the next-post emotion task from Twitter to Reddit. The modeling
pipeline is unchanged: [prepare_reddit.py](../src/prepare_reddit.py) adapts a
Reddit dump into the three columns everything downstream already expects, and
the existing `DATASET_TAG` mechanism carries it the rest of the way.

## Why Reddit (and not Instagram)

- **Instagram is blocked at acquisition.** Its Graph API only exposes accounts
  you own or business accounts that authorize you; there is no sanctioned path
  to an arbitrary user's caption timeline, which is what this task needs.
- **Instagram is the wrong modality.** The entire feature stack is a 384-d
  sentence embedding. On Instagram the signal is the image — captions are
  short, emoji-dense, and often promotional. Worth revisiting only under a
  multimodal redesign (e.g. CLIP image embeddings), which is a different
  project.
- **Reddit can provide denser histories.** Sentiment140 is an emoticon-search
  sample. Reddit dumps avoid that particular sampling mechanism, but length,
  content and subreddit filters still remove intervening comments. The actual
  task is "next retained comment", not guaranteed consecutive activity.

## Getting a dump

The standard source is the **Pushshift monthly archives** — newline-delimited
JSON, one comment (`RC_YYYY-MM`) or submission (`RS_YYYY-MM`) per line, from
~2005 onward. Each comment record carries exactly what the pipeline needs:

| Dump field | Pipeline column |
|---|---|
| `author` (factorized) | `user_id` |
| `created_utc` | `post_created` |
| `body` (cleaned) | `cleaned_text` |
| `subreddit` | kept for the confound check below |

**You only need one month.** The s140 run used 50k posts / 1,405 users; a
single monthly comment dump clears that by orders of magnitude after
filtering. Use `--max-records` to stop reading early rather than decompressing
the whole file.

> **Verify current hosting before planning around it.** Pushshift's live API
> was restricted after Reddit's 2023 API changes and the canonical download
> location for the historical dumps has moved since (Academic Torrents
> mirrors, successor archival projects). The dumps themselves are widely
> mirrored; it is the front door that keeps changing. This note was written
> 2026-07-28 and is not a substitute for checking.

**Lower-effort alternative:** ConvoKit (Cornell) ships Reddit corpora with
speaker and timestamp exposed through a pip-installable Python API — smaller
and more curated, good for a first end-to-end pass before committing to a
multi-GB dump.

## Running it

The commands below reproduce the V3 route. To reproduce V4 modeling, use
[V4.md](V4.md) after labeling and embedding the imported CSV. The importer
now stages text in temporary SQLite storage before sampling authors;
`--budget` bounds final output approximately, not temporary disk use.

```bash
pip install -r requirements-forecasting.txt  # zstandard is needed for .zst dumps

python src/prepare_reddit.py RC_2019-01.zst
python src/generate_dataset.py    data/interim/dataset_reddit_unlabeled.csv \
                                  data/interim/dataset_reddit.csv
python src/generate_embeddings.py data/interim/dataset_reddit_unlabeled.csv \
                                  data/interim/tweet_embeddings_reddit.npy
python src/preprocess_v3.py 10 reddit
python src/train_v3.py windows_reddit_w10.npz --binary
```

Useful flags on `prepare_reddit.py`: `--budget` (target post count),
`--min-words/--max-words` (keep comments tweet-shaped), `--max-records` (cap
the streaming read), `--exclude-subreddits`.

## Three caveats that affect how results should be read

1. **Labeler domain shift.** `generate_dataset.py` runs a DistilBERT trained
   on `dair-ai/emotion` — short, tweet-shaped text — truncated at 128 tokens.
   Reddit comments are longer and differently registered, so labels are
   potentially noisier here than on Twitter. `--max-words 60` narrows
   the gap; it does not close it. A Reddit-domain labeler is a possible
   experiment, not an established fix. GoEmotions would require an explicit
   mapping policy for its broader, multilabel taxonomy and neutral examples;
   it does not map directly onto this six-class task. Labels would still be
   model-generated and need independent quality checks on the target corpus.
2. **Subreddit confound.** A comment in r/aww gets labeled joy and one in
   r/depression sadness largely regardless of the author's trajectory, so the
   model can learn *which subreddit* instead of *this user's emotional
   dynamics* — the same flavor of shortcut as the row-level leak ruled out in
   E4. `subreddit` is emitted so this can be tested both ways; consider
   `--exclude-subreddits` for venue-dominated subs.
3. **UTC timestamps.** Reddit stores `created_utc` and dumps carry no user
   timezone. The `overnight` and cyclical-hour features in `preprocess_v3.py`
   assume *local* time, so across a global userbase they are substantially
   weaker here than on the (entirely PDT) Sentiment140 corpus. Expect the
   7 time features to contribute less than they did on Twitter.

## Ethics

Reddit hosts real mental-health support communities. Pulling per-author
histories from them and flagging sustained sadness is a materially more
sensitive act than the same analysis on a 2009 tweet archive. Decide the
sampling policy *before* collecting, keep the anonymization discipline already
used in [twitter_feel_plus/sadness_report.py](../twitter_feel_plus/sadness_report.py)
(no named users in any report), and keep the same non-clinical framing the
rest of the project uses.

The gated research corpora built for exactly this purpose — RSDD, SMHD, and
the eRisk/CLEF collections — require signed data-use agreements and carry
clinical scope the project has deliberately stepped away from. Out of scope
unless that decision is reversed.

## Verification status

Verified on a synthetic Pushshift-shaped fixture (2026-07-28): markup
cleaning (quote lines, markdown links, inline code, URLs, `u/` mentions),
bot/deleted-author filtering, unique-ratio spam filtering, submission
(`title`+`selftext`) handling, `.zst` streaming (byte-identical to plain
text), and the full `generate_embeddings.py` → `preprocess_v3.py 10 <tag>`
chain, which produced a correct `X=(97, 9, 391)` window tensor.

**Not yet run on real Reddit data** — no dump is on disk, and
`generate_dataset.py`'s DistilBERT labeler was stubbed for the fixture test
rather than downloading 250 MB to label synthetic sentences. The first real
run should sanity-check the emotion distribution before training on it: a
degenerate distribution is the signature of caveat #1 above.
