# TwitterFeel Plus — Sustained Sadness Signal Detector

A research-artifact pipeline that scans the DistilBERT-labeled tweet corpus
([data/interim/dataset_godknowswhat.csv](../data/interim/dataset_godknowswhat.csv))
for users whose share of `sadness`-labeled tweets remains elevated over a
sustained period.

## What this is

A method for detecting, in a public tweet corpus, sequences where a user's
share of text classified as `sadness` exceeds both a fixed threshold and the
user's own long-run baseline for a sustained number of consecutive rolling
windows.

It produces:

- a per-user CSV with summary statistics (anonymized)
- a CSV of detected flag periods (anonymized)
- a population-level aggregate report

## What this is NOT

**This is not a depression detector. It is not a diagnostic tool. It is not a
recommendation for any clinical intervention.** Specifically:

- DistilBERT's `sadness` class fires on a great deal of text that isn't
  expressing personal sadness — sarcasm, song lyrics, sympathy for others,
  references to fictional events, jokes ("I'm dying"). Sustained "sadness" in
  this signal is partly a *writing style*, not always a *feeling*.
- Clinical depression is defined by DSM-5 criteria including anhedonia, sleep
  and appetite changes, concentration loss, and suicidal ideation. Tweets do
  not reliably surface most of these even when they are present.
- The training data behind DistilBERT (`dair-ai/emotion`) is short, English,
  often non-clinical text. Generalizing its outputs to clinical interpretation
  is unsupported.
- The corpus has 54 retained users and was scraped publicly — none of them
  opted in to any mental-health screening.
- Authors of this artifact are not clinically qualified to make mental-health
  claims, and this work makes none.

The intended use is **methodological research** — studying whether
text-derived emotion signals show statistically detectable sustained patterns
in public social-media data. Not screening, not flagging individuals, not
intervention.

## Outputs are anonymized

The CSV and report outputs replace each real Twitter `user_id` with a
sequential pseudonym (`user_001`, `user_002`, ...). The mapping back to
the real ID is never saved. The aggregate report contains no individual
identifiers at all.

## How the signal works

For each user, sorted by `post_created`:

1. Compute the user's **long-run baseline sadness fraction** (share of all of
   their tweets where DistilBERT predicted `sadness`).
2. Compute a **rolling sadness fraction** over the last *N* tweets (default
   N = 20). Also compute a parallel rolling fraction over the last *D* days
   (default D = 14) for robustness checking.
3. Mark a window as **flagged** if its rolling fraction:
   - exceeds an **absolute threshold** (default 0.40), **AND**
   - exceeds the user's **personal baseline** by at least the **deviation
     threshold** (default +0.15).
4. A **flag period** is a run of ≥ `MIN_SUSTAINED` consecutive flagged windows
   (default 3). Two stacked conditions and a minimum-sustained run together
   reduce false positives from one-off bursts of sadness-coded tweets.

All thresholds and window sizes are top-of-file constants in
[sadness_flag.py](sadness_flag.py) — easy to tune for sensitivity analysis.

## Running it

```bash
# From the repo root
python twitter_feel_plus/sadness_flag.py    # ~5 s; writes per_user_sadness.csv, flag_windows.csv, params.json
python twitter_feel_plus/sadness_report.py  # ~1 s; writes aggregate.json, REPORT.md
```

All outputs land in [results/](results/).

## Files

```
twitter_feel_plus/
  README.md         this file (methodology + caveats)
  paths.py          shared path constants
  sadness_flag.py   per-user rolling sadness + flag-period detection
  sadness_report.py aggregate (anonymized) population report
  results/
    per_user_sadness.csv   one row per user (anonymized)
    flag_windows.csv       one row per detected flag period (anonymized)
    params.json            exact run parameters
    aggregate.json         population-level statistics
    REPORT.md              human-readable summary
```

## A note on responsible use

If this method were ever extended toward an actual mental-health intervention,
the required components would include: explicit informed consent from each
monitored user; clinical professionals in the loop on any "flag"; an
escalation path that respects user autonomy; and validation against clinical
ground truth, not against another text classifier. None of those are present
here, and the artifact in this folder should not be deployed as anything
beyond a research signal-detection method.
