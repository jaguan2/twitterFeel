# TwitterFeel Plus — Aggregate Sadness-Signal Report

> **This is a population-level signal-detection report, not a clinical or
> diagnostic tool. See twitter_feel_plus/README.md for the full list of
> methodological caveats.** Sadness in this dataset is the DistilBERT
> model's `sadness` class on the tweet text — not a clinical assessment.

## Parameters

- rolling tweet-window size: **20** tweets
- rolling day-window size: **14** days
- absolute flag threshold: rolling sadness fraction > **0.4**
- personal-baseline threshold: also > baseline + **0.15**
- min sustained windows to form a flag period: **3**

## Population summary

- users analyzed: **72**
- users with at least one flagged sustained-sadness period: **5** (6.9%)
- total flag periods detected (across all users): **24**

## Per-user baseline sadness fraction

Each user's long-run share of tweets DistilBERT labeled `sadness`.

- mean: **0.144** | median: **0.082**
- min: **0.000** | max: **1.000**
- 25/50/75/95 percentiles: **0.044 / 0.082 / 0.112 / 0.561**

## Flag-period statistics

- mean duration: **1.0 days** | median: **1.0 days** | max: **2.2 days**
- mean peak rolling sadness fraction: **0.671** (max **0.950**)
- mean excess over user's personal baseline: **0.268** (max **0.421**)

## Files

- `per_user_sadness.csv` — one anonymized row per user with summary stats
- `flag_windows.csv` — one row per flag period (anonymized)
- `params.json` / `aggregate.json` — exact parameters and machine-readable summary
- `REPORT.md` — this file

User IDs in the outputs are sequential pseudonyms (`user_001`, `user_002`, ...).
The mapping to the real Twitter user IDs in the source CSV is not saved.
