"""Aggregate population-level statistics over sadness_flag.py's outputs.

Reads twitter_feel_plus/results/per_user_sadness.csv and flag_windows.csv,
writes:
  results/aggregate.json   -- machine-readable summary
  results/REPORT.md        -- human-readable summary (anonymized)

Does NOT name individual users in the markdown report.
"""
from __future__ import annotations

import json
from textwrap import dedent

import numpy as np
import pandas as pd

from paths import RESULTS_DIR

PER_USER_CSV = RESULTS_DIR / "per_user_sadness.csv"
FLAG_CSV = RESULTS_DIR / "flag_windows.csv"
PARAMS_JSON = RESULTS_DIR / "params.json"

AGG_OUT = RESULTS_DIR / "aggregate.json"
REPORT_OUT = RESULTS_DIR / "REPORT.md"


def percentiles(s: pd.Series, qs=(0.25, 0.5, 0.75, 0.95)) -> dict:
    if s.empty:
        return {f"p{int(q * 100)}": None for q in qs}
    return {f"p{int(q * 100)}": float(s.quantile(q)) for q in qs}


def main() -> None:
    per_user = pd.read_csv(PER_USER_CSV)
    flag = pd.read_csv(FLAG_CSV)
    params = json.loads(PARAMS_JSON.read_text()) if PARAMS_JSON.exists() else {}

    n_users = len(per_user)
    users_flagged = int((per_user["num_flag_periods"] > 0).sum())
    pct_flagged = users_flagged / n_users * 100 if n_users else 0.0

    baseline_stats = {
        "mean": float(per_user["baseline_sad_frac"].mean()),
        "median": float(per_user["baseline_sad_frac"].median()),
        "min": float(per_user["baseline_sad_frac"].min()),
        "max": float(per_user["baseline_sad_frac"].max()),
        **percentiles(per_user["baseline_sad_frac"]),
    }

    flag_period_count_stats = {
        "mean_per_user": float(per_user["num_flag_periods"].mean()),
        "max_per_user": int(per_user["num_flag_periods"].max()),
        **percentiles(per_user["num_flag_periods"].astype(float)),
    }

    if not flag.empty:
        duration_stats = {
            "mean_days": float(flag["duration_days"].mean()),
            "median_days": float(flag["duration_days"].median()),
            "max_days": float(flag["duration_days"].max()),
            **{
                k.replace("p", "p") + "_days": v
                for k, v in percentiles(flag["duration_days"]).items()
            },
        }
        excess_stats = {
            "mean_excess_over_baseline": float(flag["excess_over_baseline"].mean()),
            "median_excess_over_baseline": float(flag["excess_over_baseline"].median()),
            "max_excess_over_baseline": float(flag["excess_over_baseline"].max()),
        }
        peak_stats = {
            "mean_peak_window_sad_frac": float(flag["peak_window_sad_frac"].mean()),
            "median_peak_window_sad_frac": float(flag["peak_window_sad_frac"].median()),
            "max_peak_window_sad_frac": float(flag["peak_window_sad_frac"].max()),
        }
    else:
        duration_stats = excess_stats = peak_stats = {}

    aggregate = {
        "params": params,
        "n_users": n_users,
        "users_with_at_least_one_flag": users_flagged,
        "pct_users_flagged": round(pct_flagged, 2),
        "n_flag_periods_total": int(per_user["num_flag_periods"].sum()),
        "baseline_sadness_fraction_per_user": baseline_stats,
        "flag_periods_per_user": flag_period_count_stats,
        "flag_period_duration": duration_stats,
        "flag_period_peak_sadness": peak_stats,
        "flag_period_excess_over_baseline": excess_stats,
    }
    AGG_OUT.write_text(json.dumps(aggregate, indent=2))

    # Human-readable markdown report.
    md = dedent(
        f"""\
        # TwitterFeel Plus — Aggregate Sadness-Signal Report

        > **This is a population-level signal-detection report, not a clinical or
        > diagnostic tool. See twitter_feel_plus/README.md for the full list of
        > methodological caveats.** Sadness in this dataset is the DistilBERT
        > model's `sadness` class on the tweet text — not a clinical assessment.

        ## Parameters

        - rolling tweet-window size: **{params.get('tweet_window', '?')}** tweets
        - rolling day-window size: **{params.get('day_window_days', '?')}** days
        - absolute flag threshold: rolling sadness fraction > **{params.get('abs_threshold', '?')}**
        - personal-baseline threshold: also > baseline + **{params.get('dev_threshold', '?')}**
        - min sustained windows to form a flag period: **{params.get('min_sustained', '?')}**

        ## Population summary

        - users analyzed: **{n_users}**
        - users with at least one flagged sustained-sadness period: **{users_flagged}** ({pct_flagged:.1f}%)
        - total flag periods detected (across all users): **{aggregate['n_flag_periods_total']}**

        ## Per-user baseline sadness fraction

        Each user's long-run share of tweets DistilBERT labeled `sadness`.

        - mean: **{baseline_stats['mean']:.3f}** | median: **{baseline_stats['median']:.3f}**
        - min: **{baseline_stats['min']:.3f}** | max: **{baseline_stats['max']:.3f}**
        - 25/50/75/95 percentiles: **{baseline_stats['p25']:.3f} / {baseline_stats['p50']:.3f} / {baseline_stats['p75']:.3f} / {baseline_stats['p95']:.3f}**
        """
    )
    if duration_stats:
        md += dedent(
            f"""
            ## Flag-period statistics

            - mean duration: **{duration_stats['mean_days']:.1f} days** | median: **{duration_stats['median_days']:.1f} days** | max: **{duration_stats['max_days']:.1f} days**
            - mean peak rolling sadness fraction: **{peak_stats['mean_peak_window_sad_frac']:.3f}** (max **{peak_stats['max_peak_window_sad_frac']:.3f}**)
            - mean excess over user's personal baseline: **{excess_stats['mean_excess_over_baseline']:.3f}** (max **{excess_stats['max_excess_over_baseline']:.3f}**)
            """
        )
    else:
        md += "\n## Flag-period statistics\n\nNo flag periods met the criteria under these parameters.\n"

    md += dedent(
        f"""
        ## Files

        - `per_user_sadness.csv` — one anonymized row per user with summary stats
        - `flag_windows.csv` — one row per flag period (anonymized)
        - `params.json` / `aggregate.json` — exact parameters and machine-readable summary
        - `REPORT.md` — this file

        User IDs in the outputs are sequential pseudonyms (`user_001`, `user_002`, ...).
        The mapping to the real Twitter user IDs in the source CSV is not saved.
        """
    )

    REPORT_OUT.write_text(md, encoding="utf-8")

    print(f"Wrote {AGG_OUT.name}")
    print(f"Wrote {REPORT_OUT.name}")
    print(
        f"\n{users_flagged}/{n_users} users ({pct_flagged:.1f}%) had at least one "
        f"flagged sustained-sadness period."
    )


if __name__ == "__main__":
    main()
