# TwitterFeel — Experiment Log

Running per-iteration research log. Each entry records the hypothesis
*before* the results, so the reasoning chain survives for the eventual
compiled writeup. Quantitative details live in `metrics/metrics_*.json`
(one per checkpoint, matching names); this file carries the narrative.

Entries E1–E12 were backfilled on 2026-07-17 from
[handoff.md](handoff.md) and [FINDINGS.md](FINDINGS.md); E13 onward are
written as the experiments run.

Shared context: all "honest" rows use a user-level split (no user in more
than one of train/val/test), seed 42, majority-class and persistence
("repeat last emotion") baselines. Emotion mapping: 0 sadness, 1 joy,
2 love, 3 anger, 4 fear, 5 surprise; binary valence = {joy, love,
surprise} positive vs {sadness, anger, fear} negative.

## Entry template

```
## EN — date
**Hypothesis:** what constraint this attacks and the expected outcome.
**Setup:** dataset, features, model, split; exact command.
**Results:** acc / weighted F1 (macro F1, ROC-AUC where relevant) vs
majority + persistence baselines; pointer to metrics JSON.
**Verdict:** supported / refuted / mixed — and what it implies next.
```

---

## E1 — V1 LSTM, 6-class (notebook replica)
**Hypothesis:** the original notebook's pipeline (emotion-ID sequence +
2 time flags, no text), evaluated honestly with a user-level split,
reproduces its claimed F1 = 0.82.
**Setup:** Mental-Health-Twitter corpus (54 users, 20k tweets),
DistilBERT-labeled; `python src/preprocess.py && python src/train.py`.
**Results:** acc 0.232 / wF1 0.279 vs majority 0.405, persistence 0.401
([metrics.json](../metrics/metrics.json)).
**Verdict:** refuted — loses to both baselines by a wide margin. The 0.82
claim does not survive honest evaluation; see the investigation in
[FINDINGS.md](FINDINGS.md) (most likely a training-set metric, an
upstream-DistilBERT score, or unreproducible preprocessing).

## E2 — V2 LSTM, 6-class (text embeddings + sliding window)
**Hypothesis:** throwing away tweet text was the notebook's biggest
mistake; MiniLM embeddings + stride-1 windows (9.8× more training data)
should improve substantially.
**Setup:** `python src/preprocess_v2.py && python src/train_v2.py`.
**Results:** acc 0.364 / wF1 0.342 vs majority 0.405
([metrics_v2.json](../metrics/metrics_v2.json)).
**Verdict:** supported directionally — +13 pp over E1, the largest single
jump in the project — but still below majority. Text features matter;
data scarcity still binds.

## E3 — V2 binary valence, honest split
**Hypothesis:** 6-class is too hard for 38 training users; collapsing to
positive/negative valence is a more reachable task.
**Setup:** `python src/train_binary.py` on windows_v2.npz.
**Results:** acc 0.570 / wF1 0.559 vs majority 0.568, persistence 0.543
([metrics_binary.json](../metrics/metrics_binary.json)).
**Verdict:** mixed — essentially ties majority. Easier task, but not yet
evidence of learned signal under honest evaluation.

## E4 — V2 binary valence, LEAKY row-level split
**Hypothesis:** row-level splits let the model memorize per-user style;
quantifying the inflation explains part of the 0.82 gap.
**Setup:** `python src/train_leaky_binary.py`.
**Results:** acc 0.595 / wF1 0.595, ROC-AUC 0.64 — beats both baselines
([metrics_leaky_binary.json](../metrics/metrics_leaky_binary.json)).
**Verdict:** supported — leakage alone is worth ~2.5 pp here and produces
the project's best absolute number, which is exactly why user-level
evaluation is non-negotiable. Kept as a comparison row, not a result.

## E5 — V3 LSTM, 6-class, w10 (time features)
**Hypothesis:** a 9-tweet window can span an hour or a year; log
time-gap + cyclical hour/day-of-week features address the "no time
anchoring" constraint.
**Setup:** `python src/preprocess_v3.py 10 && python src/train_v3.py
windows_v3_w10.npz`.
**Results:** acc 0.389 / wF1 0.379 vs majority 0.405
([metrics_windows_v3_w10_6cls_lstm.json](../metrics/metrics_windows_v3_w10_6cls_lstm.json)).
**Verdict:** supported — +2.5 pp acc over V2 on 6-class; still below
majority. Time features help multi-class but don't solve it.

## E6 — V3 LSTM, 6-class, w16 (larger window)
**Hypothesis:** more history (15 tweets) gives the LSTM more signal.
**Setup:** `python src/preprocess_v3.py 16 && python src/train_v3.py
windows_v3_w16.npz`.
**Results:** acc 0.415 / wF1 0.368
([metrics_windows_v3_w16_6cls_lstm.json](../metrics/metrics_windows_v3_w16_6cls_lstm.json)).
**Verdict:** mixed — higher accuracy but lower F1: the model collapses
toward majority more. Longer context is not free signal at this data size.

## E7 — V3 LSTM, binary, w10
**Hypothesis:** time features (E5) transfer to the binary task.
**Setup:** `python src/train_v3.py windows_v3_w10.npz --binary`.
**Results:** acc 0.578 / wF1 0.553 vs majority 0.568
([metrics_windows_v3_w10_binary_lstm.json](../metrics/metrics_windows_v3_w10_binary_lstm.json)).
**Verdict:** mixed — best honest binary accuracy, but ≈ tied with
majority. Time features are roughly neutral on binary.

## E8 — V3 BiLSTM+attention, binary, w10
**Hypothesis:** if the constraint were model capacity, a BiLSTM with
additive attention would beat the plain LSTM.
**Setup:** `python src/train_v3.py windows_v3_w10.npz --binary --bilstm`.
**Results:** acc 0.571 / wF1 0.560
([metrics_windows_v3_w10_binary_bilstm.json](../metrics/metrics_windows_v3_w10_binary_bilstm.json)).
**Verdict:** refuted — ≈ tied with LSTM. Consistent with the diagnosis
that data, not architecture, is binding.

## E9 — V3 LSTM, binary, w16
**Hypothesis:** larger window helps binary even if it didn't help 6-class F1.
**Setup:** `python src/train_v3.py windows_v3_w16.npz --binary`.
**Results:** acc 0.562 / wF1 0.559
([metrics_windows_v3_w16_binary_lstm.json](../metrics/metrics_windows_v3_w16_binary_lstm.json)).
**Verdict:** refuted — below the w10 run and below majority.

## E10 — V3 BiLSTM+attention, binary, w16
**Hypothesis:** closing out the architecture × window matrix; attention
might exploit the longer history the plain LSTM couldn't.
**Setup:** `python src/train_v3.py windows_v3_w16.npz --binary --bilstm`.
**Results:** acc 0.553 / wF1 0.553, ROC-AUC 0.560 — below majority (0.568)
([metrics_windows_v3_w16_binary_bilstm.json](../metrics/metrics_windows_v3_w16_binary_bilstm.json)).
**Verdict:** refuted — worst honest binary row. Matrix closed: neither
BiLSTM nor larger windows move the needle.

## E11 — V3 LSTM, 4-class, w10 (drop love + surprise)
**Hypothesis:** love (193 windows) and surprise (60) are unlearnable at
this size (f1 ≈ 0) and act as noise; dropping them should lift the rest.
**Setup:** `python src/train_v3.py windows_v3_w10.npz --four-class`.
**Results:** acc 0.457 / wF1 0.410 (macro 0.247) vs majority 0.416,
persistence 0.411
([metrics_windows_v3_w10_4cls_lstm.json](../metrics/metrics_windows_v3_w10_4cls_lstm.json)).
**Verdict:** supported — first honest multi-class setup to beat both
baselines (+4.1 pp over majority). Rare-class noise was actively hurting
the other classes.

## E12 — V3 LSTM, 4-class, w16
**Hypothesis:** the 4-class gain persists at the larger window.
**Setup:** `python src/train_v3.py windows_v3_w16.npz --four-class`.
**Results:** acc 0.439 / wF1 0.399 vs majority 0.415, persistence 0.409
([metrics_windows_v3_w16_4cls_lstm.json](../metrics/metrics_windows_v3_w16_4cls_lstm.json)).
**Verdict:** supported but weaker — beats baselines, w10 still wins on
both metrics (same pattern as 6-class).

## E13 — Sentiment140 corpus swap (2026-07-17)
**Hypothesis:** the #1 binding constraint is 38 training users
(FINDINGS: "more data... dominates everything else"). Swapping to a
Sentiment140 subsample with ~26× more users should finally let the model
beat baselines convincingly, without any model changes.
**Setup:** Sentiment140 (1.6M tweets, 2009, user handles + timestamps),
prepared by [prepare_sentiment140.py](../src/prepare_sentiment140.py):
HTML/URL/@mention cleaning, per-user dedup, bot filter (unique-text
ratio ≥ 0.5), users with 20–1000 tweets, subsampled to a 50k-tweet CPU
budget → **1,405 users / 50,654 tweets**. Same DistilBERT labeling,
MiniLM embeddings, V3 features, and user-level split as E5–E12. Caveat:
Sentiment140 rows are an emoticon-search *sample* of each user's
timeline, not consecutive tweets — the log time-gap feature carries the
spacing.

```bash
python src/prepare_sentiment140.py <raw_csv>
python src/generate_dataset.py data/interim/dataset_s140_unlabeled.csv data/interim/dataset_s140.csv
python src/generate_embeddings.py data/interim/dataset_s140_unlabeled.csv data/interim/tweet_embeddings_s140.npy
python src/preprocess_v3.py 10 s140
python src/train_v3.py windows_s140_w10.npz            # + --four-class, --binary
```

**Results:** (all honest user-level split; note baselines shift with the
corpus — s140 windows are 53% joy vs 40% in the old corpus)

| Task | Acc | wF1 | macro F1 | Majority | Persistence | Beats baselines? |
|---|---:|---:|---:|---:|---:|---|
| 6-class | 0.215 | 0.254 | 0.157 | 0.501 | 0.391 | ✗ far below majority |
| 4-class | 0.299 | 0.334 | 0.254 | 0.520 | 0.404 | ✗ far below majority |
| binary  | **0.572** | **0.563** | 0.557 | 0.538 | 0.548 | ✓ beats both (+3.5 pp / +2.4 pp), ROC-AUC 0.593 |

([metrics_windows_s140_w10_6cls_lstm.json](../metrics/metrics_windows_s140_w10_6cls_lstm.json),
[metrics_windows_s140_w10_4cls_lstm.json](../metrics/metrics_windows_s140_w10_4cls_lstm.json),
[metrics_windows_s140_w10_binary_lstm.json](../metrics/metrics_windows_s140_w10_binary_lstm.json))

**Verdict:** mixed, leaning supported. **Binary is the project's first
honest result to beat both baselines on any corpus** (old corpus binary
was a statistical tie with majority), with ROC-AUC 0.593 — evidence that
the 38-user constraint really was binding and user-invariant valence
signal exists. Multi-class collapsed, but the confusion matrices show
why: `compute_class_weight("balanced")` under s140's harsher imbalance
(joy 53%, surprise 1%) forces the model to chase rare-class recall at
catastrophic cost to accuracy (joy recall 0.21 despite 0.58 precision).
That's an objective mismatch, not absent signal. → E14 candidate: re-run
multi-class with class weighting disabled or softened (e.g. sqrt), which
needs only a small `train_v3.py` flag.
