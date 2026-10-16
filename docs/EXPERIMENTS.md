# Historical forecasting experiment log

This is the record of the earlier next-post prediction experiments. Current
work labels existing posts; start with the [README](../README.md),
[annotation guide](EMOTION_LABELING.md), and [model checks](LABELING_CHECK.md).
Forecasting scores below do not measure existing-post labeling quality.

Quantitative details live in `metrics/metrics_*.json`; this file retains the
experiment narrative. Entries E1–E12 were reconstructed from early notes on
2026-07-17. Later entries state their setup and findings; E15 explicitly
notes that it was recorded after implementation rather than preregistered.

V4 development starts at E15; its cleaned data and grouped cross-validation
protocol differ from E1–E14. See [V4.md](V4.md) to reproduce that workflow.

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
claim was not reproduced by these experiments. The provenance of the
original number remains unresolved; it must not be used as a verified result.

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
(earlier analysis identified the limited number of users as a constraint). Swapping to a
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

## E14 — Class-weight softening, multi-class, original corpus (2026-07-28)
**Hypothesis:** E13 argued that `compute_class_weight("balanced")` forces
the model to chase rare-class recall at catastrophic cost to accuracy.
Softening the weights (`sqrt`) or removing them (`none`) should recover
accuracy and weighted F1 on multi-class.
**Setup:** new `--class-weight {balanced,sqrt,none}` flag on
[train_v3.py](../src/train_v3.py); non-default modes get a `_cw{mode}`
filename suffix so E1–E13 artifacts are never overwritten. Run on the
**original** Mental-Health corpus (`windows_v3_w10.npz`), not s140 — the
s140 intermediates are gitignored and the raw Sentiment140 CSV is not on
disk, so the s140 version of this experiment is still pending.

```bash
python src/train_v3.py windows_v3_w10.npz --class-weight sqrt   # + --four-class
python src/train_v3.py windows_v3_w10.npz --class-weight none   # + --four-class
```

**Results:** (honest user-level split; majority 0.405 for 6-class,
0.416 for 4-class)

| Task | Weighting | Acc | wF1 | macro F1 |
|---|---|---:|---:|---:|
| 6-class | balanced (E5) | 0.3887 | 0.3791 | 0.1725 |
| 6-class | **sqrt** | **0.4612** | **0.4097** | 0.1738 |
| 6-class | none | 0.4338 | 0.3129 | 0.1259 |
| 4-class | balanced (E11) | **0.4572** | **0.4099** | **0.2474** |
| 4-class | sqrt | 0.4284 | 0.3036 | 0.1790 |
| 4-class | none | 0.4237 | 0.2679 | 0.1597 |

**Verdict:** mixed — supported for 6-class, refuted for 4-class.

*6-class:* `sqrt` is the first honest 6-class setup to beat majority
(0.4612 vs 0.4050, **+5.6 pp**), overturning the long-standing "6-class
trails majority" result from E5. But the per-class report shows the gain
is **not** rare-class learning: it comes entirely from the model
committing harder to the two dominant classes (anger recall 0.63→0.76,
joy 0.24→0.30, together 85% of test rows), while sadness (f1 0.048→0.008)
and fear (0.093→0.026) actively degrade. Macro F1 looks flat only because
`love` landed 2 lucky predictions at precision 1.00 on 61 support — that
is noise, not signal. Caveat on the margin: the majority baseline is the
*train*-derived class (joy, 0.4050 on test); the best constant predictor
in hindsight is anger at 0.4456, over which the model gains only +1.6 pp.

*4-class:* softening clearly hurts (wF1 0.410→0.304). With love/surprise
already dropped the residual imbalance is mild, and `balanced` was doing
useful work rather than distorting the objective. E11 stands as the best
4-class configuration.

**Implication:** the "balanced weighting is the problem" story from E13 is
only true when the imbalance is extreme (a 1%-frequency class). It is a
per-task knob, not a global fix — and on 6-class it buys headline metrics
by abandoning the rare classes rather than by learning them. The s140
re-run (the original E14 proposal, where imbalance is harsher: joy 53%,
surprise 1%) is the experiment that would actually test E13's claim, and
still needs the Sentiment140 raw file.

## E15 — V4 evaluation and history-feature comparison (2026-09-22)

**Question:** do explicit historical emotion features help a small LSTM,
and does it beat simple history-based predictors under grouped validation?
This entry was recorded after implementation and verification, not preregistered.

**Setup:** cleaned original MHT corpus; 11,801 windows / 53 users. Removed
204 identical user/time/text records, skipped 1,068 timestamp-tied windows.
45 development users / 10,336 windows enter five user-disjoint folds; eight
reserved test users are **not scored**. Source hashes, configurations, aggregate
scores and intervals are in
[metrics_v4_mht_binary_development.json](../metrics/metrics_v4_mht_binary_development.json).
Detailed per-user predictions/splits remain local in ignored `runs/`.

Existing labels use the explicit **one-hot history fallback**, not newly
generated soft probabilities. Existing embedding order is unverified by a
manifest and was explicitly accepted. Both LSTM variants use 64 units,
training-only normalization, no class weights, 10 fixed epochs and seed 42.
This is an initial comparison, not a tuned V4 model or a V3 architecture replica.

```powershell
python src/preprocess_v4.py data/interim/dataset_godknowswhat.csv data/interim/tweet_embeddings.npy data/interim/windows_v4_mht_w10.npz --allow-unverified-embeddings
python src/train_v4.py data/interim/windows_v4_mht_w10.npz --out runs/v4_mht_binary_emotion --task binary --models majority persistence frequency transition logistic lstm --features text-time-emotion --epochs 10
python src/train_v4.py data/interim/windows_v4_mht_w10.npz --out runs/v4_mht_binary_text --task binary --models lstm --features text-time --epochs 10
```

**Development results:**

| Model | Accuracy | Macro F1 | Mean user accuracy |
|---|---:|---:|---:|
| Training-majority baseline | 0.4709 | 0.4264 | 0.4637 |
| Persistence | 0.5600 | 0.5591 | 0.5489 |
| Historical frequency | **0.5833** | **0.5816** | **0.5846** |
| Transition | 0.5600 | 0.5591 | 0.5489 |
| Logistic, text/time/emotion | 0.5190 | 0.5188 | 0.5154 |
| LSTM, text/time | 0.5146 | 0.5142 | 0.4991 |
| LSTM, text/time/emotion | 0.5038 | 0.5036 | 0.4978 |

The majority class is learned separately in each fold; it need not be the same
class across folds. Frequency beats persistence by 2.33 percentage points;
the paired user-cluster bootstrap interval is +1.09 to +3.77 points. This is
development evidence, conditional on the models and sampled users, not a
new held-out test result.

**Verdict:** explicit one-hot historical emotions did not improve this LSTM
configuration, and both learned models trail the frequency baseline. The
useful outcome is a stronger baseline and an auditable evaluation workflow.
Next experiments should use development-only tuning and actual probability
features before considering larger architectures. Do not compare these
numbers directly with V3 test metrics, or treat this as proof that history
features cannot help under other training settings.

**Verification:** 16 regression/integration tests pass; synthetic LSTM train,
save and reload reproduce probabilities. Real MHT preprocessing, both five-fold
LSTM runs and all baseline runs completed. Test scoring was exercised only on
synthetic data. No real Reddit dump or regenerated Sentiment140 corpus was run.
