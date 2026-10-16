# Refining emotion labels: thresholds and model comparison

**The stronger development result came from changing the model.** ModernBERT
at a shared threshold of 0.3 improved micro F1, macro F1, and mixed-emotion
recovery against RoBERTa at the same threshold. It is now an explicit research
option in the app. Both models were trained on GoEmotions, so these are
exploratory development comparisons, not proof of better labeling on our tweets.
The application default remains RoBERTa with a manual shared threshold.

## Threshold-only result for the original RoBERTa model

**The exported profile improves one aggregate score, but is not a generally
better emotion mapper.** On the published GoEmotions development reference,
the joint-micro objective raises out-of-fold micro F1 from **0.609231 to
0.614223** versus the historical 0.3 threshold. Macro F1 and mixed-emotion
recovery fall. The app therefore keeps manual shared thresholds as the default;
the profile is an explicit, provisional research choice.

These results are from `goemotions_cv_v4`, evaluated on **5,426 development
comments / 5,423 distinct text hashes**. No official test predictions or test
labels enter fitting, scoring, or method selection. This is threshold research
on an existing pretrained model, not new model training or independent accuracy
evidence for tweets/uploads.

## What was compared

Each learned method chooses cutoffs using four folds and applies them to the
remaining fold. Exact duplicate texts stay together, including conflicting
published labels. The five-fold assignment uses seed 42. All original
development rows and annotations are retained; three repeated rows are grouped,
not removed. The same fixed pretrained scores are reused throughout: these are
**out-of-fold threshold decisions**, not independently retrained models.

| Method | How its cutoffs are chosen |
| --- | --- |
| Fixed 0.5 (`fixed_0.5`) | Original application setting; no threshold fitting. |
| Historical 0.3 (`fixed_0.3`) | Previously chosen using the full development set; historical comparator. |
| Fine shared (`fine_shared`) | One training-fold cutoff maximizing micro F1, grid 0.05–0.95 in steps of 0.01. |
| Joint micro per label (`regularized_per_label`) | Training-only coordinate search on joint micro F1, maximum five passes; separate cutoffs only with at least 25 positive and 25 negative training examples. |
| Per-label F1 (`regularized_macro_per_label`) | Training-only F1 maximization separately for each eligible label, with the same grid/support gate. |

Ineligible labels use the training fold's shared cutoff. The code's
`regularized_*` names refer to this support gate; there is **no explicit
shrinkage penalty**. The per-label-F1 strategy was added after inspecting the
initial joint-micro results, so it is an exploratory extension.

## Pooled out-of-fold results

All F1/precision/recall values below are scores on a 0–1 scale, not accuracy
percentages. Macro F1 weights the 28 labels equally; micro F1 pools their label
decisions. All labels have positive support in this development set.

| Method | Micro F1 | Macro F1 | Precision | Recall | Exact set match |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed 0.5 | 0.585922 | 0.447795 | 0.685792 | 0.511442 | 47.44% |
| Historical 0.3 | 0.609231 | 0.496341 | 0.584449 | 0.636207 | 44.89% |
| Fine shared | 0.607499 | 0.494251 | 0.586741 | 0.629781 | 45.28% |
| Joint micro per label | **0.614223** | 0.480234 | 0.604031 | 0.624765 | 45.96% |
| Per-label F1 | 0.605527 | 0.492843 | 0.561038 | 0.657680 | 42.20% |

There are 712 reference posts containing at least two non-neutral emotions.
“All labels recovered” requires every reference label, but allows extra labels;
exact set match above also rejects extras. Automatic acceptance means labeled
or neutral without a review flag, not proven correctness.

| Method | Mixed: all labels recovered | Mixed: label recall | Automatic acceptance | Neutral/emotion conflicts |
| --- | ---: | ---: | ---: | ---: |
| Fixed 0.5 | 9.41% | 35.33% | 82.58% | 29 |
| Historical 0.3 | 19.52% | 47.33% | 84.65% | 736 |
| Fine shared | 18.82% | 46.80% | 85.22% | 691 |
| Joint micro per label | 17.70% | 44.60% | 82.82% | 743 |
| Per-label F1 | 19.80% | 48.13% | 79.41% | 1,046 |

The joint-micro profile reduces false-positive label decisions from 2,886 to
2,613, but increases missed labels from 2,321 to 2,394. Uncertain posts increase
from 97 to 189. Excluding neutral, micro F1 changes only **0.576721 →
0.577521**. Seventeen labels lose F1, seven improve, and four are unchanged.
This is a precision-oriented aggregate tradeoff, with a cost to balanced and
mixed-emotion performance.

The alternative per-label-F1 objective slightly improves mixed recovery, but
does not improve micro or macro F1 over historical 0.3 and creates more conflicts.
**No candidate improves micro F1 while preserving both macro F1 and mixed-label
recall against that comparator.** No profile is automatically promoted.

## Examples of the per-label tradeoff

These are pooled OOF results for the joint-micro profile versus historical 0.3.
Support is the number of positive published labels, not a confidence estimate.

| Label | Support | F1: 0.3 → profile | Recall: 0.3 → profile |
| --- | ---: | ---: | ---: |
| Fear | 90 | 0.6369 → 0.6667 | 55.56% → 68.89% |
| Joy | 172 | 0.5208 → 0.5570 | 54.65% → 48.26% |
| Sadness | 143 | 0.5233 → 0.4865 | 62.94% → 50.35% |
| Caring | 153 | 0.4345 → 0.3664 | 44.44% → 31.37% |
| Confusion | 152 | 0.4367 → 0.3644 | 53.29% → 26.97% |
| Desire | 77 | 0.5679 → 0.4823 | 59.74% → 44.16% |
| Embarrassment | 35 | 0.5818 → 0.5085 | 45.71% → 42.86% |

Higher F1 need not mean more reference emotions found: joy precision improves
from 49.74% to 65.87% while recall falls. Fear makes the opposite precision/
recall exchange: precision falls from 74.63% to 64.58%, with greater recall.
Inspect each important label rather than treating the aggregate as universal.

Grief (13 positives), pride (15), relief (18), and nervousness (21) never meet
the training-support gate. Their thresholds fall back in every fold. Some other
cutoffs are unstable: caring ranges 0.18–0.77 across folds; sadness 0.24–0.73.
Neither the support gate nor a final cutoff makes those distinctions reliable.

## Uncertainty and limits

The paired 95% percentile interval for joint-micro F1 minus historical 0.3 is
**+0.000685 to +0.010064**, using 500 paired resamples of exact-text groups,
seed 42. The same resampled groups are used for each method. The proportion of
positive bootstrap differences is not a p-value.

This interval holds the fitted OOF decisions fixed. It omits threshold-fitting,
method-selection, annotation, and upstream-model uncertainty. The model was
trained on GoEmotions, and its development data may already have influenced
upstream checkpoint selection. Historical 0.3 was selected using this entire
development set before the experiment. Choosing a winning strategy after
inspecting these OOF results adds further selection optimism. **A positive
interval is not independent confirmation of improved unseen-data accuracy.**

Author IDs are unavailable; grouping exact duplicate text cannot capture
dependence between different comments by one person or near duplicates. There
are also 174 published neutral-plus-emotion reference rows. We retain them
unchanged, although the app's own human rubric requires neutral alone. A method
can score better under the published annotations while increasing the app's
manual-review burden.

The official test split remains reserved and unscored by this experiment.
Independent human review of the user's target data is still required. The
first 30 development practice posts establish rubric understanding, not an
independent accuracy benchmark; a second reviewer can follow later.

## Trying the provisional profile in the app

In **Run analysis → Emotion threshold settings**, explicitly choose a saved
provisional research profile instead of **Manual shared threshold**. The app
does not select it automatically or replace the default threshold. You can
inspect every saved cutoff before running it on an imported dataset version.

The exported profile is refitted on **all development rows after** the OOF
method comparison. Its performance on those same refit rows is not the OOF
estimate above. The app checks that the profile's emotion-model provenance
matches the installed files and refuses a mismatch. Each new analysis copies
the selected profile identity/hash, exact cutoffs, and provenance into its
configuration. Model support scores themselves remain unchanged; this is
decision-threshold selection, not probability calibration.

Only the latest complete, integrity-checked profile for a given reference/model
pair appears in the selector. Earlier experiment files remain available on
disk. Old analysis results retain their original choices. The shared-threshold
Model lab sandbox is a separate alternative rule; its slider does not shift
each learned cutoff or overwrite saved profile decisions.

## Reproduction and artifacts

The recorded corpus revision is
`7951440944924ac61b6e2f9a2a3c715834005c40`. The tested emotion model revision is
`d75048347613a25d77de8cf6412eaae9fa7b26be`. Existing local source scores can be
reused; threshold refinement performs no model inference:

```powershell
$thresholdReference = "runs/public_benchmarks/goemotions/7951440944924ac61b6e2f9a2a3c715834005c40/official-v1/reference.csv"
$thresholdScores = "runs/public_benchmarks/goemotions/7951440944924ac61b6e2f9a2a3c715834005c40/official-v1/development_run_81e15771fab3/predictions.jsonl"
.venv-workbench/Scripts/python.exe src/threshold_tuning.py --reference $thresholdReference --predictions $thresholdScores --out runs/workbench/refinements/reproduction_01
```

Use a new output directory. On a fresh checkout, first prepare/score the pinned
public development reference with `src/public_benchmark.py --evaluate
--revision 7951440944924ac61b6e2f9a2a3c715834005c40`; use the returned run's
predictions path rather than the local example above. This scores development,
not official test. Both commands require the dependencies/models described in
[WORKBENCH.md](WORKBENCH.md).

The full local experiment is `runs/workbench/refinements/goemotions_cv_v4/`:
`report.json` contains the pooled metrics, fold thresholds, support/stability,
paired intervals and caveats; `oof_predictions.jsonl` records every development
decision and fold; `calibration.json` is the separate all-development profile.
File hashes bind the report, reference, source scores/config, and exported
profile. These large/local artifacts are gitignored.

## Alternative model check

The completed comparison used
[`cirimus/modernbert-base-go-emotions`](https://huggingface.co/cirimus/modernbert-base-go-emotions),
pinned at `cb5ed17bc26c8cd1f017fabe700a5d723f825287`. All 5,426 development
posts, 5,423 text groups, reference labels, and fold assignments match the
original-model experiment. Both models score the original text with a 256-token
limit; their tokenizers may retain different amounts of text within that limit.
The candidate runs locally on CPU using safetensors, without downloaded code.
Scoring took approximately 86 seconds in this environment.

| Model and rule | Micro F1 | Macro F1 | Precision | Recall | Exact set match | Mixed: all reference labels found |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RoBERTa, shared 0.3 | 0.6092 | 0.4963 | 0.5844 | 0.6362 | 44.90% | 19.52% |
| ModernBERT, shared 0.3 | **0.6195** | **0.5378** | 0.5796 | **0.6654** | 43.86% | **24.44%** |
| RoBERTa, training-fold shared cutoff | 0.6075 | 0.4943 | 0.5867 | 0.6298 | 45.28% | 18.82% |
| ModernBERT, training-fold shared cutoff | **0.6192** | **0.5358** | **0.5928** | **0.6481** | 44.95% | **22.05%** |
| ModernBERT, joint micro per-label cutoffs | 0.6211 | 0.5247 | 0.6178 | 0.6244 | 46.66% | 18.82% |

For shared 0.3, the paired micro-F1 difference is +0.01030, with a conditional
95% text-group bootstrap interval of +0.00294 to +0.01708. For separately fitted
shared cutoffs, the difference is +0.01170, interval +0.00442 to +0.01890.
Selecting this candidate after earlier experiments still adds selection bias.
Neither interval establishes improved accuracy on independent target data.

At shared 0.3, ModernBERT improves overall recall and mixed-label recall
(47.33% to 50.47%), but precision and exact-match decline. Neutral/emotion
review flags increase from 736 to 946. Its highest micro-F1 profile sacrifices
some of the mixed-emotion benefit. For examining mixed emotions, **ModernBERT
with a manual shared 0.3 cutoff is the more relevant research comparison**;
the slightly higher-F1 per-label profile answers a different tradeoff.

In **Run analysis**, choose **ModernBERT · research candidate** under **Emotion
model**. Choose **Manual shared threshold**, then set 0.30 to reproduce that
rule on a new dataset. Alternatively inspect and explicitly select the matching
ModernBERT research profile. Switching models clears the old profile selection;
a profile whose model fingerprint differs is refused. Each run records the
actual model and copied settings. The sentiment model remains independent.

The complete paired report is under
`runs/model_comparisons/20261015T230413176038Z_df0b23f5/comparison.json`, with
the candidate scores and grouped refinement in its `modernbert/` directory.
A tracked summary is
[`metrics/emotion_model_comparison_v1.json`](../metrics/emotion_model_comparison_v1.json).
No official test examples were scored.

To reproduce with a new output directory generated by the comparison runner:

```powershell
.venv-workbench/Scripts/python.exe src/compare_models.py --run --reference $thresholdReference --baseline runs/workbench/refinements/goemotions_cv_v4
```

The pinned candidate is downloaded only if missing. The author's card identifies
this model as superseded; this is a bounded test of one available candidate,
not a claim that it is the best available model. Training overlap, annotation
policy differences, and the need for an independent target-data reference remain.
