# TwitterFeel

## Study progression and findings

**Research record | Edition 1 | Compiled 2026-10-21**

This record follows the study from its original forecasting question to the
current existing-post emotion and sentiment workbench. It brings together
findings, limitations, and changes in interpretation that were spread across
the repository. It is a retrospective synthesis of saved evidence, not a new
experiment or an independently reviewed publication.

### How to read the dates and evidence

Dates are shown in America/New_York where a timestamp is available. A recorded
date comes from an explicit experiment entry, report, or artifact timestamp.
An archive date comes from Git and does not establish when an experiment ran.
Undated original work is placed first for context, without assigning it a year.
E1-E12 were reconstructed on 2026-07-17; their execution dates are unknown.
All entries in this edition were backfilled on 2026-10-21.

Results marked **reported** come from saved documents or artifacts; they were
not rerun for this record. Checks marked **reproduced** were performed during
the 2026-10-21 review. A **hypothesis** or **planned** item remains unresolved.
Later corrections are dated explicitly rather than silently changing the past.

### Progression at a glance

| Date or date basis | Change in the study |
| --- | --- |
| Original work, undated | Proposal moves from mental-health risk to next-tweet emotion prediction |
| 2026-05-26, Git archive | Sustained sadness-label pattern analysis is preserved |
| 2026-07-17, reconstructed | E1-E12 explore text, time, task scope and model architecture |
| 2026-07-17 and 2026-07-28 | More users and different class weighting yield limited gains |
| 2026-09-22 | V4 introduces stronger baselines and auditable grouped evaluation |
| 2026-10-15 | Existing-post labeling, public references, thresholds and model comparison |
| 2026-10-16 | Execution checks expose concrete labeling weaknesses |
| 2026-10-21 | Historical audit confirms saved-code defects and evaluation uncertainty |

Scores from different tasks are not directly comparable. Six-class forecasting
accuracy, binary forecasting accuracy, and multi-label micro/macro F1 have
different meanings. Source IDs such as [S01] refer to the final source register.

<!-- pagebreak -->

## Original study | Date not established

**Evidence: historical documents; reported.**

### Question and method changed during the original project

The proposal asks whether social-media patterns can predict future mental-health
conditions. Its proposed method is TF-IDF with XGBoost on four categories of
existing tweets. Such a classification experiment alone would not evaluate
future onset of a condition. The final report instead asks how to predict the
emotion of a user's next tweet using their tweet history. [S01, S02]

The final workflow describes roughly 20,000 tweets, six transformer-generated
emotion labels, weekend and overnight features, batches of ten posts from one
user, an LSTM, and SMOTE. The saved preprocessing uses the first nine posts as
history and the tenth emotion as the target. Users with fewer than 20 or more
than 1,000 posts are excluded in the filtered variation. [S02, S03]

### Finding as originally reported

The filtered-data/SMOTE variation reports approximately 0.82 accuracy, precision,
recall and F1. A later audit, dated 2026-10-21, recalculates its matrix as
0.8233 accuracy and 0.8182 weighted F1. The original report acknowledges that
oversampling increased prediction counts, but does not establish the split or
resampling order. The source implementation of that training run is absent
from the inspected history. These are reported results with unresolved
evaluation provenance. [S02, S03]

### What this changed

Subsequent repository experiments pursued reproduction of the high forecasting
score. The research target was a classifier-generated emotion label, rather
than an independently observed emotional state or clinical outcome. [S04, S05]

## 2026-05-26 | Sustained sadness-label patterns

**Date basis: Git archive, commit e3ca7a0; execution date unknown. Reported.**

An archived side study examines rolling sadness-label frequencies relative to
each user's full-history baseline. Its saved report analyzes 72 users and
records 5 users with at least one flagged period, totaling 24 periods. Settings
include 20-post windows, a 0.40 absolute threshold, a 0.15 excess over baseline,
and at least three consecutive flagged windows. [S06]

These counts describe behavior of a text classifier and a selected threshold
rule. They do not validate depression detection. The use of a full-history
baseline also makes this a retrospective analysis; a prospective version would
need a baseline calculated only from information available at each time.
That prospective restriction is an interpretation added in this edition.

<!-- pagebreak -->

## 2026-07-17 | Forecasting reproduction, E1-E12

**Date basis: reconstructed from early notes on this date. Execution dates
unknown. Results reported; qualifications below were added 2026-10-21.**

The experiments progressively add text embeddings, sliding windows, richer time
features, larger histories, different architectures and simpler prediction
tasks. User-disjoint splits aim to measure transfer to unseen users. [S04]

### Six-class and four-class variants

| Experiment | Task and main change | Accuracy | Weighted F1 |
| --- | --- | ---: | ---: |
| E1 | Six classes; emotion history and time flags | 0.232 | 0.279 |
| E2 | Six classes; MiniLM text and sliding windows | 0.364 | 0.342 |
| E5 | Six classes; richer time features, window 10 | 0.389 | 0.379 |
| E6 | Six classes; window 16 | 0.415 | 0.368 |
| E11 | Four classes; omit love/surprise, window 10 | 0.457 | 0.410 |
| E12 | Four classes; window 16 | 0.439 | 0.399 |

E1's saved majority and persistence accuracies are 0.348 and 0.410 on 293 test
windows. E2's majority accuracy is 0.405 on a different window set. E11 exceeds
its majority baseline of 0.416, but changes the task by excluding two target
classes. Its macro F1 is only 0.247. Improvements in headline accuracy do not
establish reliable performance across emotions. [S04, S07]

### Binary variants

| Experiment | Main change | Accuracy | Weighted F1 |
| --- | --- | ---: | ---: |
| E3 | Text embeddings, user-disjoint split | 0.570 | 0.559 |
| E4 | Row-level random split | 0.595 | 0.595 |
| E7 | Richer time features, window 10 | 0.578 | 0.553 |
| E8 | BiLSTM and attention, window 10 | 0.571 | 0.560 |
| E9 | LSTM, window 16 | 0.562 | 0.559 |
| E10 | BiLSTM and attention, window 16 | 0.553 | 0.553 |

E3 approximately ties its 0.568 majority baseline. Larger windows and attention
do not consistently improve these configurations. E4 allows the same users
across partitions, but also changes class weighting and test composition;
its roughly 2.5-point difference from E3 does not isolate the effect of leakage.
Neither binary comparison explains the original six-class SMOTE result. [S04]

**Progression:** text helps relative to E1, but architecture changes do not
recover the original score. The later runs use different encodings, training
choices and evaluation partitions from the original report. The original
result remains unreproduced by these saved experiments.

<!-- pagebreak -->

## 2026-07-17 | More users: Sentiment140, E13

**Date basis: dated experiment entry. Reported.**

A corpus swap retains 1,405 users and 50,654 sampled tweets, compared with
roughly 38 training users in the earlier corpus. The same general labeler,
embedding and V3 forecasting workflow is used. [S04]

| Task | Accuracy | Weighted F1 | Majority accuracy | Persistence accuracy |
| --- | ---: | ---: | ---: | ---: |
| Six classes | 0.215 | 0.254 | 0.501 | 0.391 |
| Four classes | 0.299 | 0.334 | 0.520 | 0.404 |
| Binary | 0.572 | 0.563 | 0.538 | 0.548 |

The binary run exceeds both baselines, with ROC-AUC 0.593. Multi-class results
remain weak. The log proposes that strong class weighting under severe
imbalance is distorting the optimization objective. This is an explanation to
test, rather than a controlled demonstration of the cause. Sentiment140 is an
emoticon-search sample, so the next retained tweet is not necessarily the next
tweet actually posted by a user. [S04]

## 2026-07-28 | Class-weight tradeoffs, E14

**Date basis: dated experiment entry. Reported.**

The follow-up changes class weights on the original corpus. The intended
Sentiment140 follow-up remains unperformed in the saved record because its
raw data/intermediates are unavailable. [S04]

| Task | Weighting | Accuracy | Weighted F1 | Macro F1 |
| --- | --- | ---: | ---: | ---: |
| Six classes | Balanced | 0.3887 | 0.3791 | 0.1725 |
| Six classes | Square root | 0.4612 | 0.4097 | 0.1738 |
| Six classes | None | 0.4338 | 0.3129 | 0.1259 |
| Four classes | Balanced | 0.4572 | 0.4099 | 0.2474 |
| Four classes | Square root | 0.4284 | 0.3036 | 0.1790 |
| Four classes | None | 0.4237 | 0.2679 | 0.1597 |

Six-class square-root weighting improves accuracy mainly through the dominant
classes; rare-label performance remains poor. Four-class performance worsens
when weights are softened. Class weighting is a task-specific tradeoff, and
the six-class accuracy gain is not broad emotion recognition. [S04]

### Reddit importer: implementation evidence only

A dated synthetic-fixture check covers cleaning, filtering, decompression and
the embedding/window chain. No real Reddit corpus or resulting accuracy study
is recorded. Labeler domain shift, subreddit shortcuts and UTC time features
remain limitations; synthetic success does not resolve them. [S08]

<!-- pagebreak -->

## 2026-09-22 | V4: auditability and stronger baselines

**Date basis: dated E15 entry and saved metrics. Reported. The entry was
written after implementation, not preregistered.**

V4 separates data auditing, grouped evaluation and model comparison. It removes
204 identical user/time/text records and skips 1,068 windows containing timestamp
ties. The prepared data has 11,801 windows from 53 users. Five development folds
use 45 users and 10,336 windows; eight reserved test users are not scored. [S04,
S05, S09]

### Binary development comparison

| Model or baseline | Accuracy | Macro F1 | Mean user accuracy |
| --- | ---: | ---: | ---: |
| Training-majority | 0.4709 | 0.4264 | 0.4637 |
| Persistence | 0.5600 | 0.5591 | 0.5489 |
| Historical frequency | 0.5833 | 0.5816 | 0.5846 |
| Transition | 0.5600 | 0.5591 | 0.5489 |
| Logistic, text/time/emotion | 0.5190 | 0.5188 | 0.5154 |
| LSTM, text/time | 0.5146 | 0.5142 | 0.4991 |
| LSTM, text/time/emotion | 0.5038 | 0.5036 | 0.4978 |

Historical frequency exceeds persistence by 2.33 percentage points. The paired
user-cluster bootstrap interval is +1.09 to +3.77 points. Both LSTM variants
trail the frequency baseline. Adding the existing historical emotion features
does not help this particular LSTM configuration. [S09]

### Data pipeline corrections

Sentiment140 spam ratios are now computed before deduplication; older outputs
must be regenerated to benefit. Reddit decompression now streams through a
temporary SQLite database rather than retaining all authors' text in memory.
Temporary disk use and the eligible-author list still grow with the corpus,
and the sampling budget can exceed its target by one author's history. Real
Reddit data remains untested in this record. [S05]

### Limits and changes in interpretation

- The run uses one-hot emotion history because the old CSV lacks soft labeler
  probabilities. This does not test newly generated probability features.
- Legacy embedding order was explicitly accepted without a verification
  manifest. V4 supports manifests for new artifacts, but that does not
  retrospectively verify this run's old embeddings.
- Labels are generated by a classifier. The target is the next retained post,
  and filtering can remove intervening posts.
- User-cluster intervals condition on fitted models; they omit refitting and
  broader model-selection uncertainty. The run is development evidence, not a
  newly scored final test result.
- UTC hour features do not establish a user's local overnight behavior. [S05,
  S09]

**Progression:** the useful outcome is a stronger comparison standard and a
more auditable workflow. The evidence favors a simple history baseline over
these learned configurations; it does not prove that forecasting is impossible
or that larger architectures would solve the observed limitations.

<!-- pagebreak -->

## 2026-10-15 | Existing-post labeling and human references

**Date basis: dated pilot artifact, public-data notes and v0.2.0 release.
Reported. Order within this date is thematic unless timestamps establish it.**

The product now imports existing text, suggests multiple emotions using a
GoEmotions model, and runs a separate sentiment classifier. It saves dataset
versions, model/settings provenance and observations. Applying these pretrained
weights does not train them on imported posts or human reviews. [S10, S11]

### Initial pilot: execution without human accuracy labels

The pilot processes 400 posts at threshold 0.5: 102 labeled, 235 neutral,
59 uncertain and 4 requiring review. Only four posts receive multiple
non-neutral labels. Human-reviewed rows are zero and quality metrics are null.
The status counts demonstrate output behavior, not the proportion correctly
labeled. They also cannot establish improvement over the historical six-class
labeler. [S12]

### GoEmotions development reference

All 5,426 official development comments are evaluated. The 5,427 official test
examples are not scored in this project evaluation. [S13, S14]

| Shared threshold | Precision | Recall | Micro F1 | Exact set match | Mixed full recovery |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.50 | 0.686 | 0.511 | 0.586 | 47.4% | 9.4% |
| 0.30 | 0.584 | 0.636 | 0.609 | 44.9% | 19.5% |

Lowering the cutoff retrieves more reference emotions and introduces additional
incorrect labels. Mixed full recovery allows extra labels, while exact set
match rejects them. Neither coverage nor recovery alone establishes quality.

The model was trained on GoEmotions; these results are an in-domain reference
comparison, not independent Twitter/upload validation. The reference retains
174 neutral-plus-emotion development rows and repeated texts. Ten text hashes
overlap the official development and test splits. The project's human rubric
instead requires neutral alone, and the model author's documented test-based
threshold tuning limits the test split's independence. [S13, S14]

### Twitter-specific comparison remains incomplete

The SemEval adapter preserves its native labels and eight exact-name shared
categories. The documented official download attempts returned HTTP 406; no
actual reference preparation or model score was completed. Synthetic adapter
tests do not fill that evidence gap. This is the recorded status on this date,
not a fresh availability check. [S13]

<!-- pagebreak -->

## 2026-10-15 | Threshold refinement exposes tradeoffs

**Date basis: artifact timestamp 2026-10-15T23:01:15Z, October 15 locally.
Reported development research.**

Five decision rules use the same 5,426 development rows and 5,423 exact-text
groups. Learned cutoffs are fitted on four folds and applied to the remaining
fold; the pretrained model is not retrained. There are 712 reference posts
with multiple non-neutral emotions. [S15, S16]

| Rule | Micro F1 | Macro F1 | Mixed full recovery |
| --- | ---: | ---: | ---: |
| Fixed 0.50 | 0.585922 | 0.447795 | 9.41% |
| Historical 0.30 | 0.609231 | 0.496341 | 19.52% |
| Fine shared cutoff | 0.607499 | 0.494251 | 18.82% |
| Joint micro per-label | 0.614223 | 0.480234 | 17.70% |
| Per-label F1 | 0.605527 | 0.492843 | 19.80% |

### Finding

The joint-micro profile improves micro F1 over historical 0.30, but reduces
macro F1 and mixed-emotion recovery. False-positive decisions fall from 2,886
to 2,613 while missed labels increase from 2,321 to 2,394. Seventeen labels
lose F1, seven improve and four are unchanged. The highest aggregate score
does not identify a universally better label-selection rule. [S15]

The paired micro-F1 difference interval is +0.000685 to +0.010064, from 500
exact-text-group resamples. It holds fitted decisions fixed and omits threshold
fitting, method selection, annotation and upstream-model uncertainty. The
historical 0.30 comparator was previously chosen on the full development set;
this is not a comparison with an untouched baseline. [S15, S16]

### Decision and remaining limits

No threshold candidate improves micro F1 while preserving both macro F1 and
mixed-label recall against the historical comparator. Manual shared thresholds
remain the default; exported profiles require explicit selection and matching
model provenance. They change the label decision rule, not model weights or
probability calibration. [S15]

Rare labels have inadequate fitting support, some fold cutoffs are unstable,
and exact-text grouping cannot account for different comments by the same
author. An objective-specific improvement on these development annotations
still needs independent review on the user's target posts.

<!-- pagebreak -->

## 2026-10-15 | A different pretrained model

**Date basis: saved run identifier beginning 20261015T230413Z, October 15
locally. Reported; exploratory comparison.**

The study compares the existing GoEmotions RoBERTa model with one pinned
ModernBERT candidate. Both receive the same 5,426 posts, references and grouped
folds, using a 256-token limit. Different tokenizers can retain different
amounts of text within that limit. [S15, S17]

| Model and rule | Micro F1 | Macro F1 | Exact set match | Mixed full recovery |
| --- | ---: | ---: | ---: | ---: |
| RoBERTa, shared 0.30 | 0.6092 | 0.4963 | 44.90% | 19.52% |
| ModernBERT, shared 0.30 | 0.6195 | 0.5378 | 43.86% | 24.44% |
| ModernBERT, fitted shared | 0.6192 | 0.5358 | 44.95% | 22.05% |
| ModernBERT, joint micro per-label | 0.6211 | 0.5247 | 46.66% | 18.82% |

### Finding

At shared 0.30, ModernBERT improves micro F1, macro F1 and mixed full recovery.
Precision falls from 0.5844 to 0.5796, exact match declines, and neutral/emotion
conflict flags rise from 736 to 946. Its highest-micro-F1 profile sacrifices
some of the mixed-emotion benefit. [S15, S17]

The shared-0.30 paired micro-F1 difference is about +0.01030, with a conditional
95% text-group bootstrap interval of +0.00294 to +0.01708. This interval does
not include candidate selection or upstream-model uncertainty. [S15]

### What this changed

ModernBERT becomes an explicit research choice, with model-specific threshold
profiles. The app retains RoBERTa as its default. Model choices, copied
thresholds and file fingerprints are recorded with new analyses. [S11, S15]

Both models were trained on GoEmotions. The candidate was selected after
earlier results, the model card identifies it as superseded in the saved
record, and this bounded comparison does not survey all available models.
No official test examples are scored here. The result supports a further
target-data comparison, not a claim that the app is now more accurate on the
user's posts. [S17]

**Progression:** changing the pretrained model yields a more useful mixed-label
development comparison than optimizing one score through thresholds alone.
Whether that benefit transfers to Twitter or the user's uploads is unresolved.

<!-- pagebreak -->

## 2026-10-16 | Execution parity and concrete error cases

**Date basis: dated diagnostic notes, changelog and handoff. Reported.**

Twelve synthetic scenarios exercise the CLI and native desktop inference.
The eleven nonblank inputs agree within a 0.00001 score tolerance, with a
maximum absolute difference around 0.0000000103. Labels and statuses match;
blank text is invalid or rejected. This demonstrates execution consistency
on the constructed examples. It is not an accuracy benchmark. [S18]

### Labeling weaknesses observed at threshold 0.50

| Scenario | Observed result | Limitation exposed |
| --- | --- | --- |
| Excited to move, terrified to leave | Fear only | Mixed emotions can be missed |
| Happy about promotion, sad about departure | Sadness only | Multi-label support does not guarantee full recovery |
| Sarcastic cancelled-train complaint | Admiration | Literal positive wording can mislead the model |
| Context-dependent ambiguous statement | Neutral | Missing context can be mistaken for neutrality |
| Spanish mixed-emotion sentence | Uncertain | One abstention does not validate language detection |
| Long repeated text | Truncated; review needed | Only retained tokens contribute to prediction |

These judgments are provisional project interpretations. There are no
independent human gold labels or adjudication for these cases. They identify
challenge categories to study, not measured population error rates. [S18]

### Runtime checks have a different scope

The handoff records 108 passing unit tests, a passing JavaScript syntax check,
and a native smoke test with no JavaScript errors or listening TCP socket.
The smoke test exercises application flows with disposable research state.
The duplicate browser UI is retired and the desktop paths share an inference
engine. These checks support implementation behavior; they do not establish
label accuracy or scientific validity. [S10, S19]

### Human-review evidence remains to be built

The recommended workflow uses a 30-post blind practice round to learn the
rubric, then a separate representative target-data benchmark with a distinct
challenge set. The practice round alone cannot establish reliability. Reviewers'
judgments stay separate from model predictions. Agreement, disagreements,
adjudication, rare-label support and abstentions need to be reported. [S20]

Some older Markdown notes describe per-label thresholds as future work;
the October 15 artifacts document their later implementation. That completion
does not complete independent human review or establish target-data accuracy.

<!-- pagebreak -->

## 2026-10-21 | Historical audit and corrections

**Date basis: review conducted on this date. Arithmetic and saved-code defects
reproduced; original training procedure remains unknown.**

The original report, preserved notebook and repository history are compared
directly. The page-6 confusion matrix yields 694 correct predictions out of
843, accuracy 0.8232503 and weighted F1 0.8182259. The approximate 0.82 claim
has supporting arithmetic. [S02, S03]

### Evaluation provenance is the larger uncertainty

The matrix's class supports are [157, 154, 143, 127, 118, 144]. Reconstructing
V1 windows from the current saved corpus produces 1,323 complete windows with
target counts [185, 607, 16, 408, 100, 7]. Under the preserved mapping, class 5
is surprise; the entire filtered corpus has only 64 surprise-labeled posts.
If the original run used those same labels and windows, its 144 class-5 test
examples cannot all be distinct untouched original windows. [S03]

This is strong reason to investigate evaluation resampling or different source
artifacts. It does not prove the original order of SMOTE and splitting, or
establish that leakage alone explains the score.

### Two saved-code defects were reproduced

- The notebook declares 29 columns but assigns 28-value rows. A clean replay
  raises a mismatched-column error. The CLI preprocessing uses 28 columns.
- Incorrect final-batch slicing produces oversized trailing batches. For 21
  posts, batch lengths are 10, 10 and 29; the row-length filter discards the
  last batch. Later versions use complete sliding windows. [S03]

All four notebook versions found in accessible history contain preprocessing
only. They do not contain the original training, SMOTE or splitting code.
Saved-code defects are confirmed; their role in the original score is unknown.

### Corrections and local reproducibility

The E1 baseline summary is corrected to majority 0.348 and persistence 0.410
to match its saved metrics. The E4 explanation no longer attributes its whole
gain to leakage because class weighting and test composition also changed.
Prior conclusions must be read with these dated qualifications. [S03, S04]

The checkout has no outgoing commits at the start of this review; apparent Git
changes were line-ending differences. The current machine lacks the EXE,
workbench environment and ignored runs/model cache described in older notes.
Legacy arrays exist at the root rather than current pipeline paths. No model
was retrained and no full application test was rerun during this audit. [S03,
S19]

<!-- pagebreak -->

## 2026-10-21 | Evidence gaps and the next study record

**Synthesis as of this edition. Items below are open work, not findings from
completed experiments.**

### Questions that remain open

- **Original score:** recover the actual training/SMOTE script, frozen labels,
  split indices and predictions. Check resampling order, untouched evaluation
  rows, shared users/windows, target-feature separation and categorical encoding.
- **Forecasting:** confirm old embedding alignment or regenerate verified
  artifacts; compare probability-history features and baselines on fixed
  development users before a final test. More architecture alone is not an
  established answer to the observed weak signal.
- **Existing-post quality:** collect independent human judgments on target
  posts, separating representative sampling from enriched challenge examples.
  Assess mixed emotions, sarcasm, ambiguity, unsupported language and truncation.
- **External comparison:** complete a documented Twitter-specific benchmark
  when the official data is accessible, preserving its label scope. Existing
  GoEmotions results do not fill this gap.
- **Reproducibility:** restore the local environment and provenance artifacts
  before claiming that historical runtime checks apply to this checkout.

### Current interpretation

The study has progressed from chasing a high forecasting score toward more
explicit targets, stronger baselines, preserved source identities and independent
review requirements. The record contains useful development comparisons and
reproducible software checks. Reliable forecasting at the original reported
level, improved labeling on the user's posts, and clinical interpretation of
emotion outputs have not been established by the saved evidence.

### How to add the next entry

Keep this Markdown file as the editable master and regenerate the PDF. Append
new dated entries before the source register. If an old conclusion changes,
record a dated correction that points to the earlier entry. Do not overwrite
old scores or assign an execution date from a file modification timestamp.

Each entry should state the event date and its basis; the question; task,
dataset and split; what changed; results with baseline and sample size; the
supported conclusion; limitations; next question; and source/run references.
Mark proposals, synthetic tests, development results and final held-out results
explicitly. Add new sources to the register and retain underlying artifacts.

Rebuild from the repository root with:

`python -m pip install reportlab`

`python tools/build_study_progression.py`

The builder reads this file, creates the PDF and a source-hash manifest under
`output/pdf/`. Review the rendered pages after substantive edits.

<!-- pagebreak -->

## Source register

Paths are relative to the repository unless marked external. This is a register
of project evidence, not a fresh verification of external links or dataset
availability. The build manifest records SHA-256 hashes of available sources.
Missing ignored run artifacts limit reproduction even when summaries remain.

- **[S01]** External: `Project Proposal Presentation.pptx`, user-supplied from
  the local Downloads folder. Original proposal; date not established.
- **[S02]** External: `Social Media Project Report.pdf`, user-supplied from
  the local Downloads folder. Final original report, especially pages 3-8.
- **[S03]** `docs/HISTORICAL_REVIEW.md`; saved notebook
  `docs/CAP4773_fin_1.ipynb`; `src/preprocess.py`;
  `data/interim/dataset_godknowswhat.csv`. Review dated 2026-10-21.
- **[S04]** `docs/EXPERIMENTS.md`. E1-E15, including dated qualifications.
- **[S05]** `docs/V4.md`. Historical forecasting protocol and known limits.
- **[S06]** `twitter_feel_plus/results/REPORT.md`, `params.json`,
  `aggregate.json`; `twitter_feel_plus/README.md`; `sadness_flag.py`.
  Report first archived in commit `e3ca7a0` on 2026-05-26.
- **[S07]** `metrics/metrics.json`, `metrics_v2.json`, `metrics_binary.json`,
  `metrics_leaky_binary.json`, and the corresponding `metrics_windows_v3_*.json`.
  Full files are under `metrics/`; metric filenames abbreviated here only.
- **[S08]** `docs/REDDIT.md`; `src/prepare_reddit.py`. Synthetic verification
  dated 2026-07-28; later ingestion fixes documented in [S05].
- **[S09]** `metrics/metrics_v4_mht_binary_development.json`. Dated 2026-09-22.
- **[S10]** `CHANGELOG.md`; `app_version.json`. v0.2.0 and workspace cleanup.
- **[S11]** `docs/WORKBENCH.md`; `src/analysis_engine.py`;
  `src/analysis_jobs.py`. Current workflow and inference behavior.
- **[S12]** `metrics/emotion_labeling_pilot_v1.json`. Dated 2026-10-15.
- **[S13]** `docs/PUBLIC_DATASETS.md`. Availability checks dated 2026-10-15.
- **[S14]** `metrics/goemotions_development_v1.json`. Public-reference results.
- **[S15]** `docs/THRESHOLD_REFINEMENT.md`. Threshold/model tradeoffs and caveats.
- **[S16]** `metrics/emotion_threshold_refinement_v1.json`. Dated timestamp.
- **[S17]** `metrics/emotion_model_comparison_v1.json`. Timestamped run ID.
- **[S18]** `docs/LABELING_CHECK.md`. Diagnostic dated 2026-10-16.
- **[S19]** `HANDOFF.md`. Current scope, prior runtime checks and checkout audit.
- **[S20]** `docs/EMOTION_LABELING.md`. Annotation and evaluation protocol;
  some planning statements are superseded by later artifacts [S15-S17].

### Edition history

Edition 1, 2026-10-21: backfilled the historical documents, forecasting log,
sadness-pattern report, V4 comparison, labeling pilot, public-reference work,
threshold/model comparisons, execution diagnostics and historical audit. No
new model training or final held-out evaluation was performed for this edition.
