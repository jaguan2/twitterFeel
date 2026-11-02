# Original project context and reproduction review

Reviewed 2026-10-21. This records a source review and small in-memory checks,
not a new training run or an independent validation of model accuracy.

## Sources and evolution

The user supplied two historical documents, read without modifying them:

- `Project Proposal Presentation.pptx` (local Downloads folder): the proposal
  asks about future mental-health risk and proposes TF-IDF/XGBoost classification
  of existing tweets into four categories. That proposed classifier would not,
  by itself, evaluate future onset of a condition.
- `Social Media Project Report.pdf` (local Downloads folder): the final project
  instead asks how to predict the next tweet's emotion from tweet history.
  Pages 3-5 describe six transformer-generated labels, weekend/overnight flags,
  ten-post batches from one user, an LSTM, and SMOTE. The first nine posts provide
  history and the tenth supplies the target in the preserved preprocessing code.
  Page 6 reports approximately 0.82 accuracy, precision, recall, and F1 for the
  filtered-data/SMOTE variation. Page 8 compares four variations.

These documents are historical evidence, not current implementation instructions.
They have not been copied into the repository. Their original experiment date
and exact executable training configuration have not been established here.

The later repository experiments attempted to reproduce and improve that
forecasting result. The current v0.2.0 product labels **existing posts**, with
multiple emotion labels and a separate sentiment classifier. Forecasting and
existing-post labeling answer different questions and use different vocabularies.

## Current code structure

| Area | Main files and responsibility |
| --- | --- |
| Desktop application | `src/desktop_app.py`, `desktop_api.py`, `desktop_ui/`: native WebView2 window, Python bridge, HTML/CSS/JavaScript interface |
| Inference | `src/analysis_engine.py`, `analysis_jobs.py`: local pretrained models, independent emotion/sentiment scoring, background jobs and saved provenance |
| Research records | `src/research_workspace.py`, `review_protocol.py`, `workbench_data.py`: immutable imports, findings, blind practice and separate reviewers' judgments |
| Current label vocabulary | `src/label_schema.py`, `annotation_rubric.json`: 27 emotions plus neutral, threshold interpretation and human review definitions |
| Labeling evaluation | `src/benchmark.py`, `public_benchmark.py`, `threshold_tuning.py`, `compare_models.py`, `semeval_data.py`: human/published reference comparisons and development-only threshold/model research |
| Historical forecasting | `src/generate_dataset.py`, `generate_embeddings.py`, `preprocess*.py`, `train*.py`, `evaluation.py`: six-class pseudo-labels, history windows, baselines and learned predictors |
| Historical sadness patterns | `twitter_feel_plus/`: rolling sadness-label frequencies and sustained-pattern summaries |
| Evidence and checks | `docs/`, `metrics/`, `tests/`: methods, saved summaries and automated checks; large runtime artifacts belong under ignored `runs/` |

Current emotion inference uses a GoEmotions RoBERTa model, with ModernBERT as an
explicit research option. Sentiment uses a separate Cardiff Twitter classifier.
Inference applies pretrained weights; importing or reviewing posts does not
automatically train those models. Historical LSTM checkpoints in `models/` are
separate from the current app's downloaded model cache.

## What the original score establishes

The page-6 confusion matrix can be transcribed as follows, with rows=true and
columns=predicted, using the figure's numeric class IDs:

```text
113  22   3   9  10   0
 31  84   3  22  13   1
  0   0 139   0   4   0
  7   6   2 111   1   0
  4   6   2   3 103   0
  0   0   0   0   0 144
```

Recalculation gives 694 correct out of 843, accuracy **0.8232503**,
weighted precision **0.8190914**, weighted recall **0.8232503**, and weighted
F1 **0.8182259**. Those values agree with the report's approximately 0.82 claim.
The arithmetic is supported; the construction and independence of the evaluation
set remain unresolved.

The evaluated class supports are `[157, 154, 143, 127, 118, 144]`, much closer
to balanced than the preserved corpus. A read-only reconstruction from the
current `data/interim/dataset_godknowswhat.csv` finds:

- 20,000 posts / 72 users before filtering;
- 13,454 posts / 54 users after the 20-1,000 posts-per-user filter;
- 1,323 complete V1 windows, with target counts `[185, 607, 16, 408, 100, 7]`;
- only 64 surprise-labeled posts in the entire filtered corpus.

Under the preserved mapping, class 5 is surprise. If the report used these
same labels and window definitions, its 144 class-5 evaluation examples cannot
all be distinct, untouched original windows. This is strong evidence to check
for resampled evaluation data or a different source/label artifact. It does
not establish the exact original SMOTE/split order. The report itself mentions
oversampling increasing prediction counts, but supplies no executable split code.

## Confirmed defects in the preserved notebook

`docs/CAP4773_fin_1.ipynb` contains preprocessing only. All four versions found
in accessible Git history also lack model training, SMOTE, and train/test split
code, including the initial upload (`0d483c3`).

1. **Column-count error:** cell 10 creates `new_df` with `range(29)`, while
   cells 11 and 13 generate and assign 28 values. Replaying that assignment
   raises `ValueError: cannot set a row with mismatched columns`. The notebook
   as saved cannot produce its intended matrix in a clean sequential run.
   `src/preprocess.py` uses the correct 28-column allocation.
2. **Incorrect remainder slicing:** cell 12, also preserved in
   `src/preprocess.py::split_group`, takes `group.iloc[-num_batches * 10:]`
   for the final partial batch. For 21 posts, batch lengths become 10, 10, 29;
   for 25 posts, they become 10, 10, 25. The 28-value row filter then rejects
   those oversized trailing batches. Its confirmed effect for retained users
   is loss of trailing partial data. V2 and later use complete sliding windows.

These defects demonstrate problems in the saved implementation. Neither defect
establishes how the original high score arose: the code actually run for that
score may have differed from the saved notebook.

## Limits of the later reproduction evidence

The saved V1 report in `metrics/metrics.json` records accuracy **0.2321** and
weighted F1 **0.2791** on 293 test windows, with majority accuracy **0.3481**
and persistence accuracy **0.4096**. E1's baseline summary in the experiment
log was corrected during this review to match that artifact.

`src/train.py` uses one-hot emotion history, class weights and user-disjoint
train/validation/test groups. The original report describes SMOTE and does not
establish equivalent splits or encodings. This is a later evaluation of the
same forecasting idea, not an exact replay of the report's training procedure.
Its failure to reach 0.82 does not identify a single cause for the discrepancy.

The V2 binary row-split comparison records approximately 0.595 accuracy versus
0.570 for the user-split variant. These also differ in class weighting and test
composition, so the difference cannot isolate the causal effect of leakage.
Neither binary result directly reproduces the original six-class SMOTE run.

V4 improves auditability through source-row identities, strict chronology,
embedding manifests, user-disjoint development folds and stronger baselines.
Its saved binary development comparison has historical-frequency accuracy
**0.5833**, versus LSTM text/time **0.5146** and text/time/emotion **0.5038**.
Those development results use different data cleaning, tasks and splits from
the original report. They do not measure the current multi-label app's accuracy.

## Questions to resolve before claiming reproduction

Recover the original training notebook/script, frozen labeled input, split
indices, and any raw predictions if they still exist. Check whether SMOTE ran
before partitioning or affected validation/test rows; whether users and source
windows crossed partitions; whether the target appeared in its own features;
and how categorical emotion IDs were represented during resampling. Compare
against untouched original test examples and majority/persistence baselines.
Until those artifacts are available, retain the 0.82 score as an original
reported result with unresolved evaluation provenance.

## Local checkout observations

On 2026-10-21, before these documentation edits, `main` matched remote
`1ab5ece`, `git pull --ff-only` reported up to date, and the working tree was
clean after setting repository-local `core.autocrlf=true`. The 73 apparent
modifications were verified to be line-ending differences only.

This checkout currently lacks `TwitterFeel.exe`, `.venv-workbench`, and `runs/`.
Legacy embedding/window arrays exist at the repository root, while the current
pipeline expects them under `data/interim/`. No arrays were moved or regenerated,
no models were downloaded/trained, and no full app/test run was performed in
this review. The 2026-10-16 runtime verification in the handoff is a historical
record, not a verification of this machine's present setup.
