# Label the emotion expressed in a post

## What we are trying to do

Read one existing post and identify the emotions its text expresses. A post
can express several emotions, no clear emotion, or too little information
to decide. We are not predicting the next post or diagnosing its author.

The simple workflow is:

**Prepare text -> score each label -> select supported labels -> review and evaluate.**

The historical `generate_dataset.py` chooses one winner from six labels.
Its scores compete through softmax, so keeping several of those scores does
not make it a multi-label classifier. The LSTM experiments evaluate future
machine-generated labels; they do not measure the quality of labeling posts.

## Current commands

Start with original text when available. The CLI prefers `text`, then
`post_text`, then `cleaned_text`; `--text-column` overrides that choice. The
canonical review column `text` contains exactly what is scored.

```powershell
python src/label_posts.py sample data/raw/Mental-Health-Twitter.xls --out runs/labeling_review_v1 --size 400
python src/label_posts.py label runs/labeling_review_v1/review.csv --out runs/labeling_candidate_v1
python src/label_posts.py evaluate runs/labeling_review_v1/review.csv runs/labeling_candidate_v1/predictions.jsonl --split development --out runs/labeling_dev_v1.json
```

Before evaluation, fill `human_labels` with semicolon-separated label names
(for example `excitement;fear`) and change `review_status` from `pending` to
`reviewed`. Use `unclear` for unresolved human judgments and `invalid` for
unusable input; these are reported separately. Do not change text or identifiers.
Keep test rows untouched while choosing settings. The initial sample is random,
not the enriched challenge set recommended below. Inference downloads model
files unless a local model directory is supplied. Each output needs a new name.

The bundled raw export repeats some post IDs. Ingest retains the original
as `source_post_id` and uses source ID plus export-row identity for a unique
`post_id`. Treat these row identities as specific to that source export.
The review sampler deduplicates identical text before splitting.

## Starting vocabulary and candidate model

Use the GoEmotions vocabulary as a starting point: 27 detailed categories
and neutral. The research dataset contains English Reddit comments with
human annotations and permits multiple labels. This is a practical vocabulary,
not a complete or universally agreed theory of emotion.

`SamLowe/roberta-base-go_emotions` is a candidate that supports independent
scores for those labels. Compare it with the old model before adopting it.
Neither its published Reddit results nor a successful local run establishes
that it labels our tweets better. Scores are model support, not emotion
intensity or calibrated certainty about a person's feelings.

Retain the original text, the exact text scored, source-row identity, all
scores, selected labels, model revision, tokenizer settings, threshold
configuration, and schema version. Avoid overwriting historical datasets.
Inspect whether cleaning removes emoji, punctuation, negation, or other
emotion cues before choosing which text field to score. Record truncation.

## Related labels without losing the distinctions

Keep every fine label. An optional display layer can group them as below.
These are **proposed product groupings**, not the paper's validated ontology.
Review the grouping against human judgments before using grouped metrics.

| Proposed display group | Detailed labels |
| --- | --- |
| Anger-related | anger, annoyance, disapproval |
| Sadness-related | sadness, disappointment, grief, remorse |
| Fear-related | fear, nervousness |
| Joy-related | joy, amusement, excitement, pride, relief, optimism |
| Affection and appreciation | love, caring, admiration, gratitude, approval |
| Interest and understanding | curiosity, confusion, realization |
| Desire | desire |
| Disgust | disgust |
| Embarrassment | embarrassment |
| Surprise | surprise |
| Neutral | neutral |

Do not infer that adjacent emotions are interchangeable or lie on a fixed
intensity ladder. For example, annoyance need not mean weak anger. Surprise
may occur with joy, fear, or disappointment. Positive and negative emotions
may coexist. A group is present when at least one selected member is present;
do not add scores or interpret their maximum as a calibrated group probability.

The current CLI emits only five small display groups: annoyance/anger,
disappointment/sadness/grief, nervousness/fear, amusement/excitement/joy,
and caring/love. Other labels remain visible individually. The fuller table
above is a proposal for review, not the implemented mapping.

## Neutral, uncertainty, and invalid input

Use separate states:

- **Labeled:** one or more supported non-neutral labels pass the configured thresholds.
- **Neutral:** neutral passes its threshold and no non-neutral label passes.
- **Uncertain:** no label passes.
- **Review needed:** neutral and a non-neutral label both pass, or the text was
  truncated. Retain conflicting candidate labels and all scores for review.
- **Invalid:** blank or otherwise unusable input; do not run inference on a
  missing value silently converted to an empty string.

Unsupported language, missing context, and truncation can also warrant review
flags. Low scores do not prove neutrality. Do not force a top label when no
label passes. Do not cap every post at two labels. A default threshold of 0.5
is an initial operating setting only; tune thresholds on reviewed development
examples and freeze them before evaluating held-out examples.

## Human annotation rubric

The versioned, machine-readable rubric is
[`src/twitterfeel/labeling/annotation_rubric.json`](../src/twitterfeel/labeling/annotation_rubric.json). The desktop review
screen displays its instructions and searchable emotion definitions. The
working definitions below explain the same distinctions in prose.

### Start with a 30-post practice round

In **Label review** or **Experiments**, select the intended review sample and
choose **Prepare 30-post practice round**. This creates a separate blind review
sheet containing up to 30 unique, nonblank **development** posts. Reserved test
posts are excluded. Repeating the request with the same source and rubric
returns the same practice sheet; changing the rubric creates a separate
artifact. Smaller eligible samples produce a warning.

The practice sheet begins with pending judgments, blank labels, and blank notes.
Source annotations and model predictions are not copied. The first reviewer
must make the judgments; the software does not supply AI-generated gold labels.
Review the definitions, select all emotions supported by the text, and record
evidence and difficult distinctions. This is a calibration exercise, not an
accuracy estimate or an agreement measurement.

For this phase, one reviewer can complete the practice round and start the
larger development review. Treat those labels as a **single-reviewer draft**.
A second reviewer can later use a distinct reviewer name to label the same
examples independently, before comparing decisions and adjudicating differences.
The app preserves separate review revisions; it does not automatically adjudicate
or turn practice judgments into labels for the full benchmark.

Keep practice and benchmark results clearly identified. After practice, revise
ambiguous rubric wording if needed, then freeze/version it for the main review.
The calibration metadata records the source fingerprint, rubric version/hash,
selected post identities, and zero selected test rows. Original review sheets
remain unchanged.

### Working label definitions

These are provisional project annotation definitions, written to distinguish
nearby labels. They are not verbatim source definitions or psychological
diagnoses. Revise them during the shared pilot, then version and freeze the
sheet for the benchmark. Select multiple labels only when each has evidence;
do not select every neighbor merely because the distinction is difficult.

| Label | Working definition and useful distinction |
| --- | --- |
| admiration | Respect or appreciation for someone or something's qualities; distinguish from approving a particular action. |
| amusement | Finding something funny or entertaining; distinguish from general happiness. |
| anger | Expressing hostility, outrage, or being wronged; distinguish from an objection without expressed anger. |
| annoyance | Irritation or being bothered; distinguish from disapproval, which evaluates something negatively. |
| approval | Expressing agreement, endorsement, or a favorable judgment; distinguish from admiration for qualities. |
| caring | Concern for another's well-being or a wish to help; distinguish from affectionate attachment. |
| confusion | Difficulty understanding or making sense of something; distinguish from an interested question without confusion. |
| curiosity | Interest in learning or discovering something; a question alone need not express curiosity. |
| desire | Wanting or longing for something; distinguish from expecting a good outcome. |
| disappointment | An expectation or hope was not met; distinguish from sadness without that specific cause. |
| disapproval | Objection or an unfavorable judgment; does not require anger or disgust. |
| disgust | Revulsion or repulsion; distinguish from simply judging something wrong. |
| embarrassment | Self-conscious discomfort about exposure, awkwardness, or others' judgment; distinguish from regret about harm done. |
| excitement | Energized anticipation or enthusiasm; distinguish from quieter contentment or general joy. |
| fear | Feeling threatened or afraid of danger or harm; distinguish from anticipation-related nervousness. |
| gratitude | Thankfulness for a benefit, help, or kindness; distinguish from admiration alone. |
| grief | Sorrow centered on a significant loss; distinguish from sadness with no expressed loss. |
| joy | Happiness, delight, or pleasure; amusement and excitement can be selected too when independently expressed. |
| love | Affectionate attachment or deep fondness; distinguish from concern alone. |
| nervousness | Apprehension, tension, or unease about an uncertain situation; may coexist with explicit fear. |
| optimism | Hope or expectation of a favorable outcome; distinguish from wanting that outcome without expecting it. |
| pride | Satisfaction in an achievement, identity, or affiliation; distinguish from admiration directed elsewhere. |
| realization | Expressing a new recognition or understanding; distinguish from merely reporting an already-known fact. |
| relief | Release of worry or tension after a concern eases; distinguish from general joy. |
| remorse | Regret or guilt about one's own action or responsibility; distinguish from disappointment in an outcome alone. |
| sadness | Unhappiness, sorrow, or feeling down; use grief/disappointment too only with their specific evidence. |
| surprise | Unexpectedness, astonishment, or being taken aback; has no automatic positive or negative meaning. |
| neutral | Understandable text without a clear expressed emotion; never a substitute for missing context or unreadable input. |

Annotate the emotion expressed by the post's speaker, using only the displayed
text. Do not infer a hidden mental state from the topic or author's history.
Do not automatically transfer a quoted person's emotion to the speaker.

1. Mark blank/unusable input as invalid. Mark insufficient context as uncertain.
2. Select every clearly supported label, including emotions in different clauses.
3. Choose neutral only when the text is understandable and expresses no clear
   emotion. Never pair a human neutral judgment with an emotion label.
4. Record an evidence phrase and a brief reason for ambiguous choices. Flag
   sarcasm, quotation, negation, mixed emotions, and context dependence.
5. Keep independent reviewers' original decisions. Discuss disagreement and
   record an adjudicated judgment separately; unresolved cases stay uncertain.

| Text | Example judgment | Reason |
| --- | --- | --- |
| "Excited to move, but scared to leave everyone." | excitement, fear | Two explicit emotions coexist. |
| "The meeting starts at three." | neutral | An understandable factual statement. |
| "Well, that was something." | uncertain | Emotion depends on missing context. |
| "I'm not angry; I'm disappointed." | disappointment | Negation rules out the explicitly denied anger. |
| "She said 'I'm terrified.' The interview is attached." | neutral, or uncertain if stance is unclear | Quoted fear alone does not establish the speaker's emotion. |

These examples illustrate the rubric; human judgment is still needed. Annotators
should have a label-definition sheet and run a shared pilot of about 30 posts
before the main review. Refine wording after the pilot, then freeze it. Hide
model predictions during annotation to reduce anchoring.

## A benchmark that can tell us whether this is better

Start with roughly 300-500 real posts. This is an initial benchmark, not enough
to make strong claims about every rare label. Keep a random representative
sample separate from an enriched challenge set containing mixed emotion,
sarcasm, neutral language, quotation, and model disagreements. Report their
results separately; enrichment changes the apparent class frequencies.

Split development and final evaluation examples by user where IDs are available;
keep duplicates and near duplicates together. Freeze the split before threshold
selection. Have two reviewers independently label at least a substantial shared
subset, ideally all evaluation posts. A single reviewer's draft is not a verified
gold standard. Record agreement, disagreements, and adjudication outcomes.

For each model, report:

- Per-label precision, recall, F1, and positive support; macro and micro F1.
- Example-level exact set match and Jaccard overlap as complementary measures.
- Neutral precision/recall, mixed-emotion recall, uncertainty rate, and invalid count.
- Coverage (fraction receiving an automatic label) alongside quality on those
  covered posts. Do not silently exclude abstentions from full-set metrics.
- Representative error examples and uncertainty intervals, clustering by user
  when repeated posts from the same user are present.

**Implemented now:** `evaluate` reports per-label precision/recall/F1/support
(including neutral), micro F1, macro F1 across all labels and across labels with
positive support, exact match, mean example Jaccard, automatic acceptance
fraction, and excluded human-review status counts. Empty uncertain predictions
remain in the metrics as misses. `label` separately reports prediction status
counts, label counts, and truncation counts. Review-needed candidate labels
still enter the full-set comparison; acceptance fraction excludes those rows.

**Development threshold comparisons:** the desktop **Experiments** page now
compares shared thresholds 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, and 0.8 using the
selected reviewer's saved development judgments and the active run's stored
scores. It reports precision, recall, micro F1, exact match, per-label support,
and mixed-emotion recovery. If no reviewed development rows exist, the report
stays not ready: pending human input is not replaced by automatic reference
labels. Test labels do not enter this comparison.

The provisional selection rule chooses the highest development micro F1; ties
prefer the threshold closest to 0.5 and then the higher threshold. This compares
operating settings for the same model. It does not train a model, change saved
settings, or establish held-out accuracy. Use another analysis run to adopt a
chosen threshold, after reviewing enough independent examples.

The report includes a 95% percentile bootstrap interval for micro F1 using 200
resamples and seed 42. It resamples users when complete user IDs are available;
otherwise it explicitly falls back to posts and warns that dependence between
posts from one person is not accounted for. Legacy user IDs are recovered only
when the complete source fingerprint and every row/post/text identity match.
Missing identities remain missing. Small samples, few users, and rare labels
produce warnings. These intervals do not include uncertainty from choosing the
best threshold or from annotation disagreements.

For human multi-emotion posts, **all labels recovered** asks whether every
reference emotion was selected; additional incorrect labels can still occur.
**Mixed label recall** measures recovered reference labels across those posts.
Exact set match is stricter and also rejects extra labels. None is reliable for
a category with little or no human support; inspect precision and support too.

**Still to add for the recommended benchmark:** reviewer agreement and
adjudication tooling, metrics restricted to automatically accepted rows, grouped
metrics, near-duplicate detection, a separately sampled challenge set, and an
automated six-label baseline comparison. Per-label threshold tuning is also a
future extension. Individual reviewers and calibration records are supported;
independent human review and adjudication must still happen.

Unknown/uncertain human judgments are not all-negative emotion examples.
Report them separately and state which adjudicated rows enter each metric.
Rare labels may have no evaluation positives; show their support and avoid
presenting an undefined score as reliable evidence of failure or success.

The six-label baseline cannot emit neutral or additional fine labels. Compare
both models on the shared six labels, retaining multi-label human targets,
then evaluate the candidate's broader coverage separately. Also report its
unsupported cases and abstentions. Do not compare historical next-post accuracy
with multi-label post-labeling F1 or equate same-name labels across taxonomies
without documenting that choice.

Choose the operating thresholds and any optional grouping on development data.
Only then inspect final evaluation performance. With few positive development
examples, prefer a shared threshold over heavily tuned per-label settings.
If evidence remains weak, grow the benchmark rather than claim improvement.

## Reprocessing gate

A small inference smoke run checks that the pipeline works. A reviewed benchmark
checks labeling quality. Full-corpus reprocessing should produce a versioned
candidate dataset, with provenance and a distribution/error report, after the
review supports the chosen operating settings. Model output remains automatic
annotation, not human ground truth.

## Sources

- [GoEmotions paper](https://aclanthology.org/2020.acl-main.372/): dataset,
  vocabulary, multiple emotions, and evaluation limitations.
- [Original dataset repository](https://github.com/google-research/google-research/tree/master/goemotions):
  annotation fields, including very unclear examples with no selected emotions.
- [Candidate model card](https://huggingface.co/SamLowe/roberta-base-go_emotions):
  multi-label training, inference, and label-dependent operating thresholds.

The workflow, decision states, display groups, and benchmark plan above are
project recommendations, not claims established by those sources.
