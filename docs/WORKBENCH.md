# TwitterFeel v0.2.0: using the desktop research workspace

## Open and prepare the app

Double-click **`TwitterFeel.exe`** in the project folder. The installed app opens
in its own Windows WebView2 window, with bundled interface assets and a direct
Python bridge. No HTTP server or external browser tab is involved. Closing the
window closes the app.

For a fresh checkout, use Python 3.12 and Microsoft Edge WebView2 Runtime:

```powershell
python -m venv --system-site-packages .venv-workbench
.venv-workbench/Scripts/python.exe -m pip install -r requirements-desktop.txt
.venv-workbench/Scripts/python.exe src/setup_models.py --task all
```

That one model-setup command installs both pretrained classifiers. `--task
emotion` and `--task sentiment` select one. Existing complete caches at the
correct revisions are reused without downloading again. The setup script checks
cached source metadata and available SHA-256 checksums, records file hashes, and
leaves model weights in `runs/model_cache/`. The sentiment source checkpoint is
converted to the runtime safetensors format using `torch.load(weights_only=True)`;
the source checkpoint and its fingerprint are retained. Package installation is
the separate pip command above.

Pinned upstream revisions:

| Task | Model | Revision |
| --- | --- | --- |
| Emotion | `SamLowe/roberta-base-go_emotions` | `d75048347613a25d77de8cf6412eaae9fa7b26be` |
| Sentiment | `cardiffnlp/twitter-roberta-base-sentiment-latest` | `3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7` |

Setup may access Hugging Face to download missing files. App analysis reads
cached weights locally; it does not send posts to a hosted model. Both current
classifiers target English. Their original model cards describe the
[multi-label emotion model](https://huggingface.co/SamLowe/roberta-base-go_emotions)
and [Twitter sentiment model](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest).

The small native EXE requires this installed checkout; it is not a portable
bundle of Python, models, and data. `build-exe.cmd` rebuilds it using the Windows
.NET Framework compiler. For development, use
`.venv-workbench/Scripts/python.exe src/desktop_app.py`.

## Appearance and design

Use **Appearance** in the top bar to choose **Light**, **Dark**, or **Follow
Windows**. The app saves your choice on this device and restores it at launch.
Follow Windows also responds when the Windows color preference changes. Changing
appearance keeps the current page and unfinished form entries in place.

The interface follows the **Research desk** concept, previewed in the Appearance
panel: a calm dashboard with neutral surfaces, violet highlights, and a consistent
layout in both themes. Navigation groups the overview under Workspace; importing,
analysis, exploration, review, and findings under Research; and model testing and
experiment evidence under Validation. The main process is **Import → Analyze →
Review → Explain**. Emotion labels and sentiment remain separate throughout.

## 1. Add your data

Open **Datasets**. Choose a file, paste text, or click **Paste from clipboard**.
Clipboard reading happens only when requested. Select a paste format, provide a
dataset name, and click **Preview data**.

Supported inputs are **UTF-8 CSV, TSV, JSONL, and TXT**, up to **20 MB and
100,000 rows**. XLSX and binary Excel files are not supported. CSV/TSV need a
header with distinct nonblank column names. JSONL has one JSON object per line.
TXT treats each line as a post. For pasted CSV, quote text containing commas or
newlines as usual.

Choose **Which column contains the post text?** from the preview. Optionally map
a user identifier; this is useful for keeping users separate during review.
Check the first rows, then click **Save dataset version**. Import does not
automatically run analysis. There is no assumption that repository corpora are
your research dataset.

The saved canonical data contains the selected exact text, post ID, source row,
and optional user ID. Source post IDs are retained as strings. Repeated/blank
source IDs receive unique export-row identities while their original IDs are
kept separately. Previous machine-label columns are excluded from canonical
data so they cannot silently become reference labels. Blank text remains in
the dataset and is marked unusable during analysis.

Saving the same normalized name again creates **v2, v3, ...**, even if the
content is identical. Each import has a new immutable ID. Its folder contains
the exact UTF-8 source copy, `dataset.csv`, and `manifest.json` with selected
columns, source/data hashes, row counts, dataset version, and app version. The
library checks hashes when reading the saved data. To correct text or mapping,
import another version instead of editing stored files.

## 2. Run the pretrained models

Open **Run analysis**, select a dataset version, and optionally name the run.
Emotion labeling is included; **Also analyze sentiment** controls a separate
classifier. Choose the emotion threshold and click **Run analysis**.

**Emotion model** offers the current RoBERTa baseline and the installed
ModernBERT research candidate. The latter improved several measures in the
published development comparison; your own data still needs human review.
Under **Emotion threshold settings**, choose a manual shared cutoff or inspect
a saved profile for that model. Switching models clears the selected profile.
The application refuses profiles from different model files and records the
chosen model, profile fingerprint, and cutoffs with the run. RoBERTa with a
manual threshold remains the default. See the
[refinement comparison](THRESHOLD_REFINEMENT.md) for results and tradeoffs.

**Emotion threshold settings** also offers saved provisional research profiles
when available. Select one explicitly to use its per-emotion cutoffs, or leave
**Manual shared threshold** selected. Inspect the saved cutoffs first. The app
does not automatically replace the default: the current GoEmotions refinement
raises aggregate micro F1 slightly while reducing macro F1, mixed-emotion
recovery, and automatic acceptance versus historical 0.3. It is an experimental
tradeoff, not a generally better mapper. See
[THRESHOLD_REFINEMENT.md](THRESHOLD_REFINEMENT.md) for all five methods and limits.

Profiles are checked against the installed emotion model's provenance; a
mismatch is refused. New runs copy the profile hash and exact cutoffs into their
configuration. The selector shows the latest complete profile for each
reference/model pair, while earlier experiments remain on disk. Saved results
are not changed by selecting a profile for a new analysis.

**This performs inference, not model training.** It applies downloaded weights
to existing posts. It does not learn from your imported data, predict future
posts, or establish that the labels are correct.

The emotion model scores 27 categories plus neutral independently. Multiple
emotions may pass the threshold. No passing label means uncertain; neutral plus
an emotion produces a review flag. Blank text is invalid. Emotion text longer
than 256 tokens is truncated and flagged for review. Scores describe model
support, not intensity or calibrated certainty about the author.

The sentiment model separately scores negative, neutral, and positive tone and
chooses the highest. It applies its published username/link placeholder
preprocessing and records the exact scored text. Its own 256-token truncation
flag is retained. Sentiment is not a collapse of the emotion vocabulary:
negative sentiment and a fear label answer different questions.

Progress runs in the background while you navigate the app. Only one analysis
runs at a time. **Keep the app open until it completes.** Closing it stops the
process; interrupted runs are incomplete and do not appear as completed
experiments. This version has no persistent background service or automatic
resume. After completion, use **Open results** or **Read findings**.

Each analysis writes a new folder in `runs/analyses/`. It keeps exact post text,
identities and hashes, emotion and optional sentiment scores, selected labels,
review flags, summary counts, settings, dataset manifest, app version, model
revisions/file hashes, tokenizer/task settings, and package versions. Failed
runs retain failure information. Changing a setting creates another run rather
than replacing old predictions.

## 3. Inspect and explain the outputs

**Overview** summarizes the active experiment. **Data explorer** filters posts
and shows detailed scores and sentiment. Existing project experiments remain
available for comparison, but a historical pilot is not an analysis of your
imported data.

**Model lab** explores a different threshold on saved emotion scores. This
sandbox does not replace predictions or alter an evaluation. To adopt a setting,
run a new analysis. Co-occurrence counts describe selected labels appearing
together; they do not establish psychological similarity.

The sandbox slider uses one shared cutoff. It does not adjust each learned
cutoff in a saved per-emotion profile. The
[refinement experiment](THRESHOLD_REFINEMENT.md) keeps out-of-fold comparisons
separate from a provisional profile refitted on all development rows.

**Findings & notes** turns the saved outputs into deterministic descriptions:
how many posts were analyzed, frequent emotion labels, selected-label overlap,
sentiment counts when available, and posts needing review. Each observation
includes count-based evidence and limitations. Emotion percentages can overlap
and need not add to 100%. No sentiment output means unavailable; the app does
not infer it from emotion labels.

Automatic findings are templates based on stored counts, not newly generated
discoveries or an LLM interpretation. They do not measure accuracy, establish
causes, diagnose anyone, or prove how authors feel. Read the underlying posts
before interpreting a pattern. Sarcasm, quotations, negation, mixed emotions,
language differences, and missing context can mislead the classifiers.

Add a title and your own evidence/uncertainty in **Research notes**. The sidebar
reviewer identity becomes the author. Notes are append-only and attached to a
completed analysis; adding a correction creates another note. Automatic
descriptions and authored observations remain distinct. **Export findings**
writes Markdown and JSON with the report, limitations, and notes beneath
`runs/workbench/exports/`, then shows the location.

## 4. Measure labeling quality with people

Create a blind review sample for an imported dataset and use **Label review**.
Choose a reviewer name, begin with development examples, and select every
supported emotion. Use neutral alone, unclear for missing context, and invalid
for unusable text. Model suggestions are hidden. Each save preserves another
revision; different reviewers start with separate blank judgments.

Start by selecting **Prepare 30-post practice round** in Label review or
Experiments. The app creates a separate, repeatable sample of up to 30 unique
nonblank development posts using the selected source and versioned rubric.
Reserved test posts never enter this practice round. Human labels and notes
start blank, even if the source already contains judgments. Repeating the same
request returns the same practice artifact rather than another queue.

Use the searchable in-app definitions from `src/twitterfeel/labeling/annotation_rubric.json`, record
an evidence phrase, and note difficult distinctions. You can be the first
reviewer now and add a second independent reviewer later. No AI-generated labels
are used as the reference. Thirty practice judgments are not a reliable accuracy
estimate. Revise and freeze the rubric after practice, then continue the larger
development review. Practice judgments stay separate from the original review
sheet and are not automatically promoted into the full benchmark.

See [EMOTION_LABELING.md](EMOTION_LABELING.md) for all label definitions and the
annotation rubric. Prefer independent reviewers and adjudication. A single
reviewer's draft is only a preliminary reference, and existing review sheets
are not automatically created from model answers.

In **Experiments**, compare saved predictions with reviewed examples. Metrics
include per-label precision/recall/F1/support, macro/micro F1, exact label match,
and overlap. Uncertain model predictions count as misses; unresolved human
judgments are reported separately. Evaluations keep reference snapshots and
source hashes, and error inspection identifies missed/extra labels.

**Compare thresholds on reviewed development posts** evaluates the active run's
stored scores at shared thresholds 0.2 through 0.8 in steps of 0.1. Pending,
unclear, and invalid human judgments are counted separately; without reviewed
development examples, the comparison remains not ready. It reports precision,
recall, micro F1, exact match, per-label support, and mixed-emotion recovery.
Its provisional recommendation maximizes development micro F1, resolving ties
by proximity to 0.5 and then the higher threshold. Saved settings are unchanged.

The comparison also reports 95% micro-F1 intervals from 200 bootstrap resamples.
Complete user identities enable resampling users rather than individual posts.
Older samples can recover missing user IDs only after verifying the source file
fingerprint and every row/post/text identity. If identities remain incomplete,
the report warns and falls back to posts. This cannot account for multiple
posts from the same person. Small samples and rare labels receive warnings;
intervals do not include threshold-selection or annotation disagreement
uncertainty. Practice-round results are explicitly identified as provisional.

Tune model/threshold choices on development posts. Reserve test posts for final
assessment. Compare runs on the same posts, reference labels, split, and
vocabulary. **Accuracy on your data has not been demonstrated merely by running
the models.** Synthetic execution checks show whether code paths agree; they
are separate from human-reference accuracy evidence. The current emotion
[diagnostic examples](LABELING_CHECK.md) expose missed mixed emotions, a sarcasm
error, and ambiguous text assigned neutral.

## Version history and local persistence

The sidebar version button opens **About**, including the release/version
record from `app_version.json`. Version **0.2.0** adds user dataset intake,
versioned imports, separate sentiment analysis, background runs, and research
findings/notes. Dataset manifests and new analyses record their application
version. Updating the program does not update earlier predictions.

| Location | Saved work |
| --- | --- |
| `runs/workbench/datasets/` | Immutable source copies, canonical datasets, manifests |
| `runs/analyses/` | New analysis configurations, predictions, and summaries |
| `runs/workbench/reviews.sqlite` | Review revisions keyed by source, post, hash, reviewer |
| `runs/workbench/calibration/` | Blind development-only practice sheets and source/rubric fingerprints |
| `runs/workbench/benchmarks/` | Provisional development threshold reports and human-reference snapshots |
| `runs/workbench/refinements/` | Grouped OOF method comparisons, decisions, and provisional threshold profiles |
| `runs/workbench/findings.sqlite` | Authored observations attached to analysis runs |
| `runs/workbench/exports/` | Exported review CSVs and findings Markdown/JSON |
| `runs/workbench/evaluations/` | Human-reference snapshots and saved metrics |
| `runs/workbench/preferences.json` | Validated theme/reviewer preferences |
| `runs/model_cache/` | Model files, source metadata, and setup manifests |

Back up `runs/` to preserve research work; Git ignores it. The project folder
also contains WebView state and startup logs under `runs/workbench/`. Reviewer
names separate work but are not accounts or authentication.

## Checks

The native desktop app is the supported interface. The labeling diagnostic uses
the same analysis engine as the desktop bridge and dataset analysis jobs.

```powershell
.venv-workbench/Scripts/python.exe -m unittest discover -s tests -v
.venv-workbench/Scripts/python.exe src/check_labeling.py --out runs/labeling_diagnostic_desktop_v2
TwitterFeel.exe --smoke-test runs/workbench/native-check.json
```

The native smoke check initializes the actual native window and Python bridge,
exercises interface/model paths, and checks the app's process family for
listening TCP ports. Reports and app-render previews are local artifacts. Unit
tests use temporary workspaces rather than adding real research annotations.
