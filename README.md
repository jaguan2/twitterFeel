# TwitterFeel Research Workbench · v0.2.0

TwitterFeel is a local Windows research app for **labeling emotions and sentiment
in existing posts**. Import your own text, run pretrained models, inspect their
answers, review examples, and save observations with the evidence.

**Open `TwitterFeel.exe` in this project folder.** It opens its own desktop
window using WebView2 and a direct Python bridge. The desktop app starts no
HTTP server or browser tab. Keep the EXE with this checkout and its environment;
it is a launcher for the installed app, not a portable installer.

## First session

1. **Datasets:** choose a file, paste data, or use **Paste from clipboard**.
   Preview it, choose the column containing the post text, optionally map a user
   identifier, and save a dataset version. Supported inputs are UTF-8 CSV, TSV,
   JSONL, or TXT with one post per line, up to **20 MB and 100,000 rows**.
2. **Run analysis:** choose that saved version, a run name, and an emotion
   threshold. Keep **Also analyze sentiment** selected to run the separate
   sentiment classifier. Start the analysis and follow its progress. Keep the
   app open until the run completes.
3. **Data explorer:** read posts alongside the selected emotions, model scores,
   sentiment, and review flags. Several emotions can appear in one post.
4. **Findings & notes:** read plain-English descriptions of the counts and their
   limits. Add your own titled observations under your reviewer name, then export
   the report and notes. Automatic descriptions and authored observations remain
   separate.
5. **Label review / Experiments:** build an independent human reference and
   compare the model's answers against it before claiming labeling quality.

Importing saves data; it does not run a model. Existing repository experiments
remain available for inspection, but they are not automatically imported as your
research data. Saving the same dataset name again creates another immutable
version. Earlier imports and completed analyses stay available.

For a quick TXT example, paste these lines and select TXT:

```text
I'm excited to move, but scared to leave everyone behind.
The meeting starts at three.
I'm not angry; I'm disappointed.
```

The first example illustrates why multiple emotion labels may be useful. It
does not guarantee the current model will detect both emotions.

## Setup on another checkout

Use Python 3.12 and the Microsoft Edge WebView2 Runtime. From the repository root:

```powershell
python -m venv --system-site-packages .venv-workbench
.venv-workbench/Scripts/python.exe -m pip install -r requirements-desktop.txt
.venv-workbench/Scripts/python.exe src/setup_models.py --task all
```

The last command installs both pinned pretrained models and reuses complete
matching downloads. It does not install Python packages. Use `--task emotion`
for emotion-only analysis or `--task sentiment` to add sentiment later. Model
setup needs Internet access when files are missing; analysis uses local files
and processes post text locally.

If needed, rebuild the native EXE with `build-exe.cmd`. For development, run
`.venv-workbench/Scripts/python.exe src/desktop_app.py` directly.

## What the models mean

Published human annotations provide a larger benchmark than the initial synthetic
examples. See [public datasets](docs/PUBLIC_DATASETS.md) for GoEmotions and the
Twitter-specific SemEval comparison. The **Experiments** page keeps published
reference results separate from your own blind review. A model's training
dataset can support an in-domain reproduction; it does not establish accuracy
on your uploads. Threshold selection uses development data only.

The [refinement report](docs/THRESHOLD_REFINEMENT.md) compares finer cutoffs,
per-emotion cutoffs, and a ModernBERT research candidate on the same development
posts. In **Run analysis**, choose the emotion model and optionally its matching
saved threshold profile. RoBERTa with manual shared thresholds remains the
default. Model selection and profile fingerprints are saved with every new run.

For your own posts, begin with the 30-post practice round and the searchable
emotion definitions in **Label review**. No model suggestions become human
reference labels automatically.

| Output | Plain-English meaning |
| --- | --- |
| Emotion labels | Possible expressions such as joy, fear, sadness, or excitement; several labels may pass independently. |
| Neutral emotion result | Neutral passed and no emotion passed at the chosen threshold. This is still a model judgment. |
| Uncertain / needs review | No label passed, labels conflicted, or text was truncated. |
| Sentiment | A separate prediction of negative, neutral, or positive overall tone. It is not calculated by grouping emotion labels. |
| Support score | Strength of model support for a label, not emotional intensity or proven certainty about the author. |

Emotion uses [SamLowe's GoEmotions RoBERTa model](https://huggingface.co/SamLowe/roberta-base-go_emotions),
with 27 emotion categories and neutral. Sentiment uses
[Cardiff NLP's English Twitter RoBERTa model](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest).
Their different training tasks mean sentiment and emotion can disagree without
being directly contradictory.

**Running analysis applies already trained models. It does not train or improve
a model on your imported data.** Accuracy on your posts has not been
demonstrated. Mixed emotions, sarcasm, quotation, negation, language differences,
and missing context require human inspection. Counts and automatic findings
describe model behavior; they do not establish causes, diagnoses, or how a person
actually feels.

## Versioning and saved work

The app version is recorded in [`app_version.json`](app_version.json) and shown
in the sidebar/About dialog. Imports retain exact UTF-8 source bytes, column
mapping, row identity, content hashes, dataset version, and app version. Runs
save predictions, counts, settings, dataset provenance, model revisions/file
hashes, and package versions. Updating the app does not relabel older results.

| Location | Contents |
| --- | --- |
| `runs/workbench/datasets/` | Immutable imported dataset versions and source copies |
| `runs/analyses/` | New analysis runs, predictions, summaries, and configurations |
| `runs/workbench/reviews.sqlite` | Separate reviewers' append-only label judgments |
| `runs/workbench/findings.sqlite` | Authored research observations |
| `runs/workbench/exports/` | Exported reviews and Markdown/JSON findings |
| `runs/workbench/evaluations/` | Human-reference snapshots and metrics |
| `runs/model_cache/` | Downloaded pretrained models and setup provenance |

These local artifacts are gitignored. Back up `runs/` to preserve your research.
The [workbench guide](docs/WORKBENCH.md) explains the complete workflow and
remaining limitations; [the annotation guide](docs/EMOTION_LABELING.md) defines
the emotion vocabulary and human-review process.

## Command-line and historical work

The current code is grouped by responsibility:

| Files | Purpose |
| --- | --- |
| `src/desktop_app.py`, `desktop_api.py`, `desktop_ui/` | Native window, app actions, interface |
| `src/analysis_engine.py`, `analysis_jobs.py` | Shared local models and background analysis |
| `src/research_workspace.py`, `review_protocol.py`, `workbench_data.py` | Datasets, findings, review persistence, and practice rounds |
| `src/benchmark.py`, `public_benchmark.py`, `threshold_tuning.py`, `compare_models.py` | Reference evaluation and model comparisons |
| `src/check_labeling.py`, `check_research.py`, `tests/` | Diagnostics and automated checks |
| `src/preprocess*.py`, `train*.py`, `evaluation.py`, `twitter_feel_plus/` | Historical forecasting and sadness-pattern research |

`src/label_posts.py` still provides `sample`, `label`, and `evaluate` commands for
the emotion-only workflow. The desktop application is the single supported UI;
`build-exe.cmd` rebuilds its launcher.

Earlier research predicted a future post's machine-generated label from posting
history. It answers a different question. Reproduction material remains in
[V4](docs/V4.md), [the experiment log](docs/EXPERIMENTS.md),
[Reddit ingestion](docs/REDDIT.md), and [the sadness-pattern subproject](twitter_feel_plus/README.md).
Its TensorFlow/LSTM dependencies are in `requirements-forecasting.txt`; existing-post
analysis needs no temporal windows, embeddings, or LSTM training.

## Checks

```powershell
.venv-workbench/Scripts/python.exe -m unittest discover -s tests -v
TwitterFeel.exe --smoke-test runs/workbench/native-check.json
```

The [actual-model diagnostic](docs/LABELING_CHECK.md) documents execution parity
and concrete labeling weaknesses. Synthetic execution checks are separate from
accuracy measurements against human reference labels.
