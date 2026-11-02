# TwitterFeel handoff

Updated: 2026-11-02 (America/New_York).

## Purpose and user priorities

TwitterFeel is now a local Windows research app for labeling **existing posts**
with multiple emotions and, separately, overall sentiment. Future-post prediction
is historical research and is outside the current product focus.

The user wants a clean application that accepts their datasets or clipboard
content, makes analysis understandable, supports model testing, and explains
findings in plain language. They referenced DefTechLink's dashboard as a visual
direction. They will provide the first human reviews; a second reviewer can be
added later. Published labeled datasets supplement that process.

Running analysis applies pretrained classifiers. It does not train or improve
their weights on imported posts. Threshold refinement changes which labels pass,
not the underlying model scores or their probability calibration.

## Original project and reproduction context

The user supplied the original proposal presentation and final project report
on 2026-10-21. The proposal explored mental-health risk; the final report instead
predicted the next tweet's six-class, machine-generated emotion from history.
Later forecasting experiments attempted to reproduce its approximately 0.82
result. Current existing-post labeling is a separate task.

The [historical review](docs/HISTORICAL_REVIEW.md) records the source comparison,
code map, and checks. The report's confusion matrix supports 0.8233 accuracy and
0.8182 weighted F1, but the original evaluation provenance remains unresolved.
Its unusually balanced evaluation counts raise questions about resampling or
different source artifacts. The preserved notebook has a reproducible 29-column
versus 28-value assignment error and incorrect remainder slicing; its accessible
history contains no original training/SMOTE/split implementation. Code or
evaluation issues are credible explanations to investigate, not a proven cause
of the reported score. Do not describe later runs as exact reproductions or use
the historical score as a verified benchmark for the current app.

## Dated study record

The user requested a chronological PDF of findings and limitations on 2026-10-21.
The editable master is [STUDY_PROGRESSION.md](docs/STUDY_PROGRESSION.md), with
the generated [PDF](output/pdf/twitterfeel-study-progression.pdf) and a source
hash manifest under `output/pdf/`. Edition 1 backfills the original documents,
forecasting experiments, sadness-pattern study, V4, labeling/public benchmarks,
threshold/model comparisons and the historical audit.

Add future findings as dated entries in the master, separating event dates from
archive/reconstruction dates and identifying proposals, synthetic checks,
development results and held-out evidence. Record later corrections explicitly.
Rebuild with `python tools/build_study_progression.py` (requires `reportlab`),
then inspect rendered pages. The PDF is generated from the Markdown master;
keep both synchronized when updating the record.

## Packaging refactor - 2026-11-02

The user authorized the staged organization work. `pyproject.toml` now defines
package metadata, dependencies, optional desktop/forecasting/reporting/dev
extras, and console entry points. Install with `pip install -e ".[desktop]"`
from the repository root; start with `python -m twitterfeel`. The native EXE
launcher uses that module entry point. Existing script commands remain valid.

`TWITTERFEEL_HOME` configures the writable workspace for all commands; the
native app also accepts `--workspace PATH`. UI assets and the default rubric
are packaged resources. Imports do not create data/model folders. Historical
research implementations are grouped separately inside the package, keeping
one normal Python import tree. Data, metrics, models, study PDFs and existing
research artifacts were not relocated or regenerated.

Verification on 2026-11-02: the original 108-test baseline passed before the
migration; all 113 tests pass afterward on Python 3.11, including package,
resource and compatibility checks. Ruff correctness checks and JavaScript syntax
pass. Both source and wheel distributions build. The wheel was installed into
an isolated temporary target and checked outside the checkout for imports,
assets, rubric, fresh-workspace bridge actions and module entry points. The
Windows launcher compiled, and module invocation/quoted report-path forwarding
passed with a Python stub. That is not a real-model/native-window smoke test.
UI assets and the rubric match their committed bytes; 14 historical modules
retain identical function bodies after accounting for relocated imports and
explicit directory initialization. Data, model weights and saved metrics have
no changes.

A local `.venv-dev` reuses available system packages for structural verification;
it is not a newly locked research reproduction environment. The workbench
model cache and native runtime have not been restored. Preserve the dated
2026-10-16 runtime results as historical verification, not current native QA.

## Current application

- Version: `0.2.0`, defined in `app_version.json`.
- Launch `TwitterFeel.exe` from this checkout. It opens a native pywebview / Edge
  WebView2 window with bundled HTML/CSS/JavaScript and a direct Python bridge.
  There is no HTTP server or external browser tab.
- The EXE is a small launcher requiring the checkout, `.venv-workbench`, and
  local model caches. It is **not a portable installer**.
- Workflow: Datasets → Run analysis → Data explorer → Label review → Findings &
  notes. Model lab and Experiments expose testing and evaluation evidence.
- Imports support UTF-8 CSV, TSV, JSONL, and one-post-per-line TXT, with file,
  paste, and clipboard options. Limits: 20 MB and 100,000 rows. Users select text
  and optional user-ID columns before saving an immutable dataset version.
- Completed analyses retain their dataset identity, model provenance, settings,
  predictions, and summaries. Independent reviewers' judgments are append-only;
  model predictions never become human reference labels automatically.
- Emotion uses independent label scores: several labels may pass. Sentiment is
  a separate positive/neutral/negative classifier. Scores are model support,
  not emotional intensity or proof of someone's internal state.

## Latest design work

The **Appearance** button in the top bar opens preview cards for **Light**,
**Dark**, and **Follow Windows**. Preferences persist on disk and are applied
before the interface is painted. Follow Windows listens for system theme changes.
Changing appearance preserves the current page and unfinished form entries.
The picker supports keyboard navigation.

The implemented concept is **Research desk**: a pale neutral light canvas or deep
navy dark surfaces, restrained violet accents, readable evidence cards, and a
consistent layout. Navigation is grouped into Workspace, Research, and Validation.
The Appearance dialog also explains the concept and the process
**Import → Analyze → Review → Explain**.

Previously rendered previews (generated artifacts, not committed assets; absent
from the checkout inspected 2026-10-21):

- [Light overview](runs/workbench/native-appearance-check-overview-light.png)
- [Dark overview](runs/workbench/native-appearance-check-overview-dark.png)
- [Appearance and concept](runs/workbench/native-appearance-check-appearance-dark.png)

## Code map

The installable implementation is under `src/twitterfeel/`. Old `src/*.py`
commands are forwarding compatibility entry points, not duplicate code.
See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for dependency and workspace rules.

| Location | Responsibility |
| --- | --- |
| `src/twitterfeel/desktop/` | Native window, bridge, bundled interface assets and EXE metadata |
| `src/twitterfeel/inference/` | Local model inference, batch jobs and pinned model setup |
| `src/twitterfeel/labeling/` | Labeling CLI, multi-label schema and packaged rubric |
| `src/twitterfeel/workspace/` | Imports, findings, review persistence and practice protocol |
| `src/twitterfeel/evaluation/` | Human/public benchmarks, threshold and model comparisons |
| `src/twitterfeel/common/` | Writable workspace configuration, resource loading and hashes |
| `src/twitterfeel/diagnostics/` | Actual-model parity and research workflow checks |
| `src/twitterfeel/research/forecasting/` | Preserved forecasting pipelines, vocabulary and evaluation |
| `src/twitterfeel/research/sadness/` | Preserved sadness-pattern implementation |
| `src/DesktopLauncher.cs`, `build-exe.cmd` | Checkout-based Windows launcher |
| `tests/`, `tools/` | Grouped tests and repository utilities |

## Models and findings

| Task | Model | Pinned revision |
| --- | --- | --- |
| Default emotion | `SamLowe/roberta-base-go_emotions` | `d75048347613a25d77de8cf6412eaae9fa7b26be` |
| Research emotion candidate | `cirimus/modernbert-base-go-emotions` | `cb5ed17bc26c8cd1f017fabe700a5d723f825287` |
| Sentiment | `cardiffnlp/twitter-roberta-base-sentiment-latest` | `3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7` |

Emotion has 27 categories plus neutral. The app keeps RoBERTa and manual shared
thresholds as its default. ModernBERT and matching research profiles are explicit
choices in Run analysis; changing models clears the old profile selection.
Profile provenance must match the model. Both models use a 256-token limit;
truncation and conflicting/uncertain decisions require inspection.

GoEmotions comparisons used 5,426 development rows / 5,423 exact-text groups,
with the same five grouped folds. Published corpus revision:
`7951440944924ac61b6e2f9a2a3c715834005c40`.

| Development comparison | Micro F1 | Macro F1 | Mixed posts: all reference labels recovered |
| --- | ---: | ---: | ---: |
| RoBERTa, shared 0.30 | 0.6092 | 0.4963 | 19.52% |
| ModernBERT, shared 0.30 | 0.6195 | 0.5378 | 24.44% |
| ModernBERT, fitted per-label micro objective | 0.6211 | 0.5247 | 18.82% |

ModernBERT at 0.30 is the more useful next comparison for mixed emotions. Its
precision and exact-set match are slightly lower, and neutral/emotion conflicts
increase. The highest micro F1 sacrifices mixed-emotion recovery; it is not a
clear improvement for every objective. The RoBERTa per-label profile similarly
raises micro F1 to 0.614223 but lowers macro F1 and mixed recovery.

These are exploratory development comparisons on a dataset used to train both
models. They do **not** demonstrate independent accuracy on the user's tweets.
Official test data remains reserved and unscored by this work. Text grouping
does not remove author dependence or near duplicates. The initial 30-post human
practice round is development work, not a reliable accuracy benchmark. SemEval
support is prepared, but the official source download returned HTTP 406; no
completed SemEval result should be claimed.

Detailed methods and tradeoffs: [threshold refinement](docs/THRESHOLD_REFINEMENT.md).
Tracked summaries: `metrics/goemotions_development_v1.json`,
`metrics/emotion_threshold_refinement_v1.json`, and
`metrics/emotion_model_comparison_v1.json`.

Full local reports:

- `runs/workbench/refinements/goemotions_cv_v4/`
- `runs/model_comparisons/20261015T230413176038Z_df0b23f5/comparison.json`
- `runs/labeling_diagnostic_desktop_v1/diagnostics.json`

## Cleanup already completed

Retired the duplicate Streamlit UI (`src/workbench.py`), its UI tests,
`requirements-workbench.txt`, `start-workbench.cmd` / `.ps1`, `.streamlit/config.toml`,
and the redundant `build-exe.ps1`. The desktop app is the supported interface.
Desktop quick inference and batch jobs now share `AnalysisEngine`; the diagnostic
checks the CLI against the actual desktop backend.

`requirements.txt` installs the desktop requirements. Historical TensorFlow/LSTM
dependencies live separately in `requirements-forecasting.txt`. Historical
research code, useful reproduction docs, and saved results were retained.
Earlier cleanup removed obsolete `docs/FINDINGS.md` and `docs/handoff.md`; this
root handoff describes the current application.

## Setup and verification

Use Python 3.12 and Microsoft Edge WebView2 Runtime. The prior environment was
`.venv-workbench` (Python 3.12.10). On the checkout inspected 2026-10-21, that
environment, `TwitterFeel.exe`, and `runs/` are absent. Legacy embedding/window
arrays are at the repository root rather than the current `data/interim/`
locations. See [WORKBENCH.md](docs/WORKBENCH.md) for full setup, model provisioning,
and workflow instructions. No runtime setup or full test run was performed in
the historical review; the verification results below are from 2026-10-16.

```powershell
.\.venv-workbench\Scripts\python.exe -m pip install -r requirements-desktop.txt
.\.venv-workbench\Scripts\python.exe -m twitterfeel.inference.setup_models --task all
.\.venv-workbench\Scripts\python.exe -m twitterfeel

.\.venv-workbench\Scripts\python.exe -m unittest discover -s tests
node --check src/twitterfeel/desktop/assets/app.js
.\TwitterFeel.exe --smoke-test runs/workbench/native-check.json
```

Model setup may download missing files; normal inference reads local caches.
The standard `--task all` setup installs the baseline emotion and sentiment
models; candidate-model setup is described in the refinement guide.
`build-exe.cmd` rebuilds the C# launcher. UI source edits are loaded at launch
and do not themselves require rebuilding the EXE.

Latest verified state (2026-10-16):

- All **108 unit tests pass**; JavaScript syntax check passes.
- `runs/workbench/native-appearance-check.json`: `success: true`, no JavaScript
  errors, no listening TCP socket, persisted theme preferences, import drafts
  preserved, and no real research reviews modified. The smoke test uses a
  disposable workspace and exercises imports, both emotion models, matching
  profiles, sentiment, review controls, findings/export, and application pages.
- Actual-model diagnostic: 12 synthetic execution cases passed; CLI / desktop
  score differences were within tolerance. This is execution parity, not accuracy.

The prior environment recorded sandboxed WebView2 startup failures and used an
approved elevated invocation for native verification. That environment-specific
observation has not been retested on the current checkout.

## Data and Git precautions

On 2026-10-21, before the historical-review documentation edits, local `main`
and remote `origin/main` both pointed to `1ab5ece`; `git pull --ff-only` reported
up to date. The 73 apparent modifications were byte-verified as line-ending-only
differences. Repository-local `core.autocrlf=true` restored a clean working tree.
The review adds documentation changes; no commit or push was made. Review Git
status before staging and preserve any subsequent user changes.

`runs/`, the EXE, environments, model caches, and large intermediate arrays are
gitignored and do not travel with a source-only checkout. Their presence on a
previous machine does not establish that they exist in the current checkout.
Preserve and back up `runs/`, especially datasets, analyses, reviews, findings,
exports, benchmark artifacts, and model setup provenance. Preferences are in
`runs/workbench/preferences.json`; human reviews and authored findings have
separate SQLite stores. Do not delete data merely because it is ignored by Git.

## Recommended next work

1. Let the user review the actual light/dark app and refine density, navigation,
   and presentation based on that feedback. The current design is implemented,
   but it has not received final user design approval.
2. Complete the first human practice round and resolve rubric ambiguities,
   especially adjacent emotions, multiple emotions, neutral, and quotation.
3. Freeze an independent target-domain benchmark and evaluation plan before
   further model selection. Add the second reviewer later as requested.
4. Compare RoBERTa and ModernBERT on those references using micro/macro F1,
   per-label recall, mixed-emotion recovery, exact match, and review burden.
   Avoid optimizing micro F1 alone or repeatedly selecting on official test data.
5. If distribution to other computers is required, plan a real installer and
   model-provisioning flow. The current launcher alone is not sufficient.

Useful starting documents: [README](README.md), [workbench guide](docs/WORKBENCH.md),
[label definitions](docs/EMOTION_LABELING.md), [public data](docs/PUBLIC_DATASETS.md),
and [diagnostic findings](docs/LABELING_CHECK.md).
