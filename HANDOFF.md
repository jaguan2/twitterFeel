# TwitterFeel handoff

Updated: 2026-10-16 (America/New_York).

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

Local rendered previews (generated artifacts, not committed assets):

- [Light overview](runs/workbench/native-appearance-check-overview-light.png)
- [Dark overview](runs/workbench/native-appearance-check-overview-dark.png)
- [Appearance and concept](runs/workbench/native-appearance-check-appearance-dark.png)

## Code map

| Location | Responsibility |
| --- | --- |
| `src/desktop_app.py` | Native window, inline assets, isolated native smoke test and rendered captures |
| `src/desktop_api.py` | Allowlisted bridge actions, preferences, datasets, analysis, reviews, findings, experiments |
| `src/desktop_ui/` | Buildless JavaScript, HTML, CSS; no remote UI assets |
| `src/DesktopLauncher.cs`, `src/build_version.py`, `build-exe.cmd` | Windows launcher and version metadata |
| `src/analysis_engine.py`, `src/analysis_jobs.py` | Shared local inference and background batch jobs |
| `src/research_workspace.py`, `src/review_protocol.py`, `src/workbench_data.py` | Import/version storage, observations, review persistence and practice rounds |
| `src/annotation_rubric.json`, `src/label_schema.py` | Review definitions and emotion decision rules |
| `src/benchmark.py`, `src/public_benchmark.py` | Human-reference and published-reference evaluation |
| `src/threshold_tuning.py`, `src/compare_models.py` | Grouped development comparisons and saved research profiles |
| `src/check_labeling.py`, `src/check_research.py`, `tests/` | Execution diagnostics and automated checks |
| `src/preprocess*.py`, `src/train*.py`, `twitter_feel_plus/` | Retained historical forecasting / sadness-pattern research |

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

Use Python 3.12 and Microsoft Edge WebView2 Runtime. Existing local environment:
`.venv-workbench` (Python 3.12.10). See [WORKBENCH.md](docs/WORKBENCH.md) for full
setup, model provisioning, and workflow instructions.

```powershell
.\.venv-workbench\Scripts\python.exe -m pip install -r requirements-desktop.txt
.\.venv-workbench\Scripts\python.exe src/setup_models.py --task all
.\.venv-workbench\Scripts\python.exe src/desktop_app.py

.\.venv-workbench\Scripts\python.exe -m unittest discover -s tests
node --check src/desktop_ui/app.js
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

In this environment, sandboxed WebView2 startup fails; native EXE verification
has required an approved elevated tool invocation. This is a test-environment
constraint, not a reason to replace the native app with a local server.

## Data and Git precautions

The checkout contains substantial outgoing work, including new files and prior
cleanup deletions. No commit or push was made during this implementation.
Review the full Git status before staging; do not discard existing changes.
The last inspected commit was `4b7613f` (pipeline separation / evaluation work).

`runs/`, the EXE, environments, model caches, and large intermediate arrays are
gitignored. They exist locally and do not travel with a source-only checkout.
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
