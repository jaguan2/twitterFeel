# Code organization and development

The implementation is an installable Python package under `src/twitterfeel/`.
The existing `src/*.py` commands are compatibility entry points. They delegate
to the package rather than maintain a second implementation.

## Responsibilities

```text
src/twitterfeel/
  __main__.py               native application entry point
  common/                   workspace configuration, resources and provenance
  desktop/                  native window, bridge, build metadata and UI assets
  inference/                classifiers, batch jobs and pinned model setup
  labeling/                 label vocabulary, rubric and labeling CLI
  workspace/                imported datasets, reviews and research findings
  evaluation/               benchmarks, thresholds and model comparisons
  diagnostics/              actual-model consistency and workflow checks
  research/
    forecasting/            preserved six-class/binary forecasting experiments
    sadness/                preserved rolling sadness-pattern analysis
tests/                      tests grouped by the same responsibilities
tools/                      repository utilities, including the PDF builder
```

Historical research is inside the installed package so it can use normal
imports and module commands. It is separated from current application code,
and current inference does not import the historical forecasting vocabulary.
Shared hashing algorithms live in `common/provenance.py`. The six-class and
current multi-label schemas remain distinct.

`.gitattributes` fixes Python/interface source line endings at LF and Windows
batch commands at CRLF, preventing platform conversion from obscuring code diffs.

The native bridge calls inference and workspace services. Model evaluation is
separate from interactive inference. The existing buildless JavaScript/CSS UI
remains intact; this reorganization does not introduce a frontend framework or
change model rules, training objectives or evaluation splits.

## Install and run

Use Python 3.11 or 3.12; 3.12 remains the recommended desktop environment.
Run installation commands from the repository root:

```powershell
python -m venv .venv-workbench
.venv-workbench/Scripts/python.exe -m pip install -e ".[desktop]"
.venv-workbench/Scripts/python.exe -m twitterfeel.inference.setup_models --task all
.venv-workbench/Scripts/python.exe -m twitterfeel
```

Package dependencies and supported Python versions are in `pyproject.toml`.
The requirements files are compatibility aliases that install the same project.
Optional extras are `desktop`, `forecasting`, `reporting`, and `dev`. For example,
install `.[forecasting]` for the preserved TensorFlow/ingestion workflows, or
`.[reporting]` for PDF generation. Model setup can download pinned files;
package installation, unit tests and structural checks do not download models.

Use `python -m twitterfeel.labeling.label_posts --help` for the emotion-labeling
CLI. Installed console commands are `twitterfeel`, `twitterfeel-label`, and
`twitterfeel-setup-models`. The historical V4 command is
`python -m twitterfeel.research.forecasting.train_v4 --help`.

## Source files and writable workspaces

Code, UI assets and the annotation rubric are read from the installed package.
Changing the writable workspace does not change where application code loads.
All default workspace locations are resolved centrally:

1. The desktop `--workspace PATH` argument takes precedence.
2. Otherwise use `TWITTERFEEL_HOME` if it is set.
3. In a source checkout, retain that checkout as the default workspace.
4. For a wheel installed outside a checkout, default to `%LOCALAPPDATA%/TwitterFeel`
   on Windows, or `~/.local/share/twitterfeel` when LOCALAPPDATA is unavailable.

Set `TWITTERFEEL_HOME` before starting any CLI to use the same workspace for
model setup, diagnostics and research. Importing path/configuration modules
does not create directories. Services initialize storage when they are used;
historical commands initialize their output folders when execution starts.

The existing workspace-specific `src/annotation_rubric.json` override remains
supported for existing research/calibration fixtures. Otherwise the packaged
rubric is used. Its actual bytes determine the recorded hash.

`data/`, `metrics/`, `models/`, `runs/`, `output/`, and
`twitter_feel_plus/results/` have not been relocated. Legacy root-level arrays
also remain in place. Do not move or regenerate historical research artifacts
as a side effect of an organizational refactor. Newly recorded source hashes
identify the implementation modules, including shared provenance code, rather
than just the compatibility entry points.

## Tests, checks and builds

Use a separate development environment if desired:

```powershell
python -m venv .venv-dev
.venv-dev/Scripts/python.exe -m pip install -e ".[dev]"
.venv-dev/Scripts/python.exe -m unittest discover -s tests -v
.venv-dev/Scripts/python.exe -m ruff check src tests tools twitter_feel_plus
node --check src/twitterfeel/desktop/assets/app.js
.venv-dev/Scripts/python.exe -m build
.venv-dev/Scripts/python.exe tools/check_wheel.py
```

Tests import the installed package without changing `sys.path`. Existing
research/model tests use temporary fixtures, with real-model/native diagnostics
remaining separate. The initial lint rules check correctness without broadly
reformatting historical experiment code. CI runs checks on Windows with
Python 3.11 and 3.12 and builds the distribution.

`build-exe.cmd` still builds the checkout-based Windows launcher with the
workbench environment. The launcher now invokes `pythonw -m twitterfeel`;
it remains a launcher requiring this checkout and environment, not a portable
installer. Native model/window smoke checks need installed models and WebView2.

Keep `pyproject.toml`'s package version and `app_version.json` synchronized;
the package suite checks their agreement. Dependency ranges are not a fully
locked reproduction environment. Continue retaining exact package/model
provenance with research runs.

## Compatibility and historical records

Old commands such as `python src/desktop_app.py`, `python src/label_posts.py`,
and `python src/train_v4.py` continue to work. The sadness scripts under
`twitter_feel_plus/` also forward to the package. Those wrappers contain no
scientific implementation. Normal package imports are preferred for new code.

Historical Markdown commands, the dated study PDF and its source-hash snapshot
are preserved as records of their original revisions. See this document for
current implementation locations. The refactor adds no new model-accuracy
result and does not resolve the original experiment's evaluation provenance.
