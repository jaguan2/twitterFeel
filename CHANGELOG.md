# Release history

## Code organization - 2026-11-02

Package the implementation under `src/twitterfeel/`, grouping desktop,
inference, labeling, workspace, evaluation, diagnostics and historical research.
Old script paths delegate to the package. Tests use installed package imports.
`pyproject.toml` declares dependencies/extras and console entry points; CI checks
the grouped test suite, syntax/lint and distribution builds.

Bundle the interface and rubric with the package, centralize writable workspace
configuration, and remove directory creation during path-module imports.
Research source hashes follow implementation files after relocation. Data,
saved metrics, model weights and existing research reports remain in place.
This maintenance change adds no accuracy or reproduction claim.

## Workspace cleanup — 2026-10-16

Retired the duplicate browser UI, its dependencies, launch scripts, and obsolete
UI tests. Desktop demonstrations and dataset jobs now share one analysis engine;
the labeling diagnostic checks that native backend. The default requirements
install the desktop app; historical training dependencies are separate.

## 0.2.0 — 2026-10-15

The research workbench opens in a native Windows window. Import and version your own datasets, run emotion and sentiment analysis separately, inspect results, and save observations with the supporting run. Each analysis records its dataset, model files, settings, and application version.

Emotion labels remain model suggestions. Human review and held-out evaluation are needed before claiming improved accuracy.

Benchmark preparation now includes a blind 30-post practice round, a versioned
emotion rubric, development-only threshold comparisons, mixed-emotion recovery,
and bootstrap intervals. Published human-labeled GoEmotions references remain
separate from personal review, with explicit training-overlap limitations.

Grouped development comparisons now cover finer shared thresholds, individual
emotion cutoffs, and a pinned ModernBERT candidate. The app offers explicit
model selection and matching provisional profiles, records the chosen settings,
and retains the original default. Paired results and mixed-emotion tradeoffs
are available in Experiments.

Threshold refinement now compares five decision rules using five-fold
cross-validation grouped by identical development text. The joint-micro
per-label profile improves micro F1 from 0.609231 to 0.614223 versus the
historical 0.3 comparator, but reduces macro F1 and mixed-emotion recovery.
No candidate meets the balanced-improvement criterion. The app offers the
profile only as an explicit provisional choice, checks its model provenance,
and records its hash and exact cutoffs in new analysis runs. Existing defaults
and results are preserved. Official test data is not scored or used to select
thresholds. Full results and methodological limits are in
[the refinement record](docs/THRESHOLD_REFINEMENT.md).
