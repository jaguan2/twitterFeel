# Using human-labeled public emotion data

Pre-labeled data lets us test the model sooner, without asking one person to label thousands of posts first. We still need a smaller review of our own posts: a model can perform differently when the writing, topics, or label definitions change.

## Two different checks

**GoEmotions is the vocabulary match.** Our current emotion model was trained on this Reddit dataset, which uses 27 emotions plus neutral and allows several labels per comment. Its development split can help compare thresholds using the same vocabulary. This is an in-domain check of the existing model; it does not establish accuracy on our Twitter data. Keep its test split reserved after development decisions. See the [model author's description](https://huggingface.co/SamLowe/roberta-base-go_emotions) and [original dataset repository](https://github.com/google-research/google-research/tree/master/goemotions).

**SemEval-2018 E-c is the Twitter check.** Its English emotion classification dataset has 6,838 training, 886 development, and 3,259 test tweets. It supports multiple emotions, but its eleven categories differ from GoEmotions. These split counts come from Table 3 of the [official task paper](https://aclanthology.org/S18-1001/).

| SemEval categories | Relationship to the current model |
| --- | --- |
| anger, disgust, fear, joy, love, optimism, sadness, surprise | Eight exact-name shared categories; matching names do not guarantee identical annotation rules |
| anticipation, pessimism, trust | No direct current-model category; retained in the reference |

SemEval's original reference treats eleven negative indicators as neutral/no emotion. However, a tweet expressing only trust would have no positive label in the eight-category projection. Calling it neutral would erase its original annotation. The adapter therefore preserves native labels, shared labels, unsupported labels, and separate all-negative indicators. It does not create GoEmotions `human_labels` or merge this reference into a 28-label benchmark. The [official dataset instructions](https://competitions.codalab.org/competitions/17751#learn_the_details-datasets) describe the native binary labels.

## Availability and terms checked on October 15, 2026

The official full archive and separate English development download both returned HTTP 406 during this session, including the author's alternative host names. **No actual SemEval reference has been downloaded or prepared, and no model score has been produced for it.** Synthetic adapter tests are separate from dataset/model evidence. We have not substituted an unofficial mirror.

The [official terms](https://competitions.codalab.org/competitions/17751#learn_the_details-terms_and_conditions) limit data use to scientific/research purposes, prohibit sharing or redistribution to third parties, and require citation of the task paper. This is not a verified unrestricted open-data license. The paper's publication license and an evaluation script's software license do not replace the dataset terms. Keep downloaded data and derived references local, outside distributable application packages and committed source.

## Preparing the official development reference

### GoEmotions: available and evaluated

```powershell
.venv-workbench/Scripts/python.exe src/public_benchmark.py --revision 7951440944924ac61b6e2f9a2a3c715834005c40 --evaluate
```

The completed run evaluated all **5,426 official development examples**, retaining
all published annotation combinations and repeated-text frequencies. The **5,427
test examples were not scored**. Published references are separate from personal
review records. Original files, license, pinned source URLs, model fingerprints,
predictions, and the report are retained under `runs/public_benchmarks/`.

| Shared threshold | Precision | Recall | Micro F1 | Exact label-set match | All reference labels recovered on mixed-emotion posts |
| --- | --- | --- | --- | --- | --- |
| 0.50 | 0.686 | 0.511 | 0.586 | 47.4% | 9.4% |
| 0.30 | 0.584 | 0.636 | 0.609 | 44.9% | 19.5% |

The lower threshold retrieves more reference emotions but also adds more incorrect
labels. Full recovery allows extra labels, so it must be read alongside precision
and exact match. These are development results, not evidence of improved accuracy
on our tweets; the app's default threshold remains unchanged. The 0.30 micro-F1
post-bootstrap interval is approximately 0.600–0.619. It omits threshold-selection
uncertainty and cannot account for repeated authors because these reduced splits
do not include user IDs.

The source includes 174 development examples with both neutral and emotion
annotations, three repeated development texts, and ten text hashes shared between
development and test. The public reference policy preserves these facts and
reports them rather than rewriting human labels. Your own review rubric continues
to keep neutral exclusive. Do not treat the official test set as a pristine final
benchmark: the model author's card already reports test-based threshold tuning.

A compact result summary is kept in
[`metrics/goemotions_development_v1.json`](../metrics/goemotions_development_v1.json).

### SemEval: prepared adapter, download unavailable

Once the author's archive is available, the adapter can download and prepare it:

```powershell
.venv-workbench/Scripts/python.exe src/semeval_data.py --download --archive runs/public_datasets/semeval2018/raw/SemEval2018-Task1-all-data.zip --out runs/public_datasets/semeval2018/development-v001
```

For an already downloaded official archive, omit `--download`. Supply `--expected-sha256 <fingerprint>` when reproducing a particular downloaded version. The command refuses to overwrite an archive or prepared version. It accepts either the official full archive or the separate English development ZIP, using a known development member name.

Preparation saves the original archive, exact development TSV, canonical `reference.jsonl`, and a manifest containing source/member/output hashes, source URLs, app version, native label counts, the explicit eight-category mapping, and terms provenance. Only the known development member and recognized notices can be opened. Training and test labels remain unopened; no arbitrary ZIP paths are extracted and no downloaded scripts execute.

This preparation does not evaluate a model. Before a Twitter comparison is implemented, define eight-category metrics explicitly, keep all native annotations for audit, inspect unsupported-only examples, and report the limited label scope alongside results. Freeze development choices before using the reserved test split.
