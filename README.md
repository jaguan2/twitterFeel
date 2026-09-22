# TwitterFeel — Predicting Twitter Emotions from Tweet History

**Current development: [V4 workflow](docs/V4.md).** V4 adds audited data
preparation, historical emotion features, shared user-disjoint development
folds, stronger baselines, per-user metrics and reproducible run artifacts.
Test scoring is opt-in. V1–V3 scripts and results below are retained for
historical reproduction; their scores are not directly comparable to V4.

Start with the existing local data (legacy embeddings require explicit opt-in):

```powershell
python src/preprocess_v4.py data/interim/dataset_godknowswhat.csv data/interim/tweet_embeddings.npy data/interim/windows_v4_mht_w10.npz --allow-unverified-embeddings
python src/train_v4.py data/interim/windows_v4_mht_w10.npz --out runs/v4_binary_baselines --task binary
python -m unittest discover -s tests -v
```

Use new output names for subsequent runs. See [docs/V4.md](docs/V4.md) for
probability generation, LSTM feature comparisons and final test evaluation.

Given a Twitter user's last 9 tweets, predict the emotion of their 10th tweet.

TwitterFeel originated from the idea of detecting symptoms of depression on
social media. We shifted to general emotion prediction because the team isn't
clinically qualified to make mental-health claims.

**Tech:** TensorFlow / Keras 3, PyTorch (for HuggingFace embedding generation),
scikit-learn, pandas, NumPy, HuggingFace Transformers.

## Emotion mapping
```
0 sadness   3 anger
1 joy       4 fear
2 love      5 surprise
```

Binary valence collapse used in some experiments:
- positive = {joy, love, surprise}
- negative = {sadness, anger, fear}

## 📁 Project structure

```
twitterFeel/
├── data/
│   ├── raw/                    # source corpus (Mental-Health-Twitter.xls, cleaned_*)
│   └── interim/                # generated: labeled CSV, embeddings, windowed npz
├── src/
│   ├── paths.py                # shared filesystem layout — all scripts import from here
│   ├── prepare_sentiment140.py # adapt the Sentiment140 corpus into the pipeline schema
│   ├── prepare_reddit.py       # adapt a Pushshift Reddit dump into the pipeline schema
│   ├── generate_dataset.py     # DistilBERT 6-class emotion label per tweet
│   ├── generate_embeddings.py  # MiniLM 384-d sentence embedding per tweet
│   ├── preprocess.py           # V1 windows (notebook replica)
│   ├── preprocess_v2.py        # V2: embeddings + stride-1 sliding windows
│   ├── preprocess_v3.py        # V3: + cyclic time features, configurable window size
│   ├── train.py                # V1 LSTM, 6-class
│   ├── train_v2.py             # V2 LSTM, 6-class
│   ├── train_binary.py         # V2 binary valence (honest user-level split)
│   ├── train_leaky_binary.py   # V2 binary valence (leaky row-level split)
│   └── train_v3.py             # V3: 6/4-class or binary, LSTM or BiLSTM+attention
├── models/                     # trained checkpoints (emotion_*.keras)
├── metrics/                    # metrics_*.json, one per checkpoint (matching names)
├── docs/
│   ├── EXPERIMENTS.md          # per-iteration research log (E1–E14), hypothesis-first
│   ├── REDDIT.md               # Reddit corpus: acquisition, commands, caveats
│   ├── FINDINGS.md             # full results + investigation of the original F1=0.82 claim
│   ├── handoff.md              # handoff notes / remaining work
│   └── CAP4773_fin_1.ipynb     # original course notebook the project started from
├── twitter_feel_plus/          # sustained-sadness signal subproject (see its README)
└── requirements.txt            # pinned deps (TF 2.16+ for Python 3.12)
```

## Quick start

```bash
pip install -r requirements.txt

# One-time data generation (~20 min on CPU)
python src/generate_dataset.py        # DistilBERT 6-class emotion labels
python src/generate_embeddings.py     # MiniLM 384-d sentence embeddings

# Build V3 windowed features (15-history)
python src/preprocess_v3.py 16

# Train binary BiLSTM+attention
python src/train_v3.py windows_v3_w16.npz --binary --bilstm
```

All scripts read/write through `src/paths.py`; no path arguments needed
beyond the npz basename for `train_v3.py`.

## Working with other corpora

The pipeline is platform-agnostic. Everything downstream of ingest depends on
just three columns — `user_id`, `post_created`, `cleaned_text` — so a new
source needs only a `prepare_*.py` that emits them; a dataset *tag* then
carries it through the rest unchanged:

```bash
python src/prepare_reddit.py RC_2019-01.zst          # -> dataset_reddit_unlabeled.csv
python src/generate_dataset.py    data/interim/dataset_reddit_unlabeled.csv \
                                  data/interim/dataset_reddit.csv
python src/generate_embeddings.py data/interim/dataset_reddit_unlabeled.csv \
                                  data/interim/tweet_embeddings_reddit.npy
python src/preprocess_v3.py 10 reddit                # -> windows_reddit_w10.npz
python src/train_v3.py windows_reddit_w10.npz --binary
```

Corpora currently supported: **Twitter** (Mental-Health-Twitter, the default),
**Sentiment140** via `prepare_sentiment140.py`, and **Reddit** via
`prepare_reddit.py` (streams Pushshift `.zst`/`.bz2`/`.gz` dumps). See
[docs/REDDIT.md](docs/REDDIT.md) for acquisition and the caveats that affect
how Reddit results should be read.

## Historical V1–V3 results (user-level split)

Two corpora: **MHT** = the original Mental-Health-Twitter set (54 users),
**s140** = a Sentiment140 subsample (1,405 users). Baselines shift with the
corpus, so compare within a row, not across.

| Task | Corpus | Best model | Test acc | Weighted F1 | Beats baselines? |
|---|---|---|---:|---:|---|
| 6-class emotion | MHT | V3 LSTM w10, `--class-weight sqrt` | **0.461** | **0.410** | ✓ +5.6 pp over majority (0.405) — but see caveat |
| 4-class emotion (drop love + surprise) | MHT | V3 LSTM w10, balanced | 0.457 | 0.410 | ✓ +4.1 pp over majority (0.416) |
| Binary valence | s140 | V3 LSTM w10 | 0.572 | 0.563 | ✓ +3.5 pp over majority (0.538), ROC-AUC 0.593 |
| Binary valence | MHT | V3 LSTM w10 | 0.578 | 0.553 | ≈ tied with majority (0.568) |
| Binary valence (leaky row-level split, for comparison) | MHT | V2 LSTM | 0.595 | 0.595 | ✓ +2.7 pp over majority |

**The strongest honest result is s140 binary** — the only configuration that
beats both majority and persistence with a supporting ROC-AUC, and the
evidence that the original 38-user constraint really was binding.

**Caveat on the 6-class row.** Softening class weights (E14) lifts accuracy
from 0.389 to 0.461, but the per-class report shows the gain comes from the
model committing harder to the two dominant classes (anger, joy — 85% of test
rows), not from learning the rare ones: sadness and fear f1 actually *drop*.
Against the best constant predictor in hindsight (always-anger, 0.446) the
margin is only +1.6 pp. Treat it as a metrics result, not a modeling
breakthrough.

The 4-class collapse drops the two rarest emotions (`love`, `surprise`),
which together account for ~2.7% of windows and were essentially unlearnable
in the 6-class setup (f1 ≈ 0). With that noise removed, the model beats both
majority and persistence baselines, and it does so with the best macro F1 of
any multi-class run (0.247) — making it the most honest multi-class result
even though 6-class now edges it on accuracy.

See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for the per-experiment log
(hypothesis recorded before results), and
[docs/FINDINGS.md](docs/FINDINGS.md) + [docs/handoff.md](docs/handoff.md) for
the investigation of the original `F1 = 0.82` claim (not reproducible under
any defensible evaluation we tried) and pointers to the remaining work.
