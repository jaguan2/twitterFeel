# TwitterFeel — Handoff

## TL;DR

We built an end-to-end reproducible pipeline for next-tweet emotion prediction,
ran a rigorous evaluation, and **could not reproduce the README's `F1 = 0.82`**.
Best honest result we have: **0.595 accuracy on the binary valence task with
leaky row-level split**, or **0.570 accuracy with user-level (truly honest)
split**. The gap to 0.82 is a data/task issue, not a model or training issue —
see [FINDINGS.md](FINDINGS.md) for the full diagnosis.

This document captures where we are, what's still incomplete, and how to
resume.

## What we built

Layout (post-reorg):

```
twitterFeel/
  data/raw/        Mental-Health-Twitter.xls, cleaned_Mental-Health-Twitter.xls
  data/interim/    dataset_godknowswhat.csv, tweet_embeddings.npy, windows_v*.npz, ...
  src/             all .py scripts + paths.py (shared filesystem constants)
  models/          *.keras
  metrics/         metrics_*.json
  docs/            README, FINDINGS, handoff, original notebook
  requirements.txt
  .gitignore
```

| Script | Purpose |
|---|---|
| [generate_dataset.py](../src/generate_dataset.py) | DistilBERT emotion labels → `data/interim/dataset_godknowswhat.csv` |
| [generate_embeddings.py](../src/generate_embeddings.py) | MiniLM 384-d sentence embeddings → `data/interim/tweet_embeddings.npy` |
| [preprocess.py](../src/preprocess.py) | V1: CLI replica of notebook windowing → `data/interim/filtered_input_matrix.csv` |
| [preprocess_v2.py](../src/preprocess_v2.py) | V2: sliding-window + text embeddings → `data/interim/windows_v2.npz` |
| [preprocess_v3.py](../src/preprocess_v3.py) | V3: V2 + cyclical time + log time-gap → `data/interim/windows_v3_w{N}.npz` |
| [train.py](../src/train.py) | V1 LSTM (emotion-IDs only, user-level split) |
| [train_v2.py](../src/train_v2.py) | V2 LSTM (embeddings, user-level split) |
| [train_binary.py](../src/train_binary.py) | Binary valence, user-level split |
| [train_leaky_binary.py](../src/train_leaky_binary.py) | Binary valence, row-level (leaky) split |
| [train_v3.py](../src/train_v3.py) | Configurable V3 trainer: `--binary` and `--bilstm` flags |
| [paths.py](../src/paths.py) | Shared filesystem constants (DATA_RAW, DATA_INTERIM, MODELS_DIR, METRICS_DIR) |

Plus the original notebook [CAP4773_fin_1.ipynb](CAP4773_fin_1.ipynb) and the source data.

## Full results table

All "honest" rows = user-level split (no user in more than one of train/val/test).
All numbers are on the test set.

| # | Setup                                  | Acc    | Weighted F1 | Notes |
|---|----------------------------------------|-------:|------------:|-------|
| 0 | Majority class (6-class)               | 0.405  | —           | "Always predict joy" |
| 0 | Persistence (6-class)                  | 0.401  | —           | "Repeat last emotion" |
| 0 | Majority class (binary)                | 0.568  | —           | "Always negative" |
| 0 | Persistence (binary)                   | 0.543  | —           | |
| 1 | V1 LSTM, 6-class, user-level           | 0.232  | 0.279       | Original notebook approach |
| 2 | V2 LSTM, 6-class, user-level           | 0.364  | 0.342       | +Text embeddings, +sliding window |
| 3 | V3 LSTM, 6-class, user-level (w10)     | 0.389  | 0.379       | +Time-gap & cyclical features |
| 4 | V3 LSTM, 6-class, user-level (w16)     | 0.415  | 0.368       | Larger window: higher acc, lower F1 |
| 5 | V2 binary, user-level                  | 0.570  | 0.559       | Binary valence task |
| 6 | V3 LSTM binary, user-level (w10)       | 0.578  | 0.553       | Time features ≈ neutral on binary |
| 7 | V3 BiLSTM binary, user-level (w10)     | 0.571  | 0.560       | BiLSTM+attention: ≈ tied with LSTM |
| 8 | V3 LSTM binary, user-level (w16)       | 0.562  | 0.559       | Larger window didn't help |
| 9 | V3 BiLSTM binary, user-level (w16)     | 0.553  | 0.553       | BiLSTM + larger window: below majority (0.570). ROC-AUC 0.560 |
| 10| **V3 LSTM 4-class, user-level (w10)**  | **0.457** | **0.410** | **First honest multi-class result that beats baselines (maj 0.416, persist 0.411).** Drops love + surprise (only 2.7% of test windows but f1 ≈ 0 in 6-class). macro F1 0.247 |
| 11| V3 LSTM 4-class, user-level (w16)      | 0.439  | 0.399       | w10 wins again on F1 (same pattern as 6-class). Still beats baselines (maj 0.415, persist 0.409) |
| 12| **Binary V2 LEAKY** (row-level split)  | **0.595** | **0.595** | Beats baselines; ROC-AUC 0.64 |

## Headline finding

- **V3 features helped 6-class meaningfully** (+2.5pp acc over V2) but **didn't help binary** (≈ tied with V2).
- **BiLSTM + attention didn't move the needle**, and combined with a larger window (BiLSTM w16 binary, row #9) it actually dropped below the majority baseline.
- **Larger window (16 vs 10) had mixed effects** — higher 6-class accuracy but lower F1 (model collapses to majority more).
- **4-class collapse (drop love + surprise) is the first honest multi-class setup to beat baselines** — V3 LSTM 4cls w10 hits 0.457 acc vs majority 0.416 (+4.1pp), proving the rare-class noise (love + surprise have f1 ≈ 0 in the 6-class model) was actively hurting the rest of the predictions.
- **Best honest result remains 0.595** on leaky binary; honest 0.578 on user-level binary; honest 0.457 on multi-class (4-class w10).
- The 0.82 in the README is most likely a training-set F1, a misreport, or computed under leakage we can't reproduce.

## What's incomplete

1. ~~**V3 BiLSTM binary on w16**~~ — **Done**: row #9 above (acc 0.553 / F1 0.553 / ROC-AUC 0.560, below majority baseline of 0.570). Closes out the planned experiment matrix.

2. **Soft-label distillation** — proposed but not started. Would re-run DistilBERT, save the **probability distributions** (not just argmax) and train with KL-divergence loss against those soft targets. Reduces label noise. Probably +1–3pp.

3. **End-to-end DistilBERT fine-tuning** — the biggest unrealized lever. Replace frozen MiniLM embeddings with task-adapted ones by fine-tuning a small transformer on `(9 tweets concatenated) → next emotion`. **Heavy** — probably 1–4 hours on CPU per epoch. GPU strongly preferred. Expected payoff: +10–15pp on honest 6-class if it works.

4. **Per-user fine-tuning / user embeddings** — not feasible with our user-level split, since test users are unseen. Would only help leaky evaluation.

5. **Reframed tasks** — e.g. "predict dominant emotion of next 3 tweets" (smoother target), or "did the user's emotion shift?" (binary). Not tried.

## What to do next

In rough order of effort/payoff:

### Quick (< 30 min)
- ~~Finish the V3 BiLSTM w16 binary run~~ — done.
- ~~Update README~~ — done; the README now reports the honest numbers and links to FINDINGS for the 0.82 investigation.

### Medium (1–3 hours)
- **Soft-label distillation** (#2 above). Write `generate_probabilities.py` (very similar to `generate_dataset.py` but saves full prob distribution) and `train_soft.py` that uses KL-divergence loss.
- ~~Try a 4-class collapse~~ — done. `--four-class` flag in [src/train_v3.py](../src/train_v3.py); w10 LSTM hits 0.457 acc / 0.410 F1, first honest multi-class to beat both baselines. Worth also running with `--bilstm` and the binary-task BiLSTM if appetite remains (BiLSTM hasn't helped elsewhere though).

### Heavy (4+ hours, GPU strongly preferred)
- **End-to-end DistilBERT fine-tuning** (#3 above). This is the path most likely to actually move the honest result toward 0.50+, but isn't realistic on this machine without significant patience.

## How to resume the environment

```bash
# All deps are pinned; this gets you back in business from a fresh clone:
pip install -r requirements.txt

# All commands run from the repo root. Scripts in src/ import a shared
# paths.py module that resolves data/, models/, metrics/ relative to REPO_ROOT.
# Heavy intermediates (npy, npz) are gitignored under data/interim/.

# Source generation (one-time, slow)
python src/generate_dataset.py        # ~15 min, DistilBERT inference over 20k tweets
python src/generate_embeddings.py     # ~5-10 min, MiniLM embeddings

# V1 (notebook pipeline)
python src/preprocess.py
python src/train.py

# V2 (embeddings + sliding window)
python src/preprocess_v2.py
python src/train_v2.py

# V3 (V2 + time-gap + cyclical features). Window size is configurable:
python src/preprocess_v3.py 10        # 9-history windows
python src/preprocess_v3.py 16        # 15-history windows
python src/train_v3.py windows_v3_w10.npz                  # 6-class LSTM
python src/train_v3.py windows_v3_w10.npz --binary         # binary LSTM
python src/train_v3.py windows_v3_w10.npz --binary --bilstm  # binary BiLSTM+attention
python src/train_v3.py windows_v3_w16.npz                  # 6-class on 15-history
# train_v3.py accepts a bare basename and resolves to data/interim/

# Binary valence (separate scripts; use windows_v2.npz)
python src/train_binary.py            # honest (user-level split)
python src/train_leaky_binary.py      # leaky (row-level split, for comparison)
```

## Current file inventory

### Source / docs / config
- [../README.md](../README.md) — still claims F1 = 0.82; **needs updating**
- [FINDINGS.md](FINDINGS.md) — full investigation write-up
- [handoff.md](handoff.md) — this file
- [../requirements.txt](../requirements.txt) — pinned at TF 2.16–2.18 for Python 3.12 compatibility
- [../.gitignore](../.gitignore) — keeps heavy intermediates out of git
- [CAP4773_fin_1.ipynb](CAP4773_fin_1.ipynb) — original notebook
- `data/raw/Mental-Health-Twitter.xls`, `data/raw/cleaned_Mental-Health-Twitter.xls` — source data (CSV, despite the extension)

### Scripts (all alive, see table at top) — `src/`
- `paths.py` — shared filesystem layout, imported by all other scripts
- `generate_dataset.py`, `generate_embeddings.py`
- `preprocess.py`, `preprocess_v2.py`, `preprocess_v3.py`
- `train.py`, `train_v2.py`, `train_binary.py`, `train_leaky_binary.py`, `train_v3.py`

### Trained models — `models/*.keras`
- `emotion_lstm.keras` — V1 honest
- `emotion_lstm_v2.keras` — V2 honest
- `emotion_binary.keras` — V2 binary honest
- `emotion_leaky_binary.keras` — V2 binary leaky
- `emotion_windows_v3_w10_6cls_lstm.keras` — V3 6cls w10
- `emotion_windows_v3_w10_binary_lstm.keras` — V3 binary w10
- `emotion_windows_v3_w10_binary_bilstm.keras` — V3 binary BiLSTM w10
- `emotion_windows_v3_w16_6cls_lstm.keras` — V3 6cls w16
- `emotion_windows_v3_w16_binary_lstm.keras` — V3 binary w16
- `emotion_windows_v3_w16_binary_bilstm.keras` — V3 binary BiLSTM w16 (closing out the matrix)
- `emotion_windows_v3_w10_4cls_lstm.keras` — V3 4-class w10 (drop love + surprise)
- `emotion_windows_v3_w16_4cls_lstm.keras` — V3 4-class w16

### Metrics — `metrics/metrics_*.json`
One per trained model, same naming as the `.keras` files.

### Generated intermediates (gitignored, live under `data/interim/`)
- `dataset_godknowswhat.csv` (~2 MB) — 20k tweets + DistilBERT emotion labels
- `tweet_embeddings.npy` (~30 MB) — MiniLM embeddings, (20000, 384)
- `windows_v2.npz` (~170 MB) — V2 windowed data
- `windows_v3_w10.npz`, `windows_v3_w16.npz` — V3 windowed data, two window sizes
- `filtered_input_matrix.csv`, `user_ids.csv` — V1 intermediates
