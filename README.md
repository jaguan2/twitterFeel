# TwitterFeel — Predicting Twitter Emotions from Tweet History

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

## Repository layout

```
data/raw/        source data (Mental-Health-Twitter.xls, cleaned_*.xls)
data/interim/    generated artifacts (embeddings, windowed npz, ...)  -- gitignored
src/             all Python scripts + paths.py (shared filesystem layout)
models/          *.keras (trained model checkpoints)
metrics/         metrics_*.json (one per trained model)
docs/            FINDINGS.md (investigation), handoff.md (handoff notes),
                 CAP4773_fin_1.ipynb (original course notebook)
requirements.txt pinned deps (TF 2.16+ for Python 3.12 wheels)
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

## Headline results (honest, user-level split)

| Task | Best model | Test acc | Weighted F1 | Beats baselines? |
|---|---|---:|---:|---|
| 6-class emotion | V3 LSTM w10 | 0.389 | 0.379 | ✗ trails majority (0.405) |
| 4-class emotion (drop love + surprise) | V3 LSTM w10 | **0.457** | **0.410** | ✓ +4.1 pp over majority (0.416) |
| Binary valence  | V3 LSTM w10 | 0.578 | 0.553 | ≈ tied with majority (0.568) |
| Binary valence (leaky row-level split, for comparison) | V2 LSTM | 0.595 | 0.595 | ✓ +2.7 pp over majority |

The 4-class collapse drops the two rarest emotions (`love`, `surprise`),
which together account for ~2.7% of windows and were essentially unlearnable
in the 6-class setup (f1 ≈ 0). With that noise removed, the model becomes
the first honest multi-class setup to beat both majority and persistence
baselines. See [docs/FINDINGS.md](docs/FINDINGS.md) for the full diagnosis
of why 6-class trails majority and what's left to try.

See [docs/FINDINGS.md](docs/FINDINGS.md) and [docs/handoff.md](docs/handoff.md)
for the full results table, investigation of the original `F1 = 0.82` claim
(not reproducible under any defensible evaluation we tried), and pointers to
the remaining work.
