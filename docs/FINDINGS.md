# TwitterFeel — Findings

End-to-end reproduction notes for the next-tweet emotion-prediction pipeline,
including an investigation of the README's claimed `F1 = 0.82`.

## Project context

TwitterFeel began as a depression-symptom detection project on Twitter and pivoted
to general 6-class emotion detection (sadness/joy/love/anger/fear/surprise)
because the team wasn't clinically qualified to make mental-health claims. This
write-up evaluates the model on its pivoted goal: given a user's most recent
9 tweets, predict the emotion of their 10th tweet.

## Pipeline

```
cleaned_Mental-Health-Twitter.xls   (20,000 tweets, CSV)
        |
        v
generate_dataset.py        DistilBERT emotion classifier
                           (bhadresh-savani/distilbert-base-uncased-emotion)
        |
        v
dataset_godknowswhat.csv   (20,000 rows + emotion labels)
        |
        +-------------------------------+
        |                               |
        v                               v
   v1 pipeline                     v2 pipeline (text features)
   preprocess.py                   generate_embeddings.py  (MiniLM-L6-v2, 384d)
        |                               |
        v                               v
   filtered_input_matrix.csv       tweet_embeddings.npy
   user_ids.csv                         |
        |                               v
        |                          preprocess_v2.py  (sliding window + emb features)
        |                               |
        |                               v
        |                          windows_v2.npz
        |                               |
        v                               +---------------------+
     train.py                           |                     |
        |                               v                     v
        v                          train_v2.py          train_binary.py
   emotion_lstm.keras              emotion_lstm_v2     emotion_binary.keras
   metrics.json                    .keras              metrics_binary.json
                                   metrics_v2.json     train_leaky_binary.py
                                                       emotion_leaky_binary.keras
                                                       metrics_leaky_binary.json
```

## Headline results

All splits use the same random seed (42). All "honest" rows use a
**user-level** split (no user appears in more than one of train/val/test).
The leaky row uses a row-level random split for comparison.

| Setup                                                  | Test Acc | Weighted F1 | Beats baselines? |
|--------------------------------------------------------|---------:|------------:|------------------|
| Random guess (uniform over classes)                    |    0.167 |           — | —                |
| Majority class (6-class, predict `joy`)                |    0.405 |           — | —                |
| Persistence (predict last tweet's emotion)             |    0.401 |           — | —                |
| **V1 LSTM** (emotion-IDs + 2 time flags, user-level)   |    0.232 |       0.279 | ❌ loses to both  |
| **V2 LSTM** (text embeddings + sliding window, user-level) |  0.364 |       0.342 | ❌ loses to both  |
| **Binary V2** (positive vs negative, user-level)       |    0.570 |       0.559 | ❌ ≈ majority (0.568) |
| **Binary V2 LEAKY** (row-level split)                  |    0.595 |       0.595 | ✅ beats both     |

The leaky binary model is the only setup that beats both baselines, and the
absolute improvement (≈ 2–7 pp) is real but modest.

## The `F1 = 0.82` investigation

The README claims `F1 = 0.82`. We could not reproduce it. We tried:

- V1 features with user-level split → 0.28 F1
- V2 features (text embeddings + sliding window, fixed `split_group`) with user-level split → 0.34 F1
- Binary collapse (positive/negative) with user-level split → 0.56 F1
- Binary collapse with row-level (leaky) split → 0.60 F1

The closest we got was **0.595** — 22 pp below the README's claim. The most
likely explanations are:

1. **Training-set F1 reported as test F1.** Leaky models hit 0.65–0.76 training
   accuracy in fewer than 10 epochs; with more epochs they would easily reach
   0.82+ on training data while staying ~0.40 on test.
2. **Score from the upstream DistilBERT** (≈0.93 F1 on `dair-ai/emotion`) was
   conflated with the downstream LSTM.
3. **Different dataset or preprocessing** than what's in the current repo.

We can't fully rule any of these out, but the data and pipeline in this repo
do not support 0.82 under any defensible evaluation.

## Is it a model issue or a training issue?

**Neither — it's a data and task issue.** Evidence:

- **Capacity is fine.** The leaky V2 model reached 76% training accuracy in 9
  epochs. The model can fit the training set; it just can't generalize.
- **Training duration is fine.** Early stopping fires at epoch 5–10 in every
  experiment because validation loss plateaus or rises. Training longer makes
  the train/test gap worse, not better.
- **The train/test gap is the symptom.** Across honest splits, training
  accuracy climbs while held-out accuracy stalls — classic data scarcity
  signature, not architecture or learning-rate signature.

The binding constraints, in rough order of impact:

1. **38 training users.** Far too few to learn user-invariant emotional
   dynamics. The model overfits to those 38 individuals' styles.
2. **Tweet-to-tweet emotion is mostly autocorrelated.** Persistence captures
   most of the easily-predictable signal; there isn't much room above it.
3. **Label noise.** Targets are themselves predictions from another model
   (`DistilBERT` on cleaned text). Training to mimic another model's argmax
   compounds error.
4. **No time anchoring.** A 9-tweet window can span an hour or a year; the
   model has no `time_since_previous_tweet` feature.
5. **Severe class imbalance.** Surprise (60 windows) and love (193) cannot be
   learned well at this dataset size regardless of class weighting.

A bigger LSTM, a Transformer, more epochs, more dropout — none of these
address any of the five constraints above. Real improvement requires
*different data* or *a different task framing*.

## What this project *did* show

- **Text embeddings beat emotion-ID-only by a wide margin** (+13 pp accuracy
  from V1 → V2). The original notebook's choice to throw away tweet text was
  the biggest single missed opportunity.
- **Sliding-window stride-1 windowing gave 9.8× more training data** than the
  notebook's non-overlapping batches and fixed the silent slicing bug in
  `split_group` that dropped trailing partial batches.
- **User-level evaluation is essential.** Row-level splits inflate scores via
  per-user style memorization without telling you anything useful about
  generalization.
- **Binary valence is a far more reachable task** than 6-class emotion for
  this dataset.

## Where this could go from here

Roughly in expected-impact order:

1. **More data.** More users, ideally a public next-emotion or emotion-stream
   dataset. The 38-user constraint dominates everything else.
2. **End-to-end fine-tuning** of a small transformer instead of frozen MiniLM
   embeddings; task-specific representations should add a few points.
3. **Time-gap features** (`minutes_since_previous_tweet`, log-scaled) and
   cyclical hour/day-of-week encoding.
4. **Soft labels** — train on DistilBERT probability distributions, not
   argmax. Smoother gradient under label noise.
5. **Per-user embeddings.** Concatenate a learned per-user vector with the
   LSTM head so the model can specialize cleanly.
6. **Reframe the task.** "Next tweet's emotion" is mostly persistence; "did
   this user's emotion shift in the next N tweets" or "dominant emotion in
   the next 24h" is more useful and probably more learnable.

## Reproducing

All commands run from the repo root. Scripts live in `src/` and import
filesystem paths from `src/paths.py`; data, models, and metrics are written
to `data/interim/`, `models/`, and `metrics/`.

```bash
pip install -r requirements.txt

# V1 (notebook-equivalent pipeline)
python src/generate_dataset.py        # ~15 min on CPU
python src/preprocess.py
python src/train.py                   # ~30 s

# V2 (text embeddings + sliding window)
python src/generate_embeddings.py     # ~5-10 min on CPU
python src/preprocess_v2.py
python src/train_v2.py                # ~2 min

# Binary valence (positive vs negative)
python src/train_binary.py            # user-level (honest) split
python src/train_leaky_binary.py      # row-level (leaky) split for comparison
```

Final metrics live in
[../metrics/metrics.json](../metrics/metrics.json),
[../metrics/metrics_v2.json](../metrics/metrics_v2.json),
[../metrics/metrics_binary.json](../metrics/metrics_binary.json), and
[../metrics/metrics_leaky_binary.json](../metrics/metrics_leaky_binary.json).
