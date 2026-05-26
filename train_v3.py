"""V3 training: configurable task (6-class or binary) and architecture
(LSTM or BiLSTM+attention) over the V3 feature windows.

Usage:
    python train_v3.py windows_v3_w10.npz                      # 6-class, LSTM
    python train_v3.py windows_v3_w10.npz --binary             # binary, LSTM
    python train_v3.py windows_v3_w10.npz --binary --bilstm    # binary, BiLSTM+attention
    python train_v3.py windows_v3_w16.npz --bilstm             # 6-class, BiLSTM+attention, larger window

Saves model + metrics with a descriptive suffix encoding (npz stem, task, arch).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers

REPO_ROOT = Path(__file__).resolve().parent

NUM_EMOTIONS = 6
EMOTION_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]
POSITIVE_CLASSES = {1, 2, 5}

RANDOM_SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15


def to_binary(y: np.ndarray) -> np.ndarray:
    return np.isin(y, list(POSITIVE_CLASSES)).astype(np.int64)


def user_level_split(user_ids, seed):
    rng = np.random.default_rng(seed)
    unique_users = np.unique(user_ids)
    rng.shuffle(unique_users)
    n = len(unique_users)
    n_train = int(round(n * TRAIN_FRAC))
    n_val = int(round(n * VAL_FRAC))
    train_u = set(unique_users[:n_train])
    val_u = set(unique_users[n_train : n_train + n_val])
    test_u = set(unique_users[n_train + n_val :])
    train_idx = np.array([i for i, u in enumerate(user_ids) if u in train_u])
    val_idx = np.array([i for i, u in enumerate(user_ids) if u in val_u])
    test_idx = np.array([i for i, u in enumerate(user_ids) if u in test_u])
    return train_idx, val_idx, test_idx


def build_lstm(input_shape, num_classes, binary):
    out_units = 1 if binary else num_classes
    activation = "sigmoid" if binary else "softmax"
    loss = "binary_crossentropy" if binary else "sparse_categorical_crossentropy"
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.LSTM(128, dropout=0.3),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(out_units, activation=activation),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3), loss=loss, metrics=["accuracy"]
    )
    return model


def build_bilstm_attention(input_shape, num_classes, binary):
    out_units = 1 if binary else num_classes
    activation = "sigmoid" if binary else "softmax"
    loss = "binary_crossentropy" if binary else "sparse_categorical_crossentropy"

    inputs = layers.Input(shape=input_shape)
    seq = layers.Bidirectional(layers.LSTM(96, return_sequences=True, dropout=0.3))(
        inputs
    )
    # Single-head additive attention over time
    attn_scores = layers.Dense(1, activation="tanh")(seq)            # (B, T, 1)
    attn_weights = layers.Softmax(axis=1)(attn_scores)               # (B, T, 1)
    context = layers.Multiply()([seq, attn_weights])                 # (B, T, 192)
    pooled = layers.Lambda(
        lambda x: tf.reduce_sum(x, axis=1),
        output_shape=lambda s: (s[0], s[2]),
    )(context)
    h = layers.Dense(64, activation="relu")(pooled)
    h = layers.Dropout(0.4)(h)
    outputs = layers.Dense(out_units, activation=activation)(h)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3), loss=loss, metrics=["accuracy"]
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", type=str, help="Path to windows_v3_w*.npz")
    parser.add_argument("--binary", action="store_true", help="Binary valence task")
    parser.add_argument(
        "--bilstm",
        action="store_true",
        help="Use BiLSTM + attention instead of plain LSTM",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()

    npz_path = REPO_ROOT / args.npz
    stem = npz_path.stem  # e.g. windows_v3_w10
    arch = "bilstm" if args.bilstm else "lstm"
    task = "binary" if args.binary else "6cls"
    tag = f"{stem}_{task}_{arch}"
    model_out = REPO_ROOT / f"emotion_{tag}.keras"
    metrics_out = REPO_ROOT / f"metrics_{tag}.json"

    tf.keras.utils.set_random_seed(RANDOM_SEED)

    data = np.load(npz_path)
    X = data["X"].astype(np.float32)
    y_multi = data["y"].astype(np.int64)
    user_ids = data["user_ids"]
    last_emo_multi = data["last_emotion"]
    print(f"Loaded {npz_path.name}  X={X.shape}  task={task}  arch={arch}")

    y = to_binary(y_multi) if args.binary else y_multi
    last_emo = to_binary(last_emo_multi) if args.binary else last_emo_multi

    train_idx, val_idx, test_idx = user_level_split(user_ids, RANDOM_SEED)
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_va, y_va = X[val_idx], y[val_idx]
    X_te, y_te = X[test_idx], y[test_idx]
    last_emo_te = last_emo[test_idx]
    print(f"Users {len(np.unique(user_ids[train_idx]))}/"
          f"{len(np.unique(user_ids[val_idx]))}/"
          f"{len(np.unique(user_ids[test_idx]))}, "
          f"rows {len(y_tr)}/{len(y_va)}/{len(y_te)}")

    if args.binary:
        classes = np.array([0, 1])
    else:
        classes = np.arange(NUM_EMOTIONS)
    present = np.array([c for c in classes if (y_tr == c).any()])
    raw = compute_class_weight("balanced", classes=present, y=y_tr)
    class_weight = {int(c): float(w) for c, w in zip(present, raw)}
    for c in classes:
        class_weight.setdefault(int(c), 1.0)
    print("Class weights:", class_weight)

    builder = build_bilstm_attention if args.bilstm else build_lstm
    model = builder(X.shape[1:], NUM_EMOTIONS, args.binary)
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3
        ),
    ]
    model.fit(
        X_tr,
        y_tr,
        validation_data=(X_va, y_va),
        epochs=args.epochs,
        batch_size=args.batch,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )

    if args.binary:
        prob = model.predict(X_te, verbose=0).ravel()
        pred = (prob >= 0.5).astype(np.int64)
        roc_auc = float(roc_auc_score(y_te, prob))
        target_names = ["negative", "positive"]
        labels = [0, 1]
        binary_f1 = float(f1_score(y_te, pred, average="binary", zero_division=0))
    else:
        prob = model.predict(X_te, verbose=0)
        pred = prob.argmax(axis=-1)
        roc_auc = None
        target_names = EMOTION_NAMES
        labels = list(range(NUM_EMOTIONS))
        binary_f1 = None

    report = classification_report(
        y_te, pred, labels=labels, target_names=target_names, digits=4, zero_division=0
    )
    cm = confusion_matrix(y_te, pred, labels=labels)
    acc = float(np.mean(pred == y_te))
    weighted_f1 = float(f1_score(y_te, pred, average="weighted", zero_division=0))
    macro_f1 = float(f1_score(y_te, pred, average="macro", zero_division=0))

    majority = int(np.bincount(y_tr, minlength=max(labels) + 1).argmax())
    majority_acc = float(np.mean(y_te == majority))
    persistence_acc = float(np.mean(last_emo_te == y_te))

    print(f"\n=== {tag}: test report ===")
    print(report)
    print("Confusion matrix:")
    print(cm)
    print(
        f"\nAccuracy={acc:.4f}  weighted_F1={weighted_f1:.4f}  macro_F1={macro_f1:.4f}"
        + (f"  ROC-AUC={roc_auc:.4f}" if roc_auc is not None else "")
    )
    print(f"Baselines:  majority_acc={majority_acc:.4f}  "
          f"persistence_acc={persistence_acc:.4f}")

    model.save(model_out)
    metrics_out.write_text(
        json.dumps(
            {
                "tag": tag,
                "npz": str(npz_path.name),
                "task": task,
                "arch": arch,
                "accuracy": acc,
                "weighted_f1": weighted_f1,
                "macro_f1": macro_f1,
                "binary_f1_positive": binary_f1,
                "roc_auc": roc_auc,
                "baselines": {
                    "majority_class": majority,
                    "majority_acc": majority_acc,
                    "persistence_acc": persistence_acc,
                },
                "report": report,
                "confusion_matrix": cm.tolist(),
            },
            indent=2,
        )
    )
    print(f"\nSaved -> {model_out.name}, {metrics_out.name}")


if __name__ == "__main__":
    main()
