"""Binary valence prediction with a leaky row-level split.

Same V2 text-embedding windows as train_binary.py, same model, same loss --
just swaps user-level split for row-level random split. This is the ceiling
of "what's reachable if we let the model memorize user style."

Outputs:
    emotion_leaky_binary.keras
    metrics_leaky_binary.json
"""
from __future__ import annotations

import json

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

from paths import DATA_INTERIM, METRICS_DIR, MODELS_DIR

INPUT_NPZ = DATA_INTERIM / "windows_v2.npz"
MODEL_OUT = MODELS_DIR / "emotion_leaky_binary.keras"
METRICS_OUT = METRICS_DIR / "metrics_leaky_binary.json"

POSITIVE_CLASSES = {1, 2, 5}  # joy, love, surprise
RANDOM_SEED = 42
TEST_FRAC = 0.20
VAL_FRAC = 0.10


def to_binary(y: np.ndarray) -> np.ndarray:
    return np.isin(y, list(POSITIVE_CLASSES)).astype(np.int64)


def build_model(input_shape):
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.LSTM(128, dropout=0.3),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    data = np.load(INPUT_NPZ)
    X = data["X"].astype(np.float32)
    y_multi = data["y"].astype(np.int64)
    last_emotion_multi = data["last_emotion"]

    y = to_binary(y_multi)
    last_emo_binary = to_binary(last_emotion_multi)
    print(f"Loaded X={X.shape}, y={y.shape} (positive={y.mean()*100:.1f}%)")

    indices = np.arange(len(y))
    idx_tr, idx_te = train_test_split(
        indices, test_size=TEST_FRAC, random_state=RANDOM_SEED, stratify=y
    )
    idx_tr, idx_va = train_test_split(
        idx_tr, test_size=VAL_FRAC, random_state=RANDOM_SEED, stratify=y[idx_tr]
    )
    X_tr, y_tr = X[idx_tr], y[idx_tr]
    X_va, y_va = X[idx_va], y[idx_va]
    X_te, y_te = X[idx_te], y[idx_te]
    last_emo_te = last_emo_binary[idx_te]
    print(f"Rows train/val/test: {len(y_tr)}/{len(y_va)}/{len(y_te)}")

    model = build_model(X.shape[1:])
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
        epochs=60,
        batch_size=64,
        callbacks=callbacks,
        verbose=2,
    )

    test_prob = model.predict(X_te, verbose=0).ravel()
    test_pred = (test_prob >= 0.5).astype(np.int64)

    report = classification_report(
        y_te,
        test_pred,
        labels=[0, 1],
        target_names=["negative", "positive"],
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_te, test_pred, labels=[0, 1])
    acc = float(np.mean(test_pred == y_te))
    weighted_f1 = float(f1_score(y_te, test_pred, average="weighted", zero_division=0))
    binary_f1 = float(f1_score(y_te, test_pred, average="binary", zero_division=0))
    roc_auc = float(roc_auc_score(y_te, test_prob))

    majority = int(np.bincount(y_tr).argmax())
    majority_acc = float(np.mean(y_te == majority))
    persistence_acc = float(np.mean(last_emo_te == y_te))

    print("\n=== Leaky binary test report ===")
    print(report)
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)
    print(
        f"\nAccuracy = {acc:.4f}  weighted F1 = {weighted_f1:.4f}  "
        f"binary F1(pos) = {binary_f1:.4f}  ROC-AUC = {roc_auc:.4f}"
    )
    print(
        f"Baselines:  majority_acc={majority_acc:.4f}  "
        f"persistence_acc={persistence_acc:.4f}"
    )

    model.save(MODEL_OUT)
    METRICS_OUT.write_text(
        json.dumps(
            {
                "task": "binary_valence",
                "evaluation_protocol": "leaky_row_level_split",
                "warning": (
                    "Row-level split: same user's windows can appear in both "
                    "train and test. Numbers inflate via per-user style memorization."
                ),
                "accuracy": acc,
                "weighted_f1": weighted_f1,
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
    print(f"\nSaved -> {MODEL_OUT.name}, {METRICS_OUT.name}")


if __name__ == "__main__":
    main()
