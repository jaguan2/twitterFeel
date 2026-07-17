"""Binary valence prediction: positive vs negative next-tweet emotion.

Same V2 windows (text embeddings + sliding window) as train_v2.py, same
user-level (honest) split, but the 6-class target is collapsed:

    POSITIVE = joy(1), love(2), surprise(5)   -> 1
    NEGATIVE = sadness(0), anger(3), fear(4)  -> 0

Outputs:
    emotion_binary.keras
    metrics_binary.json
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
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers

from paths import DATA_INTERIM, METRICS_DIR, MODELS_DIR

INPUT_NPZ = DATA_INTERIM / "windows_v2.npz"
MODEL_OUT = MODELS_DIR / "emotion_binary.keras"
METRICS_OUT = METRICS_DIR / "metrics_binary.json"

# 6-class -> binary valence mapping
POSITIVE_CLASSES = {1, 2, 5}  # joy, love, surprise
NEGATIVE_CLASSES = {0, 3, 4}  # sadness, anger, fear

RANDOM_SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15


def to_binary(y: np.ndarray) -> np.ndarray:
    """0 = negative, 1 = positive."""
    return np.isin(y, list(POSITIVE_CLASSES)).astype(np.int64)


def user_level_split(user_ids, frac_train, frac_val, seed):
    rng = np.random.default_rng(seed)
    unique_users = np.unique(user_ids)
    rng.shuffle(unique_users)
    n = len(unique_users)
    n_train = int(round(n * frac_train))
    n_val = int(round(n * frac_val))
    train_u = set(unique_users[:n_train])
    val_u = set(unique_users[n_train : n_train + n_val])
    test_u = set(unique_users[n_train + n_val :])
    train_idx = np.array([i for i, u in enumerate(user_ids) if u in train_u])
    val_idx = np.array([i for i, u in enumerate(user_ids) if u in val_u])
    test_idx = np.array([i for i, u in enumerate(user_ids) if u in test_u])
    print(f"Users  train/val/test: {len(train_u)}/{len(val_u)}/{len(test_u)}")
    print(f"Rows   train/val/test: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")
    return train_idx, val_idx, test_idx


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
    user_ids = data["user_ids"]
    last_emotion_multi = data["last_emotion"]

    y = to_binary(y_multi)
    last_emo_binary = to_binary(last_emotion_multi)
    pos_pct = float(y.mean()) * 100
    print(f"Loaded X={X.shape}, y={y.shape} (positive={pos_pct:.1f}%)")

    train_idx, val_idx, test_idx = user_level_split(
        user_ids, TRAIN_FRAC, VAL_FRAC, RANDOM_SEED
    )
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_va, y_va = X[val_idx], y[val_idx]
    X_te, y_te = X[test_idx], y[test_idx]
    last_emo_te = last_emo_binary[test_idx]

    raw_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_tr)
    class_weight = {0: float(raw_weights[0]), 1: float(raw_weights[1])}
    print(f"Train positive%: {y_tr.mean()*100:.1f}  Class weights: {class_weight}")

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
        class_weight=class_weight,
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

    print("\n=== Binary test report ===")
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
                "positive_classes": ["joy", "love", "surprise"],
                "negative_classes": ["sadness", "anger", "fear"],
                "evaluation_protocol": "user_level_split",
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
