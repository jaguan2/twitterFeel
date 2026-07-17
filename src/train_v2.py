"""V2 LSTM training on text-embedding windows.

Reads windows_v2.npz (produced by preprocess_v2.py), trains a bigger LSTM,
evaluates against persistence + majority baselines.

Writes:
    emotion_lstm_v2.keras
    metrics_v2.json
"""
from __future__ import annotations

import json

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers

from paths import DATA_INTERIM, METRICS_DIR, MODELS_DIR

INPUT_NPZ = DATA_INTERIM / "windows_v2.npz"
MODEL_OUT = MODELS_DIR / "emotion_lstm_v2.keras"
METRICS_OUT = METRICS_DIR / "metrics_v2.json"

NUM_EMOTIONS = 6
EMOTION_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]

RANDOM_SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15


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


def build_model(input_shape, num_classes):
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.LSTM(128, dropout=0.3, recurrent_dropout=0.0),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    data = np.load(INPUT_NPZ)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    user_ids = data["user_ids"]
    last_emotion = data["last_emotion"]
    print(f"Loaded X={X.shape}, y={y.shape}, {len(np.unique(user_ids))} users")

    train_idx, val_idx, test_idx = user_level_split(
        user_ids, TRAIN_FRAC, VAL_FRAC, RANDOM_SEED
    )
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_va, y_va = X[val_idx], y[val_idx]
    X_te, y_te = X[test_idx], y[test_idx]
    last_emo_te = last_emotion[test_idx]

    classes_present = np.unique(y_tr)
    raw_weights = compute_class_weight("balanced", classes=classes_present, y=y_tr)
    class_weight = {int(c): float(w) for c, w in zip(classes_present, raw_weights)}
    # Fill in any missing classes with a neutral weight so Keras is happy.
    for c in range(NUM_EMOTIONS):
        class_weight.setdefault(c, 1.0)
    print("Class weights:", class_weight)

    model = build_model(X.shape[1:], NUM_EMOTIONS)
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

    test_pred = model.predict(X_te, verbose=0).argmax(axis=-1)

    report = classification_report(
        y_te,
        test_pred,
        labels=list(range(NUM_EMOTIONS)),
        target_names=EMOTION_NAMES,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_te, test_pred, labels=list(range(NUM_EMOTIONS)))

    majority = int(np.bincount(y_tr, minlength=NUM_EMOTIONS).argmax())
    majority_acc = float(np.mean(y_te == majority))
    persistence_acc = float(np.mean(last_emo_te == y_te))

    print("\n=== Test report ===")
    print(report)
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)
    print(
        f"\nBaselines:  majority_acc={majority_acc:.4f}"
        f"  persistence_acc={persistence_acc:.4f}"
    )

    model.save(MODEL_OUT)
    METRICS_OUT.write_text(
        json.dumps(
            {
                "report": report,
                "confusion_matrix": cm.tolist(),
                "baselines": {
                    "majority_class": majority,
                    "majority_acc": majority_acc,
                    "persistence_acc": persistence_acc,
                },
            },
            indent=2,
        )
    )
    print(f"\nSaved model -> {MODEL_OUT.name}")
    print(f"Saved metrics -> {METRICS_OUT.name}")


if __name__ == "__main__":
    main()
