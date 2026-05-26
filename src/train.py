"""Train an LSTM to predict the next tweet's emotion from a 9-tweet history window.

Reads (produced by CAP4773_fin_1.ipynb):
    filtered_input_matrix.csv  -- 28 columns: 27 features (9 tweets x 3) + 1 target
    user_ids.csv               -- parallel user_id per row, for user-level split

Writes:
    emotion_lstm.keras  -- trained model
    metrics.json        -- final test-set metrics + baseline comparison
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers

from paths import DATA_INTERIM, METRICS_DIR, MODELS_DIR

INPUT_MATRIX = DATA_INTERIM / "filtered_input_matrix.csv"
USER_IDS = DATA_INTERIM / "user_ids.csv"
MODEL_OUT = MODELS_DIR / "emotion_lstm.keras"
METRICS_OUT = METRICS_DIR / "metrics.json"

NUM_EMOTIONS = 6
TIMESTEPS = 9
FEATURES_PER_STEP = 3  # (emotion_encoded, weekend, overnight)
EMOTION_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]

RANDOM_SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15  # remaining 0.15 -> test


def load_data():
    matrix = pd.read_csv(INPUT_MATRIX).to_numpy(dtype=np.float32)
    user_ids = pd.read_csv(USER_IDS)["user_id"].to_numpy()

    X_flat = matrix[:, : TIMESTEPS * FEATURES_PER_STEP]
    y = matrix[:, -1].astype(np.int64)

    X = X_flat.reshape(-1, TIMESTEPS, FEATURES_PER_STEP)
    return X, y, user_ids


def one_hot_emotion(X: np.ndarray) -> np.ndarray:
    """(emotion_encoded, weekend, overnight) -> (one_hot[6], weekend, overnight)."""
    emotion = X[:, :, 0].astype(np.int64)
    one_hot = np.eye(NUM_EMOTIONS, dtype=np.float32)[emotion]   # (N, 9, 6)
    flags = X[:, :, 1:3].astype(np.float32)                     # (N, 9, 2)
    return np.concatenate([one_hot, flags], axis=-1)            # (N, 9, 8)


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
            layers.LSTM(64),
            layers.Dropout(0.3),
            layers.Dense(32, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def baselines(y_train, X_test, y_test):
    """Cheap reference points the LSTM has to beat."""
    majority = int(np.bincount(y_train, minlength=NUM_EMOTIONS).argmax())
    maj_acc = float(np.mean(y_test == majority))

    # Persistence: predict the 9th tweet's emotion (last timestep, argmax over one-hot dims 0:6)
    last_step_one_hot = X_test[:, -1, :NUM_EMOTIONS]
    persistence_pred = last_step_one_hot.argmax(axis=-1)
    persist_acc = float(np.mean(persistence_pred == y_test))

    return {
        "majority_class": majority,
        "majority_acc": maj_acc,
        "persistence_acc": persist_acc,
    }


def main():
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    X_raw, y, user_ids = load_data()
    X = one_hot_emotion(X_raw)
    print(f"Loaded X={X.shape}, y={y.shape}, {len(np.unique(user_ids))} users")

    train_idx, val_idx, test_idx = user_level_split(
        user_ids, TRAIN_FRAC, VAL_FRAC, RANDOM_SEED
    )

    X_tr, y_tr = X[train_idx], y[train_idx]
    X_va, y_va = X[val_idx], y[val_idx]
    X_te, y_te = X[test_idx], y[test_idx]

    class_weights = compute_class_weight(
        "balanced", classes=np.arange(NUM_EMOTIONS), y=y_tr
    )
    class_weight = {i: float(w) for i, w in enumerate(class_weights)}
    print("Class weights:", class_weight)

    model = build_model((TIMESTEPS, X.shape[-1]), NUM_EMOTIONS)
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2
        ),
    ]

    model.fit(
        X_tr,
        y_tr,
        validation_data=(X_va, y_va),
        epochs=50,
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
    bases = baselines(y_tr, X_te, y_te)

    print("\n=== Test report ===")
    print(report)
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)
    print(
        f"\nBaselines:  majority_acc={bases['majority_acc']:.4f}"
        f"  persistence_acc={bases['persistence_acc']:.4f}"
    )

    model.save(MODEL_OUT)
    METRICS_OUT.write_text(
        json.dumps(
            {
                "report": report,
                "confusion_matrix": cm.tolist(),
                "baselines": bases,
            },
            indent=2,
        )
    )
    print(f"\nSaved model -> {MODEL_OUT.name}")
    print(f"Saved metrics -> {METRICS_OUT.name}")


if __name__ == "__main__":
    main()
