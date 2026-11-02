"""Compare baselines and LSTM on shared user-disjoint development folds.

Default runs never score held-out test users. Use --evaluate-test only after
choosing a configuration on development results. Each output directory is new.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from twitterfeel.research.forecasting.emotion_schema import EMOTION_NAMES, file_digest
from twitterfeel.research.forecasting.evaluation import make_splits, scores, clustered_accuracy_interval, normalize_probabilities

MODELS = ("majority", "persistence", "frequency", "transition", "logistic", "lstm")


def prepare_task(data, task):
    y = data["y"].astype(int)
    history = data["emotion_history"].astype(np.float32)
    keep = np.ones(len(y), dtype=bool)
    if task == "binary":
        y = np.isin(y, [1, 2, 5]).astype(int)
        history = np.stack((history[:, :, [0, 3, 4]].sum(axis=-1),
                            history[:, :, [1, 2, 5]].sum(axis=-1)), axis=-1)
        names = ["negative", "positive"]
    elif task == "4cls":
        classes = [0, 1, 3, 4]
        keep = np.isin(y, classes)
        remap = np.full(6, -1)
        remap[classes] = np.arange(4)
        y = remap[y]
        history = history[:, :, classes]
        sums = history.sum(axis=-1, keepdims=True)
        # Out-of-task historical labels carry no preference among retained classes.
        history = np.divide(history, sums, out=np.full_like(history, .25), where=sums > 0)
        names = [EMOTION_NAMES[c] for c in classes]
    else:
        names = EMOTION_NAMES
    return y, history, keep, names


def fit_predict(name, X, history, y, users, train, predict, args, n_classes):
    counts = np.bincount(y[train], minlength=n_classes)
    prior = (counts + 1) / (len(train) + n_classes)
    if name == "majority":
        return np.tile(prior, (len(predict), 1)), None
    if name == "persistence":
        return history[predict, -1], None
    if name == "frequency":
        return history[predict].mean(axis=1), None
    if name == "transition":
        # Fit only training history-last -> target pairs; Laplace smoothing.
        transitions = np.ones((n_classes, n_classes))
        for cls in range(n_classes):
            transitions[:, cls] += history[train[y[train] == cls], -1].sum(axis=0)
        transitions /= transitions.sum(axis=1, keepdims=True)
        return history[predict, -1] @ transitions, {"transition_matrix": transitions.tolist()}
    if len(np.unique(y[train])) < 2:
        raise ValueError(f"{name}: training fold has fewer than two target classes")
    inputs = X if args.features == "text-time" else np.concatenate((X, history), axis=-1)
    if name == "logistic":
        flat = np.concatenate((inputs[:, -1], inputs.mean(axis=1)), axis=1)
        weights = None
        if args.class_weight != "none":
            present = np.flatnonzero(counts)
            raw = len(train) / (len(present) * counts[present])
            if args.class_weight == "sqrt":
                raw = np.sqrt(raw)
            weights = dict(zip(present.tolist(), raw.tolist()))
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=args.c, class_weight=weights, max_iter=2000, random_state=args.seed))
        model.fit(flat[train], y[train])
        prob = np.zeros((len(predict), n_classes))
        prob[:, model[-1].classes_] = model.predict_proba(flat[predict])
        return prob, model
    import tensorflow as tf
    from tensorflow import keras
    keras.backend.clear_session()
    keras.utils.set_random_seed(args.seed)
    # Fixed epoch budget is selected by development CV, never by the test set.
    normalization = keras.layers.Normalization(axis=-1)
    normalization.adapt(inputs[train])
    model = keras.Sequential([
        keras.layers.Input(shape=inputs.shape[1:]), normalization,
        keras.layers.LSTM(64, dropout=.3), keras.layers.Dense(32, activation="relu"),
        keras.layers.Dropout(.3), keras.layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy")
    weights = None
    if args.class_weight != "none":
        present = np.flatnonzero(counts)
        raw = len(train) / (len(present) * counts[present])
        if args.class_weight == "sqrt":
            raw = np.sqrt(raw)
        weights = {int(c): float(w) for c, w in zip(present, raw)}
        weights.update({int(c): 1. for c in np.flatnonzero(counts == 0)})
    model.fit(inputs[train], y[train], epochs=args.epochs, batch_size=args.batch,
              class_weight=weights, verbose=2)
    return model.predict(inputs[predict], verbose=0), model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--task", choices=["binary", "4cls", "6cls"], default="binary")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS[:-1]))
    parser.add_argument("--features", choices=["text-time", "text-time-emotion"], default="text-time-emotion")
    parser.add_argument("--class-weight", choices=["none", "balanced", "sqrt"], default="none")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42, help="Training/bootstrap seed; split seed is independent")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--c", type=float, default=1.)
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1 or args.c <= 0:
        parser.error("epochs, batch and c must be positive")
    if args.out.exists():
        parser.error("Output directory already exists; choose a new run name")
    data = np.load(args.npz, allow_pickle=False)
    X, users = data["X"], data["user_ids"].astype(str)
    if (X.ndim != 3 or X.shape[2] != 391 or not np.isfinite(X).all()
            or data["emotion_history"].shape != (len(X), X.shape[1], 6)
            or len(users) != len(X) or data["y"].shape != (len(X),)
            or not np.isin(data["y"], range(6)).all()):
        raise ValueError("Invalid V4 feature or label shapes/values")
    h = data["emotion_history"]
    if not np.isfinite(h).all() or (h < 0).any() or not np.allclose(h.sum(axis=-1), 1, atol=1e-5):
        raise ValueError("Invalid history probability distributions")
    y, history, keep, names = prepare_task(data, args.task)
    # Split before target filtering, so four-class filtering cannot reshuffle users.
    dev, test, folds = make_splits(users, args.split_seed, args.folds)
    dev, test = dev[keep[dev]], test[keep[test]]
    folds = [(tr[keep[tr]], va[keep[va]]) for tr, va in folds]
    if not len(test) or any(not len(tr) or not len(va) for tr, va in folds):
        raise ValueError("Empty split after task filtering")
    args.out.mkdir(parents=True)
    assignment = np.full(len(y), "excluded", dtype="U12")
    assignment[test] = "test"
    fold_id = np.full(len(y), -1)
    for fold, (_, va) in enumerate(folds):
        assignment[va] = "development"
        fold_id[va] = fold
    np.savez_compressed(args.out / "splits.npz", user_ids=users, assignment=assignment,
                        fold=fold_id, source_rows=data["source_rows"], target_time=data["target_time"])
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(npz_sha256=file_digest(args.npz), labels=names,
                  python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__)
    config["source_sha256"] = {name: file_digest(Path(__file__).with_name(name))
                               for name in ("train_v4.py", "evaluation.py", "emotion_schema.py")}
    from twitterfeel.common import provenance
    config["source_sha256"]["common/provenance.py"] = file_digest(provenance.__file__)
    if "lstm" in args.models:
        from importlib.metadata import version
        config["tensorflow"] = version("tensorflow")
    audit_path = args.npz.with_suffix(".json")
    if audit_path.exists():
        config["data_audit"] = json.loads(audit_path.read_text(encoding="utf-8"))
        if config["data_audit"].get("npz_sha256") != config["npz_sha256"]:
            raise ValueError("NPZ differs from preprocessing audit")
    (args.out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    predictions, results = {}, {}
    for name in dict.fromkeys(args.models):
        oof = np.full((len(y), len(names)), np.nan)
        fold_scores = []
        for fold, (tr, va) in enumerate(folds):
            print(f"{name}: fold {fold+1}/{len(folds)}", flush=True)
            prob, _ = fit_predict(name, X, history, y, users, tr, va, args, len(names))
            prob = normalize_probabilities(prob)
            oof[va] = prob
            fold_scores.append(scores(y[va], prob, users[va]))
        predictions[name] = oof[dev]
        result = {"development": scores(y[dev], oof[dev], users[dev]), "folds": fold_scores}
        result["development"]["accuracy_ci"] = clustered_accuracy_interval(
            y[dev], oof[dev], users[dev], seed=args.seed)
        np.savez_compressed(args.out / f"{name}_development.npz", probability=oof[dev],
                            y=y[dev], user_ids=users[dev], row_index=dev)
        if args.evaluate_test:
            prob, model = fit_predict(name, X, history, y, users, dev, test, args, len(names))
            prob = normalize_probabilities(prob)
            result["test"] = scores(y[test], prob, users[test])
            result["test"]["accuracy_ci"] = clustered_accuracy_interval(y[test], prob, users[test], seed=args.seed)
            np.savez_compressed(args.out / f"{name}_test.npz", probability=prob, y=y[test],
                                user_ids=users[test], row_index=test)
            if name == "lstm":
                model.save(args.out / "model.keras")
            elif name == "logistic":
                import joblib
                joblib.dump(model, args.out / "model.joblib")
            elif isinstance(model, dict):
                (args.out / "transition.json").write_text(json.dumps(model), encoding="utf-8")
        results[name] = result
        print(name, {k: v for k, v in result["development"].items() if k not in ("per_user", "accuracy_ci")})
        (args.out / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    for name, probability in predictions.items():
        results[name]["paired_development_accuracy_delta"] = {
            baseline: clustered_accuracy_interval(y[dev], probability, users[dev], reference, args.seed)
            for baseline, reference in predictions.items()
            if baseline in ("majority", "persistence", "frequency", "transition") and baseline != name
        }
    (args.out / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved {args.out}; test scored: {args.evaluate_test}")


if __name__ == "__main__":
    main()
