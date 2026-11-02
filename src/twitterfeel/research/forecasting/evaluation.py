"""User-disjoint development folds and user-clustered accuracy uncertainty."""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import KFold


def normalize_probabilities(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if (probabilities.ndim != 2 or not np.isfinite(probabilities).all()
            or (probabilities < 0).any() or (probabilities.sum(axis=1) <= 0).any()):
        raise ValueError("Expected finite nonnegative class probabilities with positive row sums")
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def make_splits(user_ids, seed=42, folds=5, test_fraction=0.15):
    users = np.unique(user_ids)
    if not 0 < test_fraction < 1 or folds < 2:
        raise ValueError("Require 0 < test_fraction < 1 and folds >= 2")
    np.random.default_rng(seed).shuffle(users)
    n_test = max(1, round(len(users) * test_fraction))
    test_users, dev_users = users[:n_test], users[n_test:]
    if len(dev_users) < folds:
        raise ValueError("Too few development users for the requested folds")
    development = np.flatnonzero(np.isin(user_ids, dev_users))
    test = np.flatnonzero(np.isin(user_ids, test_users))
    splits = []
    for tr, va in KFold(folds, shuffle=True, random_state=seed).split(dev_users):
        splits.append((np.flatnonzero(np.isin(user_ids, dev_users[tr])),
                       np.flatnonzero(np.isin(user_ids, dev_users[va]))))
    return development, test, splits


def scores(y, probabilities, user_ids):
    probabilities = normalize_probabilities(probabilities)
    labels = np.arange(probabilities.shape[1])
    pred = probabilities.argmax(axis=1)
    per_user = []
    for user in np.unique(user_ids):
        mask = user_ids == user
        per_user.append({"user_id": str(user), "rows": int(mask.sum()),
                         "accuracy": float(accuracy_score(y[mask], pred[mask])),
                         "macro_f1": float(f1_score(y[mask], pred[mask], labels=labels,
                                                   average="macro", zero_division=0))})
    result = {"rows": len(y), "users": len(per_user),
              "accuracy": float(accuracy_score(y, pred)),
              "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
              "weighted_f1": float(f1_score(y, pred, labels=labels, average="weighted", zero_division=0)),
              "log_loss": float(log_loss(y, probabilities, labels=labels)),
              "mean_user_accuracy": float(np.mean([p["accuracy"] for p in per_user])),
              "per_user": per_user}
    if len(labels) == 2:
        result["roc_auc"] = float(roc_auc_score(y, probabilities[:, 1])) if len(np.unique(y)) == 2 else None
    return result


def clustered_accuracy_interval(y, probabilities, user_ids, reference=None, seed=42, draws=2000):
    """Resample whole users, preserving within-user window dependence.

    Returns a window-weighted accuracy (or paired accuracy difference) CI.
    OOF intervals condition on the fitted models; they do not include refit uncertainty.
    """
    correct = (probabilities.argmax(axis=1) == y).astype(float)
    if reference is not None:
        correct -= (reference.argmax(axis=1) == y)
    users, inverse = np.unique(user_ids, return_inverse=True)
    counts = np.bincount(inverse)
    totals = np.bincount(inverse, weights=correct)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        sampled = rng.integers(0, len(users), len(users))
        values.append(totals[sampled].sum() / counts[sampled].sum())
    return {"estimate": float(correct.mean()), "lower": float(np.quantile(values, .025)),
            "upper": float(np.quantile(values, .975)), "confidence": .95,
            "method": "paired user-cluster bootstrap" if reference is not None else "user-cluster bootstrap",
            "draws": draws, "users": len(users)}
