"""Provisional threshold comparisons against human-reviewed development posts.

No model inference occurs. Test labels never participate in threshold selection.
Bootstrap intervals describe this reviewed sample, not population reliability.
"""
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np

from emotion_schema import file_digest
from label_posts import read_posts
from label_schema import LABELS, interpret
from workbench_data import load_predictions

DEFAULT_THRESHOLDS = (.2, .3, .4, .5, .6, .7, .8)


def _metrics(truth, predicted):
    tp = (truth & predicted).sum(axis=0)
    support = truth.sum(axis=0)
    guessed = predicted.sum(axis=0)
    precision = np.divide(tp, guessed, out=np.zeros(len(LABELS)), where=guessed != 0)
    recall = np.divide(tp, support, out=np.zeros(len(LABELS)), where=support != 0)
    f1 = np.divide(2 * tp, support + guessed, out=np.zeros(len(LABELS)), where=(support + guessed) != 0)
    union = (truth | predicted).sum(axis=1)
    denominator = int(support.sum() + guessed.sum())
    mixed = truth[:, :LABELS.index("neutral")].sum(axis=1) > 1
    mixed_support = int(truth[mixed].sum())
    return {"micro_f1": float(2 * tp.sum() / denominator) if denominator else 0.,
            "micro_precision": float(tp.sum() / guessed.sum()) if guessed.sum() else 0.,
            "micro_recall": float(tp.sum() / support.sum()) if support.sum() else 0.,
            "exact_match": float((truth == predicted).all(axis=1).mean()),
            "mean_label_jaccard": float(np.divide((truth & predicted).sum(axis=1), union,
                                                  out=np.zeros(len(truth)), where=union != 0).mean()),
            "mixed_emotion": {"posts": int(mixed.sum()), "label_support": mixed_support,
                "full_label_recovery": float(((truth[mixed] & predicted[mixed]) == truth[mixed]).all(axis=1).mean()) if mixed.any() else None,
                "label_recall": float((truth[mixed] & predicted[mixed]).sum() / mixed_support) if mixed_support else None},
            "per_label": {label: {"support": int(support[i]), "predicted": int(guessed[i]),
                                   "precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i])}
                          for i, label in enumerate(LABELS)}}


def _bootstrap(truth, predicted, groups):
    unique = list(dict.fromkeys(groups))
    members = [np.flatnonzero(np.asarray(groups) == group) for group in unique]
    rng = np.random.default_rng(42)
    values = []
    for _ in range(200):
        indices = np.concatenate([members[i] for i in rng.integers(0, len(members), size=len(members))])
        y, p = truth[indices], predicted[indices]
        denominator = int(y.sum() + p.sum())
        values.append(float(2 * (y & p).sum() / denominator) if denominator else 0.)
    lower, upper = np.percentile(values, [2.5, 97.5])
    return {"lower": float(lower), "upper": float(upper), "level": .95, "repetitions": 200,
            "seed": 42, "method": "percentile_cluster_bootstrap", "clusters": len(unique)}


def benchmark(review_path, predictions_path, thresholds=None, *, split="development",
              published_reference=False):
    if split != "development":
        raise ValueError("Threshold selection is restricted to development; test labels must remain held out")
    thresholds = list(DEFAULT_THRESHOLDS if thresholds is None else thresholds)
    if (not thresholds or any(isinstance(t, bool) or not isinstance(t, (int, float))
                              or not math.isfinite(t) or not 0 < t < 1 for t in thresholds)
            or len(thresholds) != len(set(thresholds))):
        raise ValueError("Choose unique, finite thresholds between zero and one")
    thresholds = sorted(float(t) for t in thresholds)
    review_path, predictions_path = Path(review_path), Path(predictions_path)
    review = read_posts(review_path)
    if not {"split", "human_labels", "review_status"}.issubset(review.columns):
        raise ValueError("Reference needs split, human_labels, and review_status")
    if not review.split.isin(["development", "test"]).all():
        raise ValueError("Reference split must be development or test")
    if not review.review_status.isin(["reviewed", "pending", "unclear", "invalid"]).all():
        raise ValueError("Reference has unknown review_status")
    if published_reference and ("reference_origin" not in review or
                               not review.reference_origin.eq("published_human_annotations").all()):
        raise ValueError("Published-reference policy requires explicit published human annotation provenance")
    overlaps = int((review.groupby("text_sha256").split.nunique() > 1).sum())
    if overlaps and not published_reference:
        raise ValueError("Identical text crosses development/test partitions")
    selected = review[review.split == "development"]
    reviewed = selected[selected.review_status == "reviewed"]
    duplicates = int(reviewed.text_sha256.duplicated().sum())
    if duplicates and not published_reference:
        raise ValueError("Duplicate reviewed text would inflate the benchmark; retain one annotation per text")
    counts = {"reference_rows": len(review), "development_rows": len(selected), "reviewed_rows": len(reviewed),
              "held_out_rows": int(review.split.eq("test").sum()),
              "review_status_counts": {str(k): int(v) for k, v in selected.review_status.value_counts().items()},
              "excluded_review_status_counts": {str(k): int(v) for k, v in selected[selected.review_status != "reviewed"].review_status.value_counts().items()}}
    hashes = {"reference_sha256": file_digest(review_path), "predictions_sha256": file_digest(predictions_path)}
    config_path = predictions_path.parent / "config.json"
    source = {}
    if config_path.is_file():
        hashes["prediction_config_sha256"] = file_digest(config_path)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        source = {"dataset": config.get("dataset"), "model": config.get("model"), "provenance": config.get("provenance")}
        expected = config.get("predictions_sha256")
        if expected and expected != hashes["predictions_sha256"]:
            raise ValueError("Predictions differ from their recorded artifact hash")
    result = {"ready": False, "split": "development", "counts": counts, "hashes": hashes,
              "source": source, "results": [], "recommended_threshold": None,
              "warnings": [], "selection_rule": "Highest development micro F1; ties prefer closest to 0.5, then higher threshold",
              "interpretation": "Provisional development comparison against human judgments; not held-out accuracy. Intervals exclude threshold-selection uncertainty."}
    if published_reference:
        result["reference_policy"] = "Preserve official development annotation frequencies and neutral overlaps; test rows excluded from every metric"
        result["counts"].update(repeated_development_texts=duplicates, development_test_text_overlaps=overlaps)
        result["warnings"].append("Official annotation frequencies are preserved. Repeated text and unavailable author IDs can make post bootstrap intervals overconfident.")
    if reviewed.empty:
        result["warnings"].append("No reviewed development posts. Complete blind human review before measuring labeling quality.")
        return result
    predictions = {row["post_id"]: row for row in load_predictions(predictions_path).to_dict("records")}
    truth_rows, paired = [], []
    for row in reviewed.to_dict("records"):
        labels = [label.strip() for label in row["human_labels"].split(";") if label.strip()]
        if (not labels or len(labels) != len(set(labels)) or not set(labels).issubset(LABELS)
                or (not published_reference and "neutral" in labels and len(labels) > 1)):
            raise ValueError(f"Invalid human labels for {row['post_id']}")
        prediction = predictions.get(row["post_id"])
        if prediction is None or prediction["text_sha256"] != row["text_sha256"]:
            raise ValueError(f"Missing or text-mismatched prediction: {row['post_id']}")
        truth_rows.append([label in labels for label in LABELS])
        paired.append(prediction)
    truth = np.asarray(truth_rows, dtype=bool)
    users_available = "user_id" in reviewed and reviewed.user_id.str.strip().ne("").all()
    groups = reviewed.user_id.tolist() if users_available else reviewed.post_id.tolist()
    result["bootstrap_unit"] = "user" if users_available else "post"
    if len(reviewed) < 100:
        result["warnings"].append(f"Only {len(reviewed)} reviewed development posts; estimates and threshold selection may be unstable.")
    if len(set(groups)) < 20:
        result["warnings"].append(f"Only {len(set(groups))} independent bootstrap units; intervals can be unstable or degenerate.")
    sparse = [label for label, support in zip(LABELS, truth.sum(axis=0)) if support < 10]
    if sparse:
        result["warnings"].append("Labels with fewer than 10 positive human examples: " + ", ".join(sparse))
    if not users_available:
        result["warnings"].append("User IDs are absent or incomplete; post bootstrap cannot account for dependence between posts by the same person.")
    result["counts"]["invalid_predictions"] = sum(row["status"] == "invalid" for row in paired)
    for threshold in thresholds:
        guessed, statuses = [], Counter()
        for row in paired:
            if row["status"] == "invalid":
                labels, status = [], "invalid"
            else:
                interpretation = interpret(row["scores"], threshold)
                labels, status = interpretation["labels"], interpretation["status"]
                if row.get("truncated", False):
                    status = "review_needed"
            guessed.append([label in labels for label in LABELS])
            statuses[status] += 1
        predicted = np.asarray(guessed, dtype=bool)
        metrics = _metrics(truth, predicted)
        result["results"].append({"threshold": threshold, **metrics, "status_counts": dict(statuses),
                                  "micro_f1_interval": _bootstrap(truth, predicted, groups)})
    best = max(result["results"], key=lambda row: (row["micro_f1"], -abs(row["threshold"] - .5), row["threshold"]))
    result.update(ready=True, recommended_threshold=best["threshold"])
    result["counts"]["per_label_support"] = {label: int(value) for label, value in zip(LABELS, truth.sum(axis=0))}
    return result
