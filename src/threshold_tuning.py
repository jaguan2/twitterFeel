"""Development-only grouped cross-validation of threshold calibration methods.

No model fitting/inference occurs. Official test annotations never enter tuning.
This corpus already influenced the pretrained model and historical 0.3 baseline;
development CV is not independent validation on Twitter or uploaded datasets.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from emotion_schema import file_digest
from label_posts import read_posts
from label_schema import LABELS, interpret
from workbench_data import load_predictions

GRID = np.round(np.arange(.05, .951, .01), 2)
METHODS = ("fixed_0.5", "fixed_0.3", "fine_shared", "regularized_per_label", "regularized_macro_per_label")


def grouped_folds(hashes, folds=5, seed=42):
    groups = np.unique(hashes)
    if len(groups) < folds:
        raise ValueError("Need at least five unique development texts for grouped cross-validation")
    np.random.default_rng(seed).shuffle(groups)
    assignment = {value: i for i, chunk in enumerate(np.array_split(groups, folds)) for value in chunk}
    return np.asarray([assignment[value] for value in hashes])


def _f1(tp, guessed, support):
    denominator = guessed + support
    return np.divide(2 * tp, denominator, out=np.zeros_like(np.asarray(tp), dtype=float), where=denominator != 0)


def _tie(values, preferred):
    return min(values, key=lambda value: (round(abs(float(value) - preferred), 10), -float(value)))


def fit_shared(scores, truth):
    support = int(truth.sum())
    values = []
    for threshold in GRID:
        guessed = scores >= threshold
        values.append(float(_f1(int((guessed & truth).sum()), int(guessed.sum()), support)))
    best = max(values)
    return float(_tie(GRID[np.isclose(values, best, rtol=0, atol=1e-12)], .5))


def fit_per_label(scores, truth, shared=None):
    shared = fit_shared(scores, truth) if shared is None else shared
    thresholds = np.full(len(LABELS), shared)
    support = truth.sum(axis=0)
    eligible = (support >= 25) & ((len(truth) - support) >= 25)
    predictions = scores >= thresholds
    tp = (truth & predictions).sum(axis=0).astype(float)
    guessed = predictions.sum(axis=0).astype(float)
    total_support = int(support.sum())
    candidates_tp, candidates_guessed = [], []
    for label in range(len(LABELS)):
        candidates = scores[:, label, None] >= GRID
        candidates_tp.append((candidates & truth[:, label, None]).sum(axis=0))
        candidates_guessed.append(candidates.sum(axis=0))
    passes = 0
    for _ in range(5):
        improved = False
        passes += 1
        for label in range(len(LABELS)):
            if not eligible[label]:
                continue
            candidates = _f1(tp.sum() - tp[label] + candidates_tp[label],
                             guessed.sum() - guessed[label] + candidates_guessed[label], total_support)
            current = float(_f1(tp.sum(), guessed.sum(), total_support))
            best = float(candidates.max())
            if best <= current + 1e-12:
                continue
            threshold = float(_tie(GRID[np.isclose(candidates, best, rtol=0, atol=1e-12)], shared))
            index = int(np.flatnonzero(GRID == threshold)[0])
            thresholds[label] = threshold
            tp[label], guessed[label] = candidates_tp[label][index], candidates_guessed[label][index]
            improved = True
        if not improved:
            break
    return thresholds, {"passes": passes, "positive_support": {label: int(v) for label, v in zip(LABELS, support)},
                        "fallback_labels": [label for label, valid in zip(LABELS, eligible) if not valid]}


def fit_macro_per_label(scores, truth, shared):
    thresholds = np.full(len(LABELS), shared)
    for label in range(len(LABELS)):
        support = int(truth[:, label].sum())
        if support < 25 or len(truth) - support < 25:
            continue
        guessed = scores[:, label, None] >= GRID
        values = _f1((guessed & truth[:, label, None]).sum(axis=0), guessed.sum(axis=0), support)
        thresholds[label] = float(_tie(GRID[np.isclose(values, values.max(), rtol=0, atol=1e-12)], shared))
    return thresholds


def metrics(truth, predicted):
    tp = (truth & predicted).sum(axis=0)
    support, guessed = truth.sum(axis=0), predicted.sum(axis=0)
    fp, fn = guessed - tp, support - tp
    precision = np.divide(tp, guessed, out=np.zeros(len(LABELS)), where=guessed != 0)
    recall = np.divide(tp, support, out=np.zeros(len(LABELS)), where=support != 0)
    f1 = _f1(tp, guessed, support)
    mixed = truth[:, :LABELS.index("neutral")].sum(axis=1) > 1
    emotional = slice(0, LABELS.index("neutral"))
    union = (truth | predicted).sum(axis=1)
    return {"micro_f1": float(_f1(tp.sum(), guessed.sum(), support.sum())),
            "micro_precision": float(tp.sum() / guessed.sum()) if guessed.sum() else 0.,
            "micro_recall": float(tp.sum() / support.sum()) if support.sum() else 0.,
            "exact_match": float((truth == predicted).all(axis=1).mean()),
            "mean_label_jaccard": float(np.divide((truth & predicted).sum(axis=1), union, out=np.zeros(len(truth)), where=union != 0).mean()),
            "macro_f1_all_labels": float(f1.mean()),
            "emotion_only_micro_f1": float(_f1(tp[emotional].sum(), guessed[emotional].sum(), support[emotional].sum())),
            "macro_f1_supported_labels": float(f1[support > 0].mean()),
            "mixed_emotion": {"posts": int(mixed.sum()),
                "full_label_recovery": float(((truth[mixed] & predicted[mixed]) == truth[mixed]).all(axis=1).mean()) if mixed.any() else None,
                "label_recall": float((truth[mixed] & predicted[mixed]).sum() / truth[mixed].sum()) if mixed.any() else None},
            "error_counts": {"false_positive_labels": int(fp.sum()), "false_negative_labels": int(fn.sum()),
                             "posts_with_any_error": int((truth != predicted).any(axis=1).sum()),
                             "neutral_emotion_conflicts": int((predicted[:, LABELS.index("neutral")] & (predicted[:, :LABELS.index("neutral")].any(axis=1))).sum())},
            "per_label": {label: {"support": int(support[i]), "precision": float(precision[i]), "recall": float(recall[i]),
                                    "f1": float(f1[i]), "false_positives": int(fp[i]), "false_negatives": int(fn[i])}
                          for i, label in enumerate(LABELS)}}


def paired_bootstrap(truth, predictions, groups):
    unique, inverse = np.unique(groups, return_inverse=True)
    statistics = {}
    for method, predicted in predictions.items():
        statistics[method] = np.stack([np.bincount(inverse, weights=values, minlength=len(unique)) for values in (
            (truth & predicted).sum(axis=1), predicted.sum(axis=1), truth.sum(axis=1))])
    rng = np.random.default_rng(42)
    differences = {method: [] for method in predictions}
    for _ in range(500):
        weights = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))
        scores = {method: float(_f1(*(stats @ weights))) for method, stats in statistics.items()}
        for method in differences:
            differences[method].append(scores[method] - scores["fixed_0.3"])
    return {method: {"lower": float(np.percentile(values, 2.5)), "upper": float(np.percentile(values, 97.5)),
                     "proportion_positive": float(np.mean(np.asarray(values) > 0)), "level": .95,
                     "repetitions": 500, "seed": 42, "unit": "exact_text_sha256", "clusters": len(unique),
                     "interpretation": "Paired percentile bootstrap difference from historical fixed 0.3; proportion positive is not a p-value"}
            for method, values in differences.items()}


def refine(reference, predictions_path, out, progress=None):
    reference, predictions_path, out = Path(reference), Path(predictions_path), Path(out)
    if out.exists():
        raise ValueError("Choose a fresh output directory; prior refinements are immutable")
    review = read_posts(reference)
    if not {"split", "human_labels", "review_status"}.issubset(review):
        raise ValueError("Reference needs split, human_labels, and review_status")
    # Test annotations are never parsed or used in threshold fitting/metrics.
    development = review[review.split.eq("development") & review.review_status.eq("reviewed")].copy()
    if development.empty:
        raise ValueError("No human-reviewed development annotations")
    truth_rows = []
    for row in development.to_dict("records"):
        labels = [x.strip() for x in row["human_labels"].split(";") if x.strip()]
        if not labels or not set(labels).issubset(LABELS) or len(labels) != len(set(labels)):
            raise ValueError("Invalid development reference labels")
        truth_rows.append([label in labels for label in LABELS])
    truth = np.asarray(truth_rows, dtype=bool)
    config_path = predictions_path.parent / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("predictions_sha256") != file_digest(predictions_path):
        raise ValueError("Predictions must match their recorded configuration hash")
    saved = {row["post_id"]: row for row in load_predictions(predictions_path).to_dict("records")}
    scores = []
    truncated = []
    for row in development.to_dict("records"):
        prediction = saved.get(row["post_id"])
        if prediction is None or prediction["text_sha256"] != row["text_sha256"]:
            raise ValueError("Development predictions must match every reference post and text hash")
        if prediction["status"] == "invalid":
            raise ValueError("Invalid development model scores must be fixed before calibration")
        scores.append([prediction["scores"][label] for label in LABELS])
        truncated.append(bool(prediction.get("truncated")))
    scores = np.asarray(scores)
    hashes = development.text_sha256.to_numpy()
    assignment = grouped_folds(hashes)
    predictions = {method: np.zeros_like(truth) for method in METHODS}
    folds = []
    for fold in range(5):
        training, validation = assignment != fold, assignment == fold
        shared = fit_shared(scores[training], truth[training])
        per_label, details = fit_per_label(scores[training], truth[training], shared)
        macro_thresholds = fit_macro_per_label(scores[training], truth[training], shared)
        thresholds = {"fixed_0.5": .5, "fixed_0.3": .3, "fine_shared": shared, "regularized_per_label": per_label,
                      "regularized_macro_per_label": macro_thresholds}
        for method, threshold in thresholds.items():
            predictions[method][validation] = scores[validation] >= threshold
        folds.append({"fold": fold, "training_rows": int(training.sum()), "validation_rows": int(validation.sum()),
                      "training_text_groups": int(len(set(hashes[training]))), "validation_text_groups": int(len(set(hashes[validation]))),
                      "shared_threshold": shared, "per_label_thresholds": dict(zip(LABELS, map(float, per_label))),
                      "macro_per_label_thresholds": dict(zip(LABELS, map(float, macro_thresholds))), **details})
        if progress:
            progress(fold + 1, 5)
    intervals = paired_bootstrap(truth, predictions, hashes)
    results = {method: {**metrics(truth, guessed), "difference_micro_f1_vs_fixed_0.3":
                       float(metrics(truth, guessed)["micro_f1"] - metrics(truth, predictions["fixed_0.3"])["micro_f1"]),
                       "paired_difference_interval": intervals[method]} for method, guessed in predictions.items()}
    for method, guessed in predictions.items():
        status_counts = {status: 0 for status in ("labeled", "neutral", "uncertain", "review_needed")}
        for index, decisions in enumerate(guessed):
            rendered = interpret(dict(zip(LABELS, map(float, decisions))))
            status_counts["review_needed" if truncated[index] else rendered["status"]] += 1
        results[method]["status_counts"] = status_counts
        results[method]["automatic_acceptance_fraction"] = (status_counts["labeled"] + status_counts["neutral"]) / len(truth)
    # Prefer a simpler method on exact F1 ties. Selection across OOF methods can
    # itself be optimistic; a later independent corpus must assess the choice.
    baseline = results["fixed_0.3"]
    balanced_candidates = [method for method in METHODS if results[method]["micro_f1"] > baseline["micro_f1"]
                           and results[method]["macro_f1_all_labels"] >= baseline["macro_f1_all_labels"]
                           and (results[method]["mixed_emotion"]["label_recall"] or 0) >= (baseline["mixed_emotion"]["label_recall"] or 0)]
    selected = max(balanced_candidates or METHODS, key=lambda method: (results[method]["micro_f1"], -METHODS.index(method)))
    all_shared = fit_shared(scores, truth)
    all_per_label, final_details = fit_per_label(scores, truth, all_shared)
    all_macro = fit_macro_per_label(scores, truth, all_shared)
    final_thresholds = {"fixed_0.5": np.full(len(LABELS), .5), "fixed_0.3": np.full(len(LABELS), .3),
                        "fine_shared": np.full(len(LABELS), all_shared), "regularized_per_label": all_per_label,
                        "regularized_macro_per_label": all_macro}[selected]
    input_hashes = {"reference_sha256": file_digest(reference), "predictions_sha256": file_digest(predictions_path),
                    "prediction_config_sha256": file_digest(config_path)}
    warnings = ["Development cross-validation on the model's own training domain is not independent Twitter/upload accuracy.",
                "The historical fixed 0.3 baseline was selected using all development data before this experiment; its OOF comparison is not an untouched baseline.",
                "Selecting the best method from these OOF results adds selection optimism. Bootstrap intervals hold fitted OOF predictions fixed and omit model/tuning uncertainty.",
                "Official test annotations are reserved and never used; published neutral overlaps and repeated development comments are preserved.",
                "Bootstrap groups identical texts; absent author IDs prevent accounting for different posts by the same person."]
    warnings.append("The macro-label objective was added after examining the initial micro-objective results; this extension is exploratory, not fresh confirmation.")
    if not balanced_candidates:
        warnings.append("No candidate improved micro F1 while preserving both macro F1 and mixed-emotion label recall versus historical 0.3. The exported profile is an objective-specific research tradeoff, not a generally better model.")
    calibration = {"schema": "emotion-calibration-v1", "status": "provisional_development_only", "method": selected,
                   "name": "GoEmotions " + ("joint micro F1" if selected == "regularized_per_label" else "macro-label F1" if selected == "regularized_macro_per_label" else selected) + " research profile", "provisional": True,
                   "thresholds": dict(zip(LABELS, map(float, final_thresholds))), "hashes": input_hashes,
                   "model_provenance": config.get("provenance", {}).get("models", {}).get("emotion"),
                   "shared_threshold": all_shared,
                   "all_development_shared_threshold": all_shared,
                   "fit_details": final_details, "test_used": False,
                   "selection": {"criterion": "Prefer candidates improving micro F1 without reducing macro F1 or mixed-label recall; otherwise export objective-specific highest micro F1 research tradeoff", "selected_method": selected,
                                 "balanced_improvement_candidates": balanced_candidates,
                                 "oof_micro_f1": results[selected]["micro_f1"],
                                 "historical_baseline_micro_f1": results["fixed_0.3"]["micro_f1"],
                                 "fit": "Refitted on all development after OOF method comparison"},
                   "limitations": warnings,
                   "policy": "Independent scores; preserve selected neutral/emotion conflicts for human review. No automatic promotion."}
    stability = {label: {"min": float(min(row["per_label_thresholds"][label] for row in folds)),
                         "max": float(max(row["per_label_thresholds"][label] for row in folds)),
                         "fallback_folds": sum(label in row["fallback_labels"] for row in folds)} for label in LABELS}
    report = {"schema": "threshold-refinement-v1", "created_at": datetime.now(timezone.utc).isoformat(),
              "rows": len(development), "unique_texts": len(set(hashes)), "test_used": False,
              "hashes": input_hashes, "model_provenance": config.get("provenance"), "methods": results,
              "selected_method": selected, "folds": folds, "threshold_stability": stability,
              "balanced_improvement_candidates": balanced_candidates,
              "method_descriptions": {"regularized_per_label": "Training-only support-gated coordinate search on joint micro F1; rare labels fall back to training shared threshold. No explicit shrinkage penalty.",
                                      "regularized_macro_per_label": "Training-only support-gated per-label F1 maximization; rare labels use training shared threshold. Exploratory addition after reviewing micro-objective results.",
                                      "fine_shared": "Training-only shared grid threshold selected separately inside each fold",
                                      "fixed_0.3": "Historical comparator selected earlier using all development data",
                                      "fixed_0.5": "Original fixed application threshold"},
              "settings": {"folds": 5, "seed": 42, "threshold_grid": GRID.tolist(), "minimum_positive_support": 25,
                           "minimum_negative_support": 25, "coordinate_max_passes": 5, "bootstrap_repetitions": 500},
              "warnings": warnings, "truncated_rows": sum(truncated)}
    out.mkdir(parents=True, exist_ok=False)
    with (out / "oof_predictions.jsonl").open("x", encoding="utf-8") as output:
        for index, source in enumerate(development.to_dict("records")):
            output.write(json.dumps({"post_id": source["post_id"], "text_sha256": source["text_sha256"],
                "fold": int(assignment[index]), "human_labels": [label for label, present in zip(LABELS, truth[index]) if present],
                "predictions": {method: [label for label, present in zip(LABELS, guessed[index]) if present]
                                for method, guessed in predictions.items()}}, allow_nan=False) + "\n")
    report["oof_predictions_sha256"] = file_digest(out / "oof_predictions.jsonl")
    with (out / "report.json").open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, allow_nan=False)
    calibration["source"] = {**input_hashes, "report_sha256": file_digest(out / "report.json")}
    with (out / "calibration.json").open("x", encoding="utf-8") as output:
        json.dump(calibration, output, indent=2, allow_nan=False)
    return report


def apply_calibration(scores, thresholds):
    """Render learned thresholds while retaining the established review policy."""
    if set(scores) != set(LABELS) or set(thresholds) != set(LABELS):
        raise ValueError("Calibration needs every label")
    for value in thresholds.values():
        if not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 < value < 1:
            raise ValueError("Calibration thresholds must be finite and between zero and one")
    # interpret validates scores, then apply independent learned decisions using
    # a virtual thresholded score map to reuse neutral/uncertainty semantics.
    interpret(scores)
    return interpret({label: 1. if scores[label] >= thresholds[label] else 0. for label in LABELS})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = refine(args.reference, args.predictions, args.out,
                    lambda fold, total: print(f"Completed fold {fold}/{total}", flush=True))
    print(json.dumps({"selected_method": result["selected_method"],
                      "micro_f1": {method: values["micro_f1"] for method, values in result["methods"].items()}}, indent=2))
