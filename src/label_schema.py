"""Independent multi-label vocabulary; deliberately separate from legacy V4."""
import hashlib
import numpy as np

SCHEMA_VERSION = "goemotions-multilabel-v1"
LABELS = (
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
    "remorse", "sadness", "surprise", "neutral",
)
# Optional display aids, not an exhaustive or validated psychological ontology.
RELATED = {
    "anger-related": ("annoyance", "anger"),
    "sadness-related": ("disappointment", "sadness", "grief"),
    "fear-related": ("nervousness", "fear"),
    "joy-related": ("amusement", "excitement", "joy"),
    "affection-related": ("caring", "love"),
}


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_thresholds(thresholds):
    """Require an explicit, finite cutoff for every emotion in a profile."""
    if not isinstance(thresholds, dict) or set(thresholds) != set(LABELS):
        raise ValueError("A threshold profile must contain every GoEmotions label")
    if any(isinstance(value, (bool, str)) for value in thresholds.values()):
        raise ValueError("Profile thresholds must be numbers between zero and one")
    try:
        values = np.asarray([thresholds[label] for label in LABELS], dtype=float)
    except (TypeError, ValueError):
        raise ValueError("Profile thresholds must be numbers between zero and one") from None
    if not np.isfinite(values).all() or (values <= 0).any() or (values >= 1).any():
        raise ValueError("Profile thresholds must be finite and between zero and one")
    return {label: float(value) for label, value in zip(LABELS, values)}


def interpret(scores, threshold=.5, thresholds=None):
    """Threshold independent scores, retaining neutral conflicts for review."""
    if not 0 < threshold < 1:
        raise ValueError("threshold must be between zero and one")
    if set(scores) != set(LABELS):
        raise ValueError("Expected the complete GoEmotions vocabulary")
    values = np.asarray([scores[label] for label in LABELS], dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Scores must be finite and between zero and one")
    cutoffs = validate_thresholds(thresholds) if thresholds is not None else dict.fromkeys(LABELS, threshold)
    selected = [label for label in LABELS if scores[label] >= cutoffs[label]]
    emotions = [label for label in selected if label != "neutral"]
    if "neutral" in selected and emotions:
        status, reason = "review_needed", "neutral_emotion_conflict"
    elif emotions:
        status, reason = "labeled", ""
    elif "neutral" in selected:
        status, reason = "neutral", ""
    else:
        status, reason = "uncertain", "no_label_above_threshold"
    return {"labels": selected, "status": status, "review_reason": reason,
            "related_groups": [name for name, labels in RELATED.items()
                               if any(label in emotions for label in labels)]}
