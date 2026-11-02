"""Preserved six-class forecasting vocabulary."""
from twitterfeel.common.provenance import file_digest, text_digest

EMOTION_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]
PROBABILITY_COLUMNS = [f"emotion_prob_{name}" for name in EMOTION_NAMES]
