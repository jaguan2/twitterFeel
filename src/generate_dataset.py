"""Generate dataset_godknowswhat.csv with emotion labels for each tweet.

Reads cleaned_Mental-Health-Twitter.xls (CSV-format despite the extension),
runs a Hugging Face 6-class emotion classifier on the cleaned_text column,
and writes a CSV containing every original column plus emotion (string) and
emotion_encoded (int) using the README mapping:
    0=sadness, 1=joy, 2=love, 3=anger, 4=fear, 5=surprise
"""
from __future__ import annotations

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from paths import DATA_INTERIM, DATA_RAW

INPUT_CSV = DATA_RAW / "cleaned_Mental-Health-Twitter.xls"
OUTPUT_CSV = DATA_INTERIM / "dataset_godknowswhat.csv"
MODEL_NAME = "bhadresh-savani/distilbert-base-uncased-emotion"
BATCH_SIZE = 64
MAX_LENGTH = 128

EXPECTED_ID2LABEL = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise",
}


def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df):,} rows from {INPUT_CSV.name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    # Sanity-check the model's label mapping matches the README's emotion encoding.
    actual = {int(k): v for k, v in model.config.id2label.items()}
    if actual != EXPECTED_ID2LABEL:
        raise RuntimeError(
            f"Model id2label {actual} does not match expected {EXPECTED_ID2LABEL}"
        )

    texts = df["cleaned_text"].fillna("").astype(str).tolist()
    preds: list[int] = []

    with torch.no_grad():
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start : start + BATCH_SIZE]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
            if (start // BATCH_SIZE) % 20 == 0:
                print(f"  {start + len(batch):>6,}/{len(texts):,}")

    df["emotion_encoded"] = preds
    df["emotion"] = df["emotion_encoded"].map(EXPECTED_ID2LABEL)

    df.to_csv(OUTPUT_CSV, index=True)
    print(f"\nWrote {len(df):,} rows to {OUTPUT_CSV.name}")
    print("\nEmotion distribution:")
    print(df["emotion"].value_counts())


if __name__ == "__main__":
    main()
