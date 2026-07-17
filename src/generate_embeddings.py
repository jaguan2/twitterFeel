"""Generate 384-dim sentence embeddings for every tweet in dataset_godknowswhat.csv.

Uses sentence-transformers/all-MiniLM-L6-v2 via the raw transformers API
(mean-pooling + L2-normalization, per the model card). Writes a single
NumPy file aligned row-by-row with the source CSV.

Output: tweet_embeddings.npy, shape (n_rows, 384), float32.

Usage:
    python generate_embeddings.py [INPUT_CSV [OUTPUT_NPY]]

Defaults to the Mental-Health-Twitter pipeline files; pass paths to embed
another corpus (any CSV with a cleaned_text column).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from paths import DATA_INTERIM

INPUT_CSV = DATA_INTERIM / "dataset_godknowswhat.csv"
OUTPUT_NPY = DATA_INTERIM / "tweet_embeddings.npy"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 64
MAX_LENGTH = 128
EMBED_DIM = 384


def mean_pool(model_output, attention_mask):
    token_emb = model_output[0]
    mask = attention_mask.unsqueeze(-1).expand(token_emb.size()).float()
    return (token_emb * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def main(input_csv: Path = INPUT_CSV, output_npy: Path = OUTPUT_NPY) -> None:
    df = pd.read_csv(input_csv)
    texts = df["cleaned_text"].fillna("").astype(str).tolist()
    print(f"Embedding {len(texts):,} tweets with {MODEL_NAME}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    out = np.zeros((len(texts), EMBED_DIM), dtype=np.float32)

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
            output = model(**enc)
            emb = mean_pool(output, enc["attention_mask"])
            emb = F.normalize(emb, p=2, dim=1)
            out[start : start + len(batch)] = emb.cpu().numpy()
            if (start // BATCH_SIZE) % 20 == 0:
                print(f"  {start + len(batch):>6,}/{len(texts):,}")

    np.save(output_npy, out)
    print(f"\nWrote {out.shape} -> {output_npy.name}")


if __name__ == "__main__":
    main(*[Path(a) for a in sys.argv[1:3]])
