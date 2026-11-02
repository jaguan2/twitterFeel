"""Label existing posts, create blind review samples, and evaluate human labels.

Run `python src/label_posts.py --help` for the three workflow commands.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, f1_score, jaccard_score

from twitterfeel.common.provenance import file_digest
from twitterfeel.labeling.label_schema import LABELS, SCHEMA_VERSION, interpret, text_hash

DEFAULT_MODEL = "SamLowe/roberta-base-go_emotions"


def read_posts(path, text_column=None):
    strings = {name: str for name in ("user_id", "post_id", "source_post_id", "text_sha256",
                                      "text", "post_text", "cleaned_text")}
    if text_column:
        strings[text_column] = str
    df = pd.read_csv(path, dtype=strings, keep_default_na=False)
    column = text_column or next((c for c in ("text", "post_text", "cleaned_text") if c in df), None)
    if column not in df:
        raise ValueError("Input needs text, post_text, or cleaned_text; or supply --text-column")
    df["text"] = df[column].astype(str)
    df.attrs["text_column"] = column
    if "source_row" not in df:
        df["source_row"] = np.arange(len(df))
    if "post_id" not in df:
        df["post_id"] = "post_" + df.source_row.astype(str)
    if df.post_id.str.strip().eq("").any():
        raise ValueError("post_id must not be blank")
    if df.post_id.duplicated().any():
        # The bundled raw export repeats some source IDs. Preserve them and
        # distinguish export rows; never assume raw/cleaned row alignment.
        if column != "post_text" or "text_sha256" in df:
            raise ValueError("post_id must be unique")
        df["source_post_id"] = df.post_id
        df["post_id"] = df.post_id + ":row_" + df.source_row.astype(str)
        if df.post_id.duplicated().any():
            raise ValueError("Source rows must uniquely identify duplicate source IDs")
    hashes = df.text.map(text_hash)
    if "text_sha256" in df and not (df.text_sha256 == hashes).all():
        raise ValueError("Text differs from its recorded hash")
    df["text_sha256"] = hashes
    return df


def new_directory(path):
    if path.exists():
        raise ValueError(f"Output already exists: {path}; choose a new name")
    path.mkdir(parents=True)


def sample(args):
    df = read_posts(args.csv, getattr(args, "text_column", None))
    # Duplicate text must not appear across development/test review partitions.
    text_column = df.attrs["text_column"]
    df = df[df.text.str.strip().ne("")].drop_duplicates("text_sha256").copy()
    if len(df) < 2:
        raise ValueError("Need at least two unique nonempty posts")
    rng = np.random.default_rng(args.seed)
    groups = (df.user_id.replace("", "missing-user") if "user_id" in df
              else df.text_sha256)
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("Need at least two users for separate review partitions")
    rng.shuffle(unique)
    test_groups = unique[:max(1, min(len(unique)-1, round(len(unique)*.25)))]
    df["split"] = np.where(groups.isin(test_groups), "test", "development")
    n = min(args.size, len(df))
    n_test = min(max(1, round(n*.25)), int((df.split == "test").sum()))
    n_dev = min(n-n_test, int((df.split == "development").sum()))
    n_test = min(n-n_dev, int((df.split == "test").sum()))
    chosen = pd.concat([df[df.split == "development"].sample(n_dev, random_state=args.seed),
                        df[df.split == "test"].sample(n_test, random_state=args.seed)])
    chosen = chosen.sample(frac=1, random_state=args.seed)
    columns = ["post_id", "source_row", "text_sha256", "text", "split"]
    if "user_id" in chosen:
        columns.insert(1, "user_id")
    if "source_post_id" in chosen:
        columns.insert(1, "source_post_id")
    review = chosen[columns].copy()
    review["human_labels"] = ""
    review["review_status"] = "pending"
    review["notes"] = ""
    new_directory(args.out)
    review.to_csv(args.out / "review.csv", index=False)
    (args.out / "sample.json").write_text(json.dumps({
        "source_sha256": file_digest(args.csv), "text_column": text_column,
        "seed": args.seed, "rows": len(review),
        "schema": SCHEMA_VERSION, "labels": LABELS,
        "split_unit": "user" if "user_id" in df else "unique_text",
        "note": "Random pilot sample, not balanced coverage of rare emotions. No model labels shown.",
    }, indent=2), encoding="utf-8")
    print(f"Created {len(review)} blind review rows: {args.out / 'review.csv'}")


def label(args):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    df = read_posts(args.csv, getattr(args, "text_column", None))
    if args.limit:
        df = df.head(args.limit)
    if df.empty:
        raise ValueError("No posts to label")
    if args.out.exists():
        raise ValueError("Output exists; choose a new run name")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, revision=args.revision, use_safetensors=True)
    # Pin the tokenizer to the same resolved remote commit as the weights.
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=getattr(model.config, "_commit_hash", None) or args.revision)
    if args.max_length > tokenizer.model_max_length:
        raise ValueError(f"max-length exceeds tokenizer limit ({tokenizer.model_max_length})")
    names = {int(i): name.lower() for i, name in model.config.id2label.items()}
    if set(names.values()) != set(LABELS) or set(names) != set(range(len(LABELS))):
        raise ValueError("Model must have the complete 28-label GoEmotions vocabulary")
    if model.config.problem_type != "multi_label_classification":
        raise ValueError("Model must declare multi_label_classification")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    new_directory(args.out)
    config = {"schema": SCHEMA_VERSION, "model": args.model, "revision": args.revision,
              "resolved_commit": getattr(model.config, "_commit_hash", None),
              "model_config": model.config.to_dict(), "source_sha256": file_digest(args.csv),
              "text_column": df.attrs["text_column"],
              "threshold": args.threshold, "max_length": args.max_length,
              "rows": len(df), "device": device, "status": "running",
              "score_meaning": "Independent model scores, not calibrated certainty or intensity",
              "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "numpy", "pandas")}}
    from twitterfeel.common import provenance
    config["pipeline_source_sha256"] = {
        name: file_digest(Path(__file__).with_name(name))
        for name in ("label_posts.py", "label_schema.py")}
    config["pipeline_source_sha256"]["common/provenance.py"] = file_digest(provenance.__file__)
    local_model = Path(args.model)
    if local_model.is_dir():
        config["local_model_sha256"] = {
            p.name: file_digest(p) for p in sorted(local_model.iterdir())
            if p.is_file() and p.suffix in (".json", ".txt", ".safetensors")}
    config_path = args.out / "config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    summary = {"rows": len(df), "status_counts": {}, "label_counts": {k: 0 for k in LABELS}, "truncated": 0}
    with (args.out / "predictions.jsonl").open("w", encoding="utf-8") as output:
        for start in range(0, len(df), args.batch):
            records = df.iloc[start:start+args.batch].to_dict("records")
            valid = [r for r in records if r["text"].strip()]
            predicted = {}
            if valid:
                texts = [r["text"] for r in valid]
                lengths = [len(ids) for ids in tokenizer(texts, truncation=False)["input_ids"]]
                enc = tokenizer(texts, truncation=True, max_length=args.max_length,
                                padding=True, return_tensors="pt").to(device)
                with torch.inference_mode():
                    values = model(**enc).logits.sigmoid().cpu().numpy()
                for row, scores, length in zip(valid, values, lengths):
                    mapped = {names[i]: float(value) for i, value in enumerate(scores)}
                    predicted[row["post_id"]] = {"scores": mapped, **interpret(mapped, args.threshold),
                                                 "truncated": length > args.max_length}
            for row in records:
                result = {k: row[k] for k in ("post_id", "source_row", "text_sha256", "text")}
                if "source_post_id" in row:
                    result["source_post_id"] = row["source_post_id"]
                result.update(predicted.get(row["post_id"], {
                    "scores": {}, "labels": [], "related_groups": [], "status": "invalid",
                    "review_reason": "empty_text", "truncated": False}))
                if result["truncated"]:
                    result["status"] = "review_needed"
                    result["review_reason"] = ";".join(filter(None, [result["review_reason"], "text_truncated"]))
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
                status = result["status"]
                summary["status_counts"][status] = summary["status_counts"].get(status, 0) + 1
                summary["truncated"] += result["truncated"]
                for name in result["labels"]:
                    summary["label_counts"][name] += 1
            print(f"Labeled {min(start+args.batch, len(df))}/{len(df)}", flush=True)
    config["status"] = "complete"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def evaluate(args):
    review = read_posts(args.review)
    required = {"human_labels", "review_status", "split"}
    if not required.issubset(review):
        raise ValueError(f"Review needs {sorted(required)}")
    if not review.review_status.isin(["pending", "reviewed", "unclear", "invalid"]).all():
        raise ValueError("Unknown review_status")
    predictions = {}
    with args.predictions.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["post_id"] in predictions:
                raise ValueError("Duplicate prediction post_id")
            predictions[row["post_id"]] = row
    selected = review[review.split == args.split]
    truth, guessed, statuses = [], [], []
    for row in selected.to_dict("records"):
        if row["review_status"] != "reviewed":
            continue
        labels = {label.strip() for label in row["human_labels"].split(";") if label.strip()}
        if not labels or not labels.issubset(LABELS) or ("neutral" in labels and len(labels) > 1):
            raise ValueError(f"Invalid human_labels for {row['post_id']}")
        prediction = predictions.get(row["post_id"])
        if prediction is None or prediction["text_sha256"] != row["text_sha256"]:
            raise ValueError(f"Missing or text-mismatched prediction: {row['post_id']}")
        if not set(prediction["labels"]).issubset(LABELS):
            raise ValueError("Unknown predicted label")
        truth.append([int(label in labels) for label in LABELS])
        guessed.append([int(label in prediction["labels"]) for label in LABELS])
        statuses.append(prediction["status"])
    if not truth:
        raise ValueError("No human-reviewed rows in this split; automatic labels are not ground truth")
    y, p = np.asarray(truth), np.asarray(guessed)
    precision, recall, f1, support = precision_recall_fscore_support(y, p, zero_division=0)
    result = {"split": args.split, "reviewed_rows": len(y), "split_rows": len(selected),
              "excluded_review_status_counts": selected[selected.review_status != "reviewed"].review_status.value_counts().to_dict(),
              "micro_f1": float(f1_score(y, p, average="micro", zero_division=0)),
              "macro_f1_all_labels": float(f1.mean()),
              "macro_f1_supported_labels": float(f1[support > 0].mean()),
              "exact_match": float((y == p).all(axis=1).mean()),
              "mean_label_jaccard": float(jaccard_score(y, p, average="samples", zero_division=0)),
              "automatic_acceptance_fraction": sum(s in ("labeled", "neutral") for s in statuses)/len(statuses),
              "review_sha256": file_digest(args.review), "predictions_sha256": file_digest(args.predictions),
              "per_label": {label: {"precision": float(a), "recall": float(b), "f1": float(c), "support": int(d)}
                            for label, a, b, c, d in zip(LABELS, precision, recall, f1, support)}}
    # Uncertain predictions are empty sets and count as misses, never silently dropped.
    if args.out.exists():
        raise ValueError("Evaluation output exists; choose a new name")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Evaluated {len(y)} reviewed posts; micro F1: {result['micro_f1']:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("sample", help="Create a blind human review CSV")
    p.add_argument("csv", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--size", type=int, default=400)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--text-column", help="Defaults to text, post_text, then cleaned_text")
    p.set_defaults(run=sample)
    p = commands.add_parser("label", help="Assign multiple emotions to existing posts")
    p.add_argument("csv", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model or local model directory")
    p.add_argument("--revision", default="main")
    p.add_argument("--threshold", type=float, default=.5)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--limit", type=int)
    p.add_argument("--text-column", help="Defaults to text, post_text, then cleaned_text")
    p.set_defaults(run=label)
    p = commands.add_parser("evaluate", help="Compare predictions with reviewed human labels")
    p.add_argument("review", type=Path)
    p.add_argument("predictions", type=Path)
    p.add_argument("--split", choices=["development", "test"], default="development")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(run=evaluate)
    args = parser.parse_args()
    if getattr(args, "size", 2) < 2 or getattr(args, "batch", 1) < 1:
        parser.error("size must be >= 2 and batch must be positive")
    if not 0 < getattr(args, "threshold", .5) < 1:
        parser.error("threshold must be between zero and one")
    if getattr(args, "max_length", 2) < 2 or (getattr(args, "limit", None) is not None and args.limit < 1):
        parser.error("max-length must be >= 2 and limit must be positive")
    try:
        args.run(args)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
