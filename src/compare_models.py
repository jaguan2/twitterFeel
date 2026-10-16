"""One pinned alternative model, same development rows and grouped OOF protocol.

This is an exploratory, in-domain comparison, never an official-test evaluation.
"""
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import time
import uuid

import numpy as np

from emotion_schema import file_digest
from label_posts import read_posts
from label_schema import LABELS, SCHEMA_VERSION, interpret
from threshold_tuning import METHODS, paired_bootstrap, refine

MODEL = "cirimus/modernbert-base-go-emotions"
REVISION = "cb5ed17bc26c8cd1f017fabe700a5d723f825287"
WARNINGS = [
    "Both models were trained on GoEmotions; this development comparison is not independent Twitter/upload accuracy.",
    "Only official development annotations are used. Official test data is never scored or used to tune thresholds.",
    "This candidate and experiment were chosen after earlier results; comparisons are exploratory and subject to selection optimism.",
    "Both models receive original text with a 256-token limit. Different tokenizers can retain different amounts of text within that limit.",
    "The candidate card identifies this March 2026 model as superseded; this bounded CPU experiment evaluates one candidate, not the entire model catalog.",
]


def _write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def validate_model_config(config):
    names = {int(key): value for key, value in config.id2label.items()}
    if names != dict(enumerate(LABELS)) or config.problem_type != "multi_label_classification":
        raise ValueError("Candidate must have the exact 28-label GoEmotions multi-label vocabulary")
    if config.model_type != "modernbert":
        raise ValueError("Only the pinned ModernBERT candidate architecture is allowed")
    return names


def download_candidate(root):
    from huggingface_hub import snapshot_download
    path = Path(root) / "runs/model_cache/modernbert-go-emotions"
    snapshot_download(MODEL, revision=REVISION, local_dir=path,
                      allow_patterns=["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
                                      "special_tokens_map.json", "vocab.json", "merges.txt", "README.md", "LICENSE"])
    return path


class CandidateModel:
    def __init__(self, root, download=False):
        import torch
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
        self.path = Path(root) / "runs/model_cache/modernbert-go-emotions"
        if not (self.path / "model.safetensors").is_file():
            if not download:
                raise ValueError("Pinned candidate is not cached; use --run to download the public model")
            download_candidate(root)
        metadata = self.path / ".cache/huggingface/download/config.json.metadata"
        if not metadata.is_file() or metadata.read_text(encoding="utf-8").splitlines()[0] != REVISION:
            raise ValueError("Candidate cache revision does not match the pinned public snapshot")
        config = AutoConfig.from_pretrained(self.path, local_files_only=True, trust_remote_code=False)
        self.names = validate_model_config(config)
        torch.set_num_threads(4)
        config.reference_compile = False
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True, trust_remote_code=False)
        self.model, load_info = AutoModelForSequenceClassification.from_pretrained(
            self.path, config=config, local_files_only=True, use_safetensors=True,
            trust_remote_code=False, attn_implementation="sdpa", output_loading_info=True)
        if load_info.get("missing_keys") or load_info.get("mismatched_keys") or load_info.get("error_msgs"):
            raise ValueError("Candidate weights did not load completely: " + str(load_info))
        self.model.eval()

    def provenance(self):
        files = {path.name: file_digest(path) for path in sorted(self.path.iterdir()) if path.is_file()}
        return {"models": {"emotion": {"id": MODEL, "revision": REVISION, "files_sha256": files,
                                      "runtime_format": "safetensors"}},
                "packages": {name: version(name) for name in ("torch", "transformers", "tokenizers", "safetensors", "numpy")},
                "task_configs": {"emotion": {"activation": "sigmoid", "max_length": 256,
                                                "preprocessing": "original_text", "batch_size": 16,
                                                "attention": "sdpa", "reference_compile": False}},
                "inference": "local_cpu_only", "model_card": "https://huggingface.co/" + MODEL, "license": "MIT per pinned model card"}

    def infer_batch(self, texts):
        import torch
        lengths = [len(ids) for ids in self.tokenizer(texts, truncation=False)["input_ids"]]
        tokens = self.tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
        with torch.inference_mode():
            values = self.model(**tokens).logits.sigmoid().cpu().numpy()
        if values.shape != (len(texts), len(LABELS)) or not np.isfinite(values).all():
            raise ValueError("Candidate returned incompatible or nonfinite predictions")
        results = []
        for scores, length in zip(values, lengths):
            scores = dict(zip(LABELS, map(float, scores)))
            row = {"scores": scores, "truncated": length > 256, **interpret(scores)}
            if row["truncated"]:
                row["status"] = "review_needed"
                row["review_reason"] = ";".join(filter(None, [row["review_reason"], "text_truncated"]))
            results.append(row)
        return results


def compare_oof(baseline_dir, candidate_dir):
    baseline_dir, candidate_dir = Path(baseline_dir), Path(candidate_dir)
    reports, rows = [], []
    for folder in (baseline_dir, candidate_dir):
        report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
        if report["oof_predictions_sha256"] != file_digest(folder / "oof_predictions.jsonl"):
            raise ValueError("OOF predictions differ from their recorded hash")
        reports.append(report)
        rows.append([json.loads(line) for line in (folder / "oof_predictions.jsonl").read_text(encoding="utf-8").splitlines()])
    if reports[0]["hashes"]["reference_sha256"] != reports[1]["hashes"]["reference_sha256"]:
        raise ValueError("Model comparison requires the same pinned human reference")
    if reports[0]["settings"] != reports[1]["settings"]:
        raise ValueError("Model comparison requires the same threshold/CV protocol")
    identity = lambda row: (row["post_id"], row["text_sha256"], row["fold"], row["human_labels"])
    if [identity(row) for row in rows[0]] != [identity(row) for row in rows[1]]:
        raise ValueError("Model comparison requires identical paired posts, labels, and folds")
    truth = np.asarray([[label in row["human_labels"] for label in LABELS] for row in rows[0]])
    groups = [row["text_sha256"] for row in rows[0]]
    comparisons = {}
    for method in METHODS:
        guessed = [np.asarray([[label in row["predictions"][method] for label in LABELS] for row in model_rows]) for model_rows in rows]
        interval = paired_bootstrap(truth, {"fixed_0.3": guessed[0], "candidate": guessed[1]}, groups)["candidate"]
        interval["interpretation"] = "Paired cluster-bootstrap candidate minus current model using the same method; proportion positive is not a p-value"
        comparisons[method] = {"baseline": reports[0]["methods"][method], "candidate": reports[1]["methods"][method],
                               "micro_f1_difference": reports[1]["methods"][method]["micro_f1"] - reports[0]["methods"][method]["micro_f1"],
                               "paired_difference_interval": interval}
    return {"methods": comparisons, "rows": len(rows[0]), "unique_texts": len(set(groups)),
            "folds_verified_identical": True, "hashes": {"baseline_report_sha256": file_digest(baseline_dir / "report.json"),
                "candidate_report_sha256": file_digest(candidate_dir / "report.json"),
                "reference_sha256": reports[0]["hashes"]["reference_sha256"]}}


def run_comparison(root, reference, baseline_dir, candidate=None, progress=None):
    root, reference, baseline_dir = Path(root).resolve(), Path(reference), Path(baseline_dir)
    frame = read_posts(reference)
    frame = frame[frame.split.eq("development") & frame.review_status.eq("reviewed")].copy()
    if frame.empty or frame.text.str.strip().eq("").any():
        raise ValueError("Need nonblank human-reviewed development posts")
    baseline = json.loads((baseline_dir / "report.json").read_text(encoding="utf-8"))
    if baseline["hashes"]["reference_sha256"] != file_digest(reference):
        raise ValueError("Reference differs from the baseline protocol")
    candidate = candidate or CandidateModel(root, download=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:8]
    folder = root / "runs/model_comparisons" / stamp / "modernbert"
    folder.mkdir(parents=True, exist_ok=False)
    config = {"schema": SCHEMA_VERSION, "status": "running", "model": MODEL, "rows": len(frame),
              "threshold": .5, "task": "emotion", "tasks": ["emotion"], "provenance": candidate.provenance(),
              "reference_sha256": file_digest(reference), "test_scored": False, "warnings": WARNINGS,
              "protocol": {"max_length": 256, "batch_size": 16, "cpu_threads": 4, "text": "original"}}
    version_path = root / "app_version.json"
    config["app_version"] = json.loads(version_path.read_text(encoding="utf-8"))["version"] if version_path.is_file() else "0.2.0"
    _write(folder / "config.json", config)
    started = time.monotonic()
    try:
        statuses, labels = Counter(), Counter()
        truncated = multiple = 0
        records = frame.to_dict("records")
        with (folder / "predictions.jsonl").open("x", encoding="utf-8") as output:
            for start in range(0, len(records), 16):
                batch = records[start:start + 16]
                results = candidate.infer_batch([row["text"] for row in batch])
                if len(results) != len(batch):
                    raise ValueError("Candidate batch lost prediction rows")
                for source, result in zip(batch, results):
                    row = {key: source[key] for key in ("post_id", "source_row", "text", "text_sha256")}
                    row.update(result)
                    output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    statuses[row["status"]] += 1
                    labels.update(row["labels"])
                    truncated += bool(row["truncated"])
                    multiple += sum(label != "neutral" for label in row["labels"]) > 1
                if progress:
                    progress(min(start + 16, len(records)), len(records))
        summary = {"rows": len(frame), "status_counts": dict(statuses), "label_counts": dict(labels),
                   "sentiment_counts": {}, "invalid_rows": 0, "truncated": truncated, "multiple_emotions": multiple,
                   "scoring_seconds": time.monotonic() - started}
        _write(folder / "summary.json", summary)
        config.update(status="complete", predictions_sha256=file_digest(folder / "predictions.jsonl"),
                      summary_sha256=file_digest(folder / "summary.json"))
        temporary = folder / "config.tmp"
        _write(temporary, config)
        temporary.replace(folder / "config.json")
        refine(reference, folder / "predictions.jsonl", folder / "refinement")
        comparison = compare_oof(baseline_dir, folder / "refinement")
        comparison.update(schema="emotion-model-comparison-v1", baseline_model="SamLowe/roberta-base-go_emotions",
                          candidate_model=MODEL, candidate_revision=REVISION, warnings=WARNINGS, test_scored=False,
                          scoring_seconds=summary["scoring_seconds"], run_id=folder.relative_to(root).as_posix(),
                          candidate_model_provenance=config["provenance"], baseline_model_provenance=baseline.get("model_provenance"))
        _write(folder.parent / "comparison.json", comparison)
        return comparison
    except Exception as exc:
        config.update(status="failed", error=str(exc))
        temporary = folder / "config.tmp"
        _write(temporary, config)
        temporary.replace(folder / "config.json")
        raise


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    result = run_comparison(args.root, args.reference, args.baseline,
                            progress=lambda n, total: print(f"Scored {n}/{total}", flush=True) if n % 160 == 0 or n == total else None)
    print(json.dumps({"run_id": result["run_id"], "methods": {method: value["micro_f1_difference"] for method, value in result["methods"].items()}}, indent=2))
