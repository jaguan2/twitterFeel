"""Pinned published human annotations; development-only GoEmotions reproduction."""
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import re
import urllib.request
import uuid

import pandas as pd

from analysis_engine import AnalysisEngine
from benchmark import benchmark
from emotion_schema import file_digest
from label_schema import LABELS, SCHEMA_VERSION, text_hash

REPOSITORY = "google-research/google-research"
COMMIT_API = f"https://api.github.com/repos/{REPOSITORY}/commits?path=goemotions/data&per_page=1"
LIMITATIONS = [
    "This model was trained on GoEmotions. This is an in-domain reference reproduction, not an independent unseen-domain test.",
    "The model card reports threshold optimization on GoEmotions test data; the official test split is reserved and is not scored here.",
    "Reddit English comments and published rater agreement do not establish accuracy on tweets or this project's uploads.",
    "Human annotations can disagree; no user identifiers are present in the official reduced splits, so intervals resample posts.",
    "All official development rows and original labels are preserved, including neutral plus emotion labels and repeated texts; this differs from the application's exclusive-neutral annotation rubric.",
]


def _fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "TwitterFeel-research-benchmark/0.2.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def _json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def prepare_goemotions(root, revision=None):
    root = Path(root).resolve()
    if revision is None:
        revision = json.loads(_fetch(COMMIT_API))[0]["sha"]
    if not isinstance(revision, str) or not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("GoEmotions revision must be a full Git commit SHA")
    folder = root / "runs/public_benchmarks/goemotions" / revision / "official-v1"
    manifest_path = folder / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for relative, digest in manifest["artifact_sha256"].items():
            if file_digest(folder / relative) != digest:
                raise ValueError("Cached GoEmotions artifact differs from its recorded hash")
        return manifest
    folder.mkdir(parents=True, exist_ok=True)
    base = f"https://raw.githubusercontent.com/{REPOSITORY}/{revision}/"
    paths = {"dev.tsv": "goemotions/data/dev.tsv", "test.tsv": "goemotions/data/test.tsv",
             "emotions.txt": "goemotions/data/emotions.txt", "LICENSE": "LICENSE",
             "README.md": "goemotions/README.md"}
    sources = {}
    for name, path in paths.items():
        raw = _fetch(base + path)
        target = folder / name
        if target.exists():
            if target.read_bytes() != raw:
                raise ValueError("Existing partial source differs from the pinned official source")
        else:
            with target.open("xb") as output:
                output.write(raw)
        sources[name] = {"url": base + path, "sha256": file_digest(folder / name)}
    labels = (folder / "emotions.txt").read_text(encoding="utf-8").splitlines()
    if labels != list(LABELS):
        raise ValueError("Official GoEmotions label order differs from the configured vocabulary")
    records, raw_counts = {}, {}
    deduplication = {}
    for filename, split in (("dev.tsv", "development"), ("test.tsv", "test")):
        rows = []
        neutral_conflicts = 0
        seen_ids = set()
        content = (folder / filename).read_text(encoding="utf-8")
        for index, item in enumerate(csv.reader(io.StringIO(content), delimiter="\t", quoting=csv.QUOTE_NONE)):
            if len(item) != 3:
                raise ValueError("Official GoEmotions TSV row must have text, labels, and comment ID")
            text, ids, identifier = item
            indices = [int(value) for value in ids.split(",")]
            if (not identifier.strip() or identifier in seen_ids or not text.strip() or not indices
                    or len(indices) != len(set(indices)) or any(i < 0 or i >= len(labels) for i in indices)):
                raise ValueError("Invalid official GoEmotions annotation")
            human_labels = [labels[i] for i in sorted(indices)]
            seen_ids.add(identifier)
            if "neutral" in human_labels and len(human_labels) > 1:
                neutral_conflicts += 1
            rows.append({"post_id": "goemotions:" + identifier, "source_row": index,
                         "text": text, "text_sha256": text_hash(text), "split": split,
                         "human_labels": ";".join(human_labels), "review_status": "reviewed",
                         "reference_origin": "published_human_annotations", "source_comment_id": identifier})
        raw_counts[split] = len(rows)
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["text_sha256"]].append(row)
        conflicts = {digest for digest, items in grouped.items() if len({r["human_labels"] for r in items}) > 1}
        deduplication[split] = {"neutral_emotion_overlap_rows_retained": neutral_conflicts,
                                "conflicting_text_rows_retained": sum(len(grouped[k]) for k in conflicts),
                                "consistent_duplicate_rows_retained": sum(len(items)-1 for digest, items in grouped.items() if digest not in conflicts)}
        records[split] = rows
    test_hashes = {row["text_sha256"] for row in records["test"]}
    test_ids = {row["post_id"] for row in records["test"]}
    overlapping = [r for r in records["development"] if r["text_sha256"] in test_hashes or r["post_id"] in test_ids]
    deduplication["development_test_overlap_rows_retained_official_splits"] = len(overlapping)
    review = pd.DataFrame(records["development"] + records["test"])
    review.to_csv(folder / "reference.csv", index=False, mode="x")
    pd.DataFrame(records["development"])[["post_id", "source_row", "text", "text_sha256"]].to_csv(folder / "development.csv", index=False, mode="x")
    artifacts = {name: file_digest(folder / name) for name in [*paths, "reference.csv", "development.csv"]}
    manifest = {"id": folder.relative_to(root).as_posix(), "name": "GoEmotions published human reference",
                "revision": revision, "created_at": datetime.now(timezone.utc).isoformat(),
                "review_path": (folder / "reference.csv").relative_to(root).as_posix(),
                "dataset_path": (folder / "development.csv").relative_to(root).as_posix(),
                "development_rows": len(records["development"]), "test_rows": len(records["test"]),
                "raw_rows": raw_counts, "deduplication": deduplication,
                "reference_origin": "published_human_annotations", "labels": list(LABELS), "sources": sources,
                "artifact_sha256": artifacts, "license": {"repository_license": "Apache-2.0", "file": "LICENSE",
                    "note": "Official repository license retained verbatim; Reddit comments retain source/content considerations described by the dataset authors."},
                "limitations": LIMITATIONS, "test_policy": "Reserved; not scored or used to choose thresholds"}
    manifest["reference_policy"] = "Official splits preserved without removing or rewriting human annotations"
    _json(manifest_path, manifest)
    return manifest


def score_goemotions_development(root, progress=None, revision=None, engine=None, reuse_predictions=None):
    root = Path(root).resolve()
    manifest = prepare_goemotions(root, revision)
    engine = engine or AnalysisEngine(root)
    frame = pd.read_csv(root / manifest["dataset_path"], dtype={"post_id": str, "text": str, "text_sha256": str}, keep_default_na=False)
    base = root / manifest["id"]
    folder = base / ("development_run_" + uuid.uuid4().hex[:12])
    folder.mkdir(exist_ok=False)
    config = {"schema": SCHEMA_VERSION, "status": "running", "rows": len(frame), "threshold": .5,
              "task": "emotion", "tasks": ["emotion"], "model": "SamLowe/roberta-base-go_emotions",
              "reference_origin": "published_human_annotations", "benchmark_kind": "in_domain_reference_reproduction",
              "dataset": manifest, "limitations": LIMITATIONS, "test_scored": False}
    version_path = root / "app_version.json"
    config["app_version"] = json.loads(version_path.read_text(encoding="utf-8"))["version"] if version_path.exists() else "0.2.0"
    _json(folder / "config.json", config)
    try:
        config["provenance"] = engine.provenance(["emotion"])
        reusable = {}
        if reuse_predictions is not None:
            from workbench_data import load_predictions
            reuse_predictions = Path(reuse_predictions)
            previous = json.loads((reuse_predictions.parent / "config.json").read_text(encoding="utf-8"))
            if previous.get("provenance", {}).get("models", {}).get("emotion") != config["provenance"]["models"]["emotion"]:
                raise ValueError("Reused scores must have identical pinned emotion model provenance")
            if previous.get("predictions_sha256") != file_digest(reuse_predictions):
                raise ValueError("Reused predictions differ from their recorded hash")
            reusable = {row["text_sha256"]: row for row in load_predictions(reuse_predictions).to_dict("records")}
            config["reused_predictions_sha256"] = file_digest(reuse_predictions)
        reused_count = 0
        statuses, labels = Counter(), Counter()
        truncated = multiple = invalid = 0
        with (folder / "predictions.jsonl").open("x", encoding="utf-8") as output:
            for index, source in enumerate(frame.to_dict("records"), 1):
                cached = reusable.get(source["text_sha256"])
                if cached is not None:
                    result = {key: value for key, value in cached.items() if key not in ("post_id", "source_row", "source_post_id", "user_id")}
                    reused_count += 1
                else:
                    result = engine.infer(source["text"], threshold=.5, tasks=["emotion"])
                row = {**source, **result}
                output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                statuses[row["status"]] += 1
                labels.update(row["labels"])
                truncated += bool(row.get("truncated"))
                multiple += sum(label != "neutral" for label in row["labels"]) > 1
                invalid += row["status"] == "invalid"
                if progress:
                    progress(index, len(frame))
        summary = {"rows": len(frame), "status_counts": dict(statuses), "label_counts": dict(labels),
                   "sentiment_counts": {}, "truncated": truncated, "multiple_emotions": multiple, "invalid_rows": invalid}
        _json(folder / "summary.json", summary)
        config.update(predictions_sha256=file_digest(folder / "predictions.jsonl"), summary_sha256=file_digest(folder / "summary.json"),
                      reused_rows=reused_count, newly_scored_rows=len(frame)-reused_count)
        config["status"] = "complete"
        temporary = folder / "config.tmp"
        _json(temporary, config)
        temporary.replace(folder / "config.json")
        report = benchmark(root / manifest["review_path"], folder / "predictions.jsonl", published_reference=True)
        report.update(reference_origin="published_human_annotations", benchmark_kind="in_domain_reference_reproduction",
                      reference_selection={"raw_rows": manifest["raw_rows"], "deduplication": manifest["deduplication"],
                                           "evaluated_development_rows": manifest["development_rows"]},
                      limitations=LIMITATIONS, test_scored=False, run_id=folder.relative_to(root).as_posix())
        _json(folder / "benchmark.json", report)
    except Exception as exc:
        config.update(status="failed", error=str(exc))
        temporary = folder / "config.tmp"
        _json(temporary, config)
        temporary.replace(folder / "config.json")
        raise
    return {"manifest": manifest, "run_id": folder.relative_to(root).as_posix(), "report": report}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--revision")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--reuse-predictions", type=Path)
    args = parser.parse_args()
    if args.evaluate:
        result = score_goemotions_development(args.root, lambda n, total: print(f"Scored {n}/{total}", flush=True) if n % 100 == 0 or n == total else None, args.revision, reuse_predictions=args.reuse_predictions)
        print(json.dumps({"run_id": result["run_id"], "recommended_threshold": result["report"]["recommended_threshold"]}))
    else:
        print(json.dumps(prepare_goemotions(args.root, args.revision), indent=2))
