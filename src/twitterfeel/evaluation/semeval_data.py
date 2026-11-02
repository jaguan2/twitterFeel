"""Prepare the official SemEval emotion development reference without relabeling it.

This adapter does not score models or read training/test labels. Dataset terms
restrict use to scientific/research purposes and prohibit redistribution.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from twitterfeel.common.config import app_release, workspace_root
from pathlib import Path
from urllib.request import urlopen
import zipfile

OFFICIAL_URL = "https://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/SemEval2018-Task1-all-data.zip"
DEVELOPMENT_URL = "https://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/E-c/English/2018-E-c-En-dev.zip"
TERMS_URL = "https://competitions.codalab.org/competitions/17751#learn_the_details-terms_and_conditions"
PAPER_URL = "https://aclanthology.org/S18-1001/"
NATIVE_LABELS = ("anger", "anticipation", "disgust", "fear", "joy", "love", "optimism", "pessimism", "sadness", "surprise", "trust")
SHARED_LABELS = ("anger", "disgust", "fear", "joy", "love", "optimism", "sadness", "surprise")
UNREPRESENTED_LABELS = ("anticipation", "pessimism", "trust")
DEV_MEMBERS = ("SemEval2018-Task1-all-data/English/E-c/2018-E-c-En-dev.txt", "2018-E-c-En-dev.txt")
TERMS_MEMBERS = ("SemEval2018-Task1-all-data/README.txt", "SemEval2018-Task1-all-data/README.md", "SemEval2018-Task1-all-data/LICENSE", "README.txt", "README.md", "LICENSE")
MAX_ARCHIVE_BYTES = 30_000_000
MAX_MEMBER_BYTES = 5_000_000


def digest(data):
    return hashlib.sha256(data).hexdigest()


def parse_development(data):
    """Keep all eleven native labels; a shared-label empty set is never neutral."""
    if len(data) > MAX_MEMBER_BYTES:
        raise ValueError("Development member exceeds its size limit")
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig"), newline=""), delimiter="\t", quoting=csv.QUOTE_NONE)
    if reader.fieldnames != ["ID", "Tweet", *NATIVE_LABELS]:
        raise ValueError("Unexpected SemEval English emotion development columns")
    records, seen = [], set()
    for source_row, row in enumerate(reader, 2):
        if source_row > 100_001:
            raise ValueError("Development file has too many rows")
        identifier, text = row.get("ID"), row.get("Tweet")
        if not identifier or identifier in seen or not text or None in row:
            raise ValueError(f"Invalid or duplicate development row {source_row}")
        seen.add(identifier)
        values = {label: row.get(label) for label in NATIVE_LABELS}
        if any(value not in ("0", "1") for value in values.values()):
            raise ValueError(f"Development labels must be binary at row {source_row}")
        native = [label for label in NATIVE_LABELS if values[label] == "1"]
        shared = [label for label in SHARED_LABELS if values[label] == "1"]
        records.append({"post_id": "semeval2018:english:dev:" + identifier,
            "source_post_id": identifier, "source_row": source_row, "text": text,
            "text_sha256": digest(text.encode("utf-8")), "split": "development",
            "reference_kind": "external_human", "native_labels": native,
            "native_binary_labels": {label: int(values[label]) for label in NATIVE_LABELS},
            "shared_labels": shared,
            "unrepresented_labels": [label for label in UNREPRESENTED_LABELS if values[label] == "1"],
            "native_all_negative": not native, "shared_all_negative": not shared})
    if not records:
        raise ValueError("Development reference contains no rows")
    return records


def prepare_development(archive_path, output_dir, *, expected_sha256=None, source_url=None):
    """Write one immutable reference version from known ZIP members only.

    Training/test members remain unopened. The original archive is preserved,
    including its held-out data, but only development records are canonicalized.
    """
    source, output = Path(archive_path), Path(output_dir)
    if output.exists():
        raise FileExistsError("Reference output already exists; use a new version directory")
    if source.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError("Archive exceeds the size limit")
    raw = source.read_bytes()
    source_hash = digest(raw)
    if expected_sha256 is not None and expected_sha256 != source_hash:
        raise ValueError("Archive fingerprint differs from the expected SHA-256")
    preserved_terms = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members = [info for info in archive.infolist() if info.filename in DEV_MEMBERS]
        if len(members) != 1:
            raise ValueError("Archive must contain exactly one known English development member")
        member = members[0]
        if member.file_size > MAX_MEMBER_BYTES:
            raise ValueError("Development member exceeds its size limit")
        development = archive.read(member)
        records = parse_development(development)
        for info in archive.infolist():
            if info.filename in TERMS_MEMBERS and info.file_size <= 100_000:
                preserved_terms[info.filename] = archive.read(info).decode("utf-8-sig", errors="replace")
    canonical = "".join(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n" for record in records).encode("utf-8")
    counts = Counter(label for record in records for label in record["native_labels"])
    version = app_release(workspace_root())["version"]
    manifest = {"schema": "semeval2018-ec-english-reference-v1", "dataset": "SemEval-2018 Task 1 E-c English",
        "split": "development", "rows": len(records), "created_at": datetime.now(timezone.utc).isoformat(),
        "app_version": version, "source_url": source_url or (DEVELOPMENT_URL if member.filename == DEV_MEMBERS[1] else OFFICIAL_URL), "paper_url": PAPER_URL,
        "source_sha256": source_hash, "development_member": member.filename,
        "development_member_sha256": digest(development), "canonical_sha256": digest(canonical),
        "native_labels": list(NATIVE_LABELS), "native_label_counts": {label: counts[label] for label in NATIVE_LABELS},
        "shared_mapping": {label: label for label in SHARED_LABELS},
        "unrepresented_labels": list(UNREPRESENTED_LABELS),
        "native_all_negative_rows": sum(record["native_all_negative"] for record in records),
        "unsupported_only_rows": sum(bool(record["native_labels"]) and not record["shared_labels"] for record in records),
        "test_labels_read": False, "training_labels_read": False, "model_scored": False,
        "terms": {"source_url": TERMS_URL, "checked_on": "2026-10-15", "license_identifier": None,
            "use": "Scientific or research purposes only", "redistribution": "Do not redistribute or share the dataset in part or full with third parties; refer them to the official website", "citation_required": True,
            "archive_notices": preserved_terms},
        "interpretation": ["The eight exact-name shared categories form a partial mapping, not the full GoEmotions vocabulary.",
            "An empty shared-label set does not establish neutral; inspect all eleven native labels.",
            "No current model accuracy is measured by this preparation step. Test labels remain held out."]}
    output.mkdir(parents=True, exist_ok=False)
    (output / "source.zip").write_bytes(raw)
    (output / "development.tsv").write_bytes(development)
    (output / "reference.jsonl").write_bytes(canonical)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def download_archive(path):
    """Download from the author; never substitute an undocumented mirror."""
    path = Path(path)
    if path.exists():
        raise FileExistsError("Archive already exists; it will not be overwritten")
    with urlopen(OFFICIAL_URL, timeout=60) as response:
        raw = response.read(MAX_ARCHIVE_BYTES + 1)
    if len(raw) > MAX_ARCHIVE_BYTES:
        raise ValueError("Archive exceeds the size limit")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        if not any(info.filename in DEV_MEMBERS for info in archive.infolist()):
            raise ValueError("Downloaded ZIP has no known English development member")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(raw)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    if args.download:
        download_archive(args.archive)
    result = prepare_development(args.archive, args.out, expected_sha256=args.expected_sha256)
    print(json.dumps({"rows": result["rows"], "manifest": str(args.out / "manifest.json"), "source_sha256": result["source_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
