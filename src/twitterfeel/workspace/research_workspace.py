"""Immutable user dataset versions and clearly separated research observations."""
from collections import Counter
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import sqlite3
import unicodedata
import uuid

import pandas as pd
from twitterfeel.common.config import app_release

MAX_BYTES = 20 * 1024 * 1024
MAX_ROWS = 100_000


def _now():
    return datetime.now(timezone.utc).isoformat()


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _version(root):
    value = app_release(root)
    return value.get("version") if isinstance(value, dict) else value


@contextmanager
def _database(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30)
    connection.row_factory = sqlite3.Row
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def _write(path, content):
    # Exclusive creation protects prior artifacts even if an ID is accidentally reused.
    with path.open("xb") as output:
        output.write(content)
        output.flush()
        import os
        os.fsync(output.fileno())


def _name(value):
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > 120:
        raise ValueError("Dataset name must contain 1 to 120 characters")
    display = unicodedata.normalize("NFKC", value).strip()
    logical = " ".join(display.casefold().split())
    return display, logical


def _parse(content, format):
    if isinstance(content, str):
        raw = content.encode("utf-8")
    elif isinstance(content, bytes):
        raw = content
    else:
        raise ValueError("Dataset content must be UTF-8 text or bytes")
    if len(raw) > MAX_BYTES:
        raise ValueError("Dataset exceeds the 20 MB limit")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Use a UTF-8 CSV, TSV, TXT, or JSONL file") from exc
    if "\x00" in text:
        raise ValueError("Binary/NUL content is not supported")
    format = str(format).lower().lstrip(".")
    if format not in {"csv", "tsv", "txt", "jsonl"}:
        raise ValueError("Supported formats are csv, tsv, txt, and jsonl")
    if not text.strip():
        raise ValueError("Dataset is empty")
    try:
        if format in {"csv", "tsv"}:
            delimiter = "," if format == "csv" else "\t"
            reader = csv.reader(io.StringIO(text), delimiter=delimiter, strict=True)
            header = next(reader)
            if not header or any(not col.strip() for col in header) or len(header) != len(set(header)):
                raise ValueError("Column names must be nonblank and unique")
            rows = []
            for number, row in enumerate(reader, 2):
                if not row:
                    row = [""] * len(header)
                if len(row) != len(header):
                    raise ValueError(f"Row {number} has {len(row)} fields; expected {len(header)}")
                rows.append(row)
                if len(rows) > MAX_ROWS:
                    break
            frame = pd.DataFrame(rows, columns=header)
        elif format == "txt":
            frame = pd.DataFrame({"text": text.splitlines()})
        else:
            rows = []
            for number, line in enumerate(text.splitlines(), 1):
                if not line.strip():
                    continue
                item = json.loads(line)
                if not isinstance(item, dict) or not item:
                    raise ValueError(f"JSONL line {number} must be a nonempty object")
                if any(not isinstance(k, str) or not k.strip() for k in item):
                    raise ValueError("JSONL column names must be nonblank strings")
                rows.append({k: "" if v is None else v if isinstance(v, str)
                             else json.dumps(v, ensure_ascii=False, separators=(",", ":")) for k, v in item.items()})
                if len(rows) > MAX_ROWS:
                    break
            frame = pd.DataFrame(rows).fillna("")
    except (pd.errors.ParserError, pd.errors.EmptyDataError, json.JSONDecodeError, csv.Error) as exc:
        raise ValueError("Could not parse dataset: " + str(exc)) from exc
    if len(frame) > MAX_ROWS:
        raise ValueError("Dataset exceeds the 100,000 row limit")
    if frame.empty:
        raise ValueError("Dataset has no data rows")
    return frame, raw, format


class DatasetLibrary:
    def __init__(self, root):
        self._root = Path(root).resolve()
        self._folder = self._root / "runs/workbench/datasets"
        self._folder.mkdir(parents=True, exist_ok=True)
        self._db = self._root / "runs/workbench/dataset_versions.sqlite"
        with _database(self._db) as db:
            db.execute("CREATE TABLE IF NOT EXISTS versions (logical_name TEXT NOT NULL, version INTEGER NOT NULL, dataset_id TEXT UNIQUE NOT NULL, PRIMARY KEY(logical_name,version))")

    def preview(self, name, content, format="csv"):
        name, _ = _name(name)
        frame, raw, format = _parse(content, format)
        return {"name": name, "columns": frame.columns.tolist(), "rows": len(frame),
                "format": format, "source_bytes": len(raw), "content_sha256": _hash(raw),
                "preview": frame.head(10).to_dict("records"), "requires_text_mapping": format != "txt"}

    def _choices(self):
        choices = {}
        for path in self._folder.glob("*/manifest.json"):
            if not path.resolve().is_relative_to(self._folder.resolve()):
                continue
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                identifier = path.parent.relative_to(self._root).as_posix()
                if value.get("id") == identifier and value.get("schema_version") == 1:
                    choices[identifier] = (path.parent, value)
            except (OSError, ValueError, AttributeError):
                continue
        return choices

    def manifest(self, id):
        choices = self._choices()
        if id not in choices:
            raise ValueError("Unknown imported dataset ID")
        return dict(choices[id][1])

    def list_datasets(self):
        return sorted((dict(value) for _, value in self._choices().values()),
                      key=lambda value: (value["created_at"], value["id"]), reverse=True)

    def load_dataset(self, id):
        choices = self._choices()
        if id not in choices:
            raise ValueError("Unknown imported dataset ID")
        folder, manifest = choices[id]
        canonical, source = folder / "dataset.csv", folder / ("source." + manifest["format"])
        if not canonical.resolve().is_relative_to(folder.resolve()) or not source.resolve().is_relative_to(folder.resolve()):
            raise ValueError("Dataset artifact escaped its version folder")
        if _hash(canonical.read_bytes()) != manifest["dataset_sha256"] or _hash(source.read_bytes()) != manifest["content_sha256"]:
            raise ValueError("Dataset artifacts changed since import")
        return pd.read_csv(canonical, dtype={"post_id": str, "text": str, "source_post_id": str, "user_id": str}, keep_default_na=False)

    def import_dataset(self, name, content, format="csv", text_column=None, user_column=None):
        display, logical = _name(name)
        frame, raw, format = _parse(content, format)
        if format == "txt" and text_column is None:
            text_column = "text"
        if not isinstance(text_column, str) or text_column not in frame:
            raise ValueError("Choose the column containing post text before importing")
        if user_column is not None and user_column not in frame:
            raise ValueError("Choose a valid user ID column or leave it unmapped")
        canonical = pd.DataFrame({"text": frame[text_column].astype(str), "source_row": range(len(frame))})
        if "post_id" in frame:
            identifiers = frame.post_id.astype(str)
            canonical["source_post_id"] = identifiers
            if identifiers.duplicated().any() or identifiers.str.strip().eq("").any():
                canonical["post_id"] = [f"{value or 'post'}:row_{i}" for i, value in enumerate(identifiers)]
            else:
                canonical["post_id"] = identifiers
        else:
            canonical["post_id"] = [f"post_{i:08d}" for i in range(len(frame))]
        if user_column is not None:
            canonical["user_id"] = frame[user_column].astype(str)
        ordered = ["post_id", "source_row", "text"] + [c for c in ("source_post_id", "user_id") if c in canonical]
        canonical = canonical[ordered]
        encoded = canonical.to_csv(index=False).encode("utf-8")
        slug = re.sub(r"[^a-z0-9]+", "-", logical).strip("-")[:45] or "dataset"
        with _database(self._db) as db:
            db.execute("BEGIN IMMEDIATE")
            registered = db.execute("SELECT COALESCE(MAX(version),0) FROM versions WHERE logical_name=?", (logical,)).fetchone()[0]
            existing = max((m["version"] for _, m in self._choices().values() if m.get("logical_name") == logical), default=0)
            version = max(registered, existing) + 1
            folder = self._folder / f"{slug}-v{version:03d}-{uuid.uuid4().hex[:12]}"
            folder.mkdir()
            identifier = folder.relative_to(self._root).as_posix()
            mapping = {"text_column": text_column, "user_column": user_column,
                       "source_id_column": "post_id" if "post_id" in frame else None}
            manifest = {"schema_version": 1, "schema": 1, "id": identifier, "name": display, "logical_name": logical,
                        "version": version, "created_at": _now(), "rows": len(canonical),
                        "columns": ordered, "text_column": text_column, "user_column": user_column,
                        "mapping": mapping, "format": format, "source_bytes": len(raw),
                        "content_sha256": _hash(raw), "source_sha256": _hash(raw),
                        "dataset_sha256": _hash(encoded), "app_version": _version(self._root),
                        "blank_texts": int(canonical.text.str.strip().eq("").sum()),
                        "repeated_texts": int(canonical.text.duplicated().sum()),
                        "preview": canonical.head(10).to_dict("records"),
                        "note": "Import only. No model analysis was run; prior machine-label columns were excluded."}
            _write(folder / ("source." + format), raw)
            _write(folder / "dataset.csv", encoded)
            _write(folder / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"))
            db.execute("INSERT INTO versions VALUES (?,?,?)", (logical, version, identifier))
        return manifest


class FindingsStore:
    def __init__(self, root):
        self._root = Path(root).resolve()
        self._db = self._root / "runs/workbench/findings.sqlite"
        with _database(self._db) as db:
            db.execute("""CREATE TABLE IF NOT EXISTS observations (
                id TEXT PRIMARY KEY, run_id TEXT NOT NULL, author TEXT NOT NULL,
                title TEXT NOT NULL, note TEXT NOT NULL, created_at TEXT NOT NULL, app_version TEXT)""")

    def _run(self, identifier):
        if not isinstance(identifier, str) or Path(identifier).is_absolute():
            raise ValueError("Choose an existing analysis run")
        folder = (self._root / identifier).resolve()
        if not folder.is_relative_to(self._root / "runs") or folder.relative_to(self._root).as_posix() != identifier:
            raise ValueError("Unknown analysis run ID")
        if not all((folder / name).is_file() for name in ("config.json", "summary.json")):
            raise ValueError("Analysis run artifacts were not found")
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        if config.get("status") != "complete" or config.get("task") not in ("emotion-and-sentiment", "emotion"):
            raise ValueError("Choose a completed existing-post analysis run")
        if not any((folder / name).is_file() for name in ("predictions.jsonl", "predictions.csv")):
            raise ValueError("Analysis predictions were not found")
        return folder

    def save(self, run_id, author, title, note):
        self._run(run_id)
        for name, value, limit in (("Author", author, 80), ("Title", title, 160), ("Observation", note, 20000)):
            if not isinstance(value, str) or not 0 < len(value.strip()) <= limit:
                raise ValueError(f"{name} must contain 1 to {limit:,} characters")
        record = {"id": uuid.uuid4().hex, "run_id": run_id, "author": author.strip(),
                  "title": title.strip(), "note": note.strip(), "created_at": _now(),
                  "app_version": _version(self._root), "kind": "authored"}
        with _database(self._db) as db:
            db.execute("INSERT INTO observations VALUES (?,?,?,?,?,?,?)",
                       tuple(record[k] for k in ("id", "run_id", "author", "title", "note", "created_at", "app_version")))
        return record

    def list(self, run_id=None):
        if run_id is not None:
            self._run(run_id)
        with _database(self._db) as db:
            rows = db.execute("SELECT * FROM observations" + (" WHERE run_id=?" if run_id is not None else "") +
                              " ORDER BY created_at DESC,id", (run_id,) if run_id is not None else ()).fetchall()
        return [dict(row, kind="authored") for row in rows]


def build_findings(run_folder):
    """Deterministic description of model output; never a quality/causal claim."""
    folder = Path(run_folder)
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    path = folder / "predictions.jsonl"
    if path.exists():
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        records = pd.read_csv(folder / "predictions.csv", keep_default_na=False, dtype=str).to_dict("records")
        for row in records:
            for field in ("labels", "sentiment"):
                if field in row:
                    row[field] = json.loads(row[field]) if row[field] else ([] if field == "labels" else {})
    total = len(records)
    labels = Counter(label for row in records for label in set(row.get("labels", [])) if label != "neutral")
    statuses = Counter(row.get("status", "unknown") for row in records)
    sentiments = Counter(row.get("sentiment", {}).get("label") for row in records if isinstance(row.get("sentiment"), dict) and row["sentiment"].get("label"))
    multi = sum(len(set(row.get("labels", [])) - {"neutral"}) > 1 for row in records)
    flagged = sum(statuses[key] for key in ("uncertain", "review_needed", "invalid"))
    observations = [{"title": "Scope of this analysis", "text": f"This run contains {total:,} posts. These are automatic model outputs, not human-verified labels.",
                     "evidence": {"rows": total, "dataset_id": config.get("dataset_id"), "task": config.get("task")}}]
    if labels:
        ranked = sorted(labels.items(), key=lambda pair: (-pair[1], pair[0]))[:3]
        detail = "; ".join(f"{label}: {count:,} posts ({count/total:.1%})" for label, count in ranked)
        observations.append({"title": "Most frequently selected emotions", "text": "The model most often selected " + detail + ". Emotion percentages can overlap because one post may receive several labels.",
                             "evidence": {"denominator": total, "top_labels": dict(ranked)}})
    else:
        observations.append({"title": "No emotion labels selected", "text": "No non-neutral emotion labels were selected at the saved settings. This does not establish that the posts contain no emotion.", "evidence": {"non_neutral_assignments": 0}})
    observations.append({"title": "Multiple emotions", "text": f"{multi:,} posts received at least two non-neutral emotion labels. This is selected-label overlap, not a measure of emotional intensity.", "evidence": {"posts": multi, "denominator": total}})
    if sentiments:
        detail = "; ".join(f"{label}: {count:,}" for label, count in sorted(sentiments.items()))
        observations.append({"title": "Sentiment polarity", "text": f"Independent sentiment outputs were available for {sum(sentiments.values()):,} of {total:,} posts: {detail}. Sentiment polarity is scored separately from the emotion labels.", "evidence": {"sentiment_counts": dict(sentiments), "denominator": sum(sentiments.values())}})
    else:
        observations.append({"title": "Sentiment was not available", "text": "This run contains no sentiment outputs. Positive or negative sentiment has not been inferred from its emotion labels.", "evidence": {"sentiment_rows": 0}})
    observations.append({"title": "Posts needing review", "text": f"{flagged:,} posts were uncertain, flagged for review, or invalid. Inspect these examples before drawing conclusions.", "evidence": {"status_counts": dict(statuses), "flagged_rows": flagged}})
    limitations = ["Counts describe this dataset and the saved model settings; they do not measure accuracy.",
                   "Sarcasm, quotation, negation, mixed emotions, and missing context can lead to misleading labels.",
                   "Model support scores do not measure emotional intensity or establish the author's mental state.",
                   "Sentiment polarity and emotion labels are separate predictions; negative sentiment is not a diagnosis or an emotion category.",
                   "Human-reviewed reference labels are needed to compare labeling quality. Patterns here do not establish causes or population-wide findings."]
    if summary.get("rows", total) != total:
        limitations.append("The saved summary row count differs from the prediction artifact. Findings use the prediction rows; inspect provenance before relying on this run.")
    return {"kind": "automatic", "overview": f"Automatic observations for {total:,} posts; interpret them as model behavior, not verified facts about people.",
            "observations": observations, "limitations": limitations,
            "provenance": {"config_sha256": _hash((folder / "config.json").read_bytes()),
                           "predictions_sha256": _hash((path if path.exists() else folder / "predictions.csv").read_bytes()),
                           "app_version": config.get("app_version"), "dataset_id": config.get("dataset_id")}}
