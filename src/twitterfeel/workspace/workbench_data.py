"""Local research workbench data and append-only human review persistence."""
from __future__ import annotations

import copy
from contextlib import contextmanager
import json
from twitterfeel.common.config import workspace_root
from pathlib import Path
import sqlite3

import pandas as pd

from twitterfeel.labeling.label_posts import read_posts
from twitterfeel.labeling.label_schema import LABELS, SCHEMA_VERSION, interpret, text_hash

REPO_ROOT = workspace_root()
PREDICTION_STATUSES = {"labeled", "neutral", "uncertain", "review_needed", "invalid"}


def discover_runs(root=REPO_ROOT):
    """Find completed labeling runs only; skip caches and damaged artifacts."""
    root = Path(root)
    runs = root / "runs"
    found = []
    if not runs.exists():
        return found
    for path in sorted(runs.rglob("config.json")):
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
            if config.get("schema") != SCHEMA_VERSION or config.get("status") != "complete":
                continue
            predictions = path.parent / "predictions.jsonl"
            summary_path = path.parent / "summary.json"
            if not predictions.is_file() or not summary_path.is_file():
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if not isinstance(summary, dict):
                continue
            found.append({"run_id": path.parent.relative_to(root).as_posix(),
                          "path": path.parent, "config": config,
                          "predictions_path": predictions, "summary": summary})
        except (OSError, ValueError, AttributeError):
            continue
    return found


def load_predictions(path):
    """Load persisted scores, rejecting identity or text integrity problems."""
    records = []
    seen = set()
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                identifier = row["post_id"]
                if not isinstance(identifier, str) or not identifier.strip():
                    raise ValueError("post_id must be a nonempty string")
                if identifier in seen:
                    raise ValueError("Duplicate prediction post_id")
                if row["status"] not in PREDICTION_STATUSES:
                    raise ValueError("Unknown prediction status")
                if not isinstance(row["text"], str) or text_hash(row["text"]) != row["text_sha256"]:
                    raise ValueError("Prediction text differs from its recorded hash")
                if not isinstance(row["labels"], list) or not set(row["labels"]).issubset(LABELS):
                    raise ValueError("Unknown prediction labels")
                if row["status"] != "invalid":
                    interpret(row["scores"])
                elif row["labels"] or row["scores"]:
                    raise ValueError("Invalid input must not carry emotion predictions")
                seen.add(identifier)
                records.append(row)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid prediction record on line {number}: {exc}") from exc
    if not records:
        raise ValueError("No prediction records found")
    return pd.DataFrame(records)


def threshold_frame(df, threshold):
    """Explore an operating threshold without modifying persisted predictions."""
    if not 0 < threshold < 1:
        raise ValueError("threshold must be between zero and one")
    records = copy.deepcopy(df.to_dict("records"))
    for row in records:
        if row["status"] == "invalid":
            continue
        row.update(interpret(row["scores"], threshold))
        if row.get("truncated", False):
            row["status"] = "review_needed"
            row["review_reason"] = ";".join(filter(None, [row["review_reason"], "text_truncated"]))
    return pd.DataFrame(records, index=df.index)


def load_review(path):
    return read_posts(Path(path))


class ReviewerStore:
    """Each save appends a revision. Review identity includes source text hash."""

    def __init__(self, db_path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS review_revisions (
                revision INTEGER PRIMARY KEY AUTOINCREMENT,
                review_path TEXT NOT NULL, post_id TEXT NOT NULL,
                text_sha256 TEXT NOT NULL, reviewer TEXT NOT NULL,
                labels_json TEXT NOT NULL, status TEXT NOT NULL, notes TEXT NOT NULL,
                saved_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
            )""")
            connection.execute("""CREATE INDEX IF NOT EXISTS review_lookup
                ON review_revisions(review_path, reviewer, post_id, revision)""")

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.db_path, timeout=15)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def _source(review_path):
        path = Path(review_path).resolve()
        source = load_review(path)
        return str(path), source

    def save(self, review_path, post_id, text_sha256, reviewer, labels, status, notes):
        if not isinstance(reviewer, str) or not reviewer.strip():
            raise ValueError("A reviewer name is required")
        if status not in {"reviewed", "unclear", "invalid"}:
            raise ValueError("Review status must be reviewed, unclear, or invalid")
        if not isinstance(labels, list) or not set(labels).issubset(LABELS):
            raise ValueError("Choose labels from the emotion vocabulary")
        if "neutral" in labels and len(set(labels)) > 1:
            raise ValueError("Neutral cannot be combined with emotion labels")
        if status == "reviewed" and not labels:
            raise ValueError("Reviewed posts need at least one label")
        if status != "reviewed" and labels:
            raise ValueError("Unclear or invalid reviews must not contain emotion labels")
        path, source = self._source(review_path)
        matching = source[source.post_id == post_id]
        if len(matching) != 1 or matching.iloc[0].text_sha256 != text_sha256:
            raise ValueError("Unknown post or stale text hash; reload the source")
        ordered = [label for label in LABELS if label in labels]
        with self._connect() as connection:
            cursor = connection.execute("""INSERT INTO review_revisions
                (review_path, post_id, text_sha256, reviewer, labels_json, status, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (path, post_id, text_sha256, reviewer.strip(), json.dumps(ordered), status, str(notes)))
            return cursor.lastrowid

    def latest(self, review_path, reviewer):
        path, source = self._source(review_path)
        hashes = dict(zip(source.post_id, source.text_sha256))
        with self._connect() as connection:
            rows = connection.execute("""SELECT * FROM review_revisions
                WHERE review_path = ? AND reviewer = ? ORDER BY revision""",
                (path, reviewer.strip())).fetchall()
        # Use the most recent revision only; changed text never inherits old labels.
        newest = {row["post_id"]: dict(row) for row in rows}
        result = {}
        for identifier, row in newest.items():
            if hashes.get(identifier) == row["text_sha256"]:
                row["labels"] = json.loads(row.pop("labels_json"))
                result[identifier] = row
        return result

    def merged_review(self, review_path, reviewer):
        source = load_review(review_path).copy()
        latest = self.latest(review_path, reviewer)
        # Separate reviewers start blank; source annotations cannot leak between them.
        source["human_labels"] = ""
        source["review_status"] = "pending"
        source["notes"] = ""
        for index, row in source.iterrows():
            saved = latest.get(row.post_id)
            if saved and saved["text_sha256"] == row.text_sha256:
                source.at[index, "human_labels"] = ";".join(saved["labels"])
                source.at[index, "review_status"] = saved["status"]
                source.at[index, "notes"] = saved["notes"]
        return source
