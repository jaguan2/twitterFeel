"""Blind development calibration and provenance-checked legacy user recovery."""
import hashlib
import json
from pathlib import Path
import time

import pandas as pd

from twitterfeel.common.provenance import file_digest
from twitterfeel.common.resources import annotation_rubric
from twitterfeel.labeling.label_posts import read_posts
from twitterfeel.labeling.label_schema import text_hash


def _review_path(root, path):
    root = Path(root).resolve()
    path = Path(path)
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(root / "runs") or path.name != "review.csv" or not path.is_file():
        raise ValueError("Choose an existing review sample inside this project")
    return root, path


def _diagnostic(frame, status, warnings, **details):
    frame.attrs["user_enrichment"] = {"status": status, "warnings": warnings, **details}
    return frame


def enrich_users(frame, review_path, root):
    """Return a copy; warnings/status are in ``attrs['user_enrichment']``.

    Every row must match a hash-verified source export before any missing user
    identity is filled. Missing identities are never invented from post IDs.
    """
    root, review_path = _review_path(root, review_path)
    result = frame.copy(deep=True)
    missing = (result.user_id.isna() | result.user_id.astype(str).str.strip().eq("")) if "user_id" in result else pd.Series(True, index=result.index)
    if not missing.any():
        return _diagnostic(result, "already_present", [], recovered_rows=0, missing_rows=0)
    fallback = "User identities could not be recovered from verified provenance. Keep missing users explicit and report any row-based fallback."
    metadata_path = review_path.parent / "sample.json"
    if not metadata_path.is_file():
        return _diagnostic(result, "unavailable", [fallback, "Review sample has no source fingerprint metadata."], recovered_rows=0, missing_rows=int(missing.sum()))
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = metadata.get("source_sha256")
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("No valid source SHA-256 was recorded")
        required = {"source_row", "post_id", "text_sha256", "text"}
        if not required.issubset(result):
            raise ValueError("Review rows do not contain complete alignment identity")
        if any(text_hash(str(row.text)) != str(row.text_sha256) for row in result.itertuples()):
            raise ValueError("Review text differs from its recorded identity")
        candidates = list((root / "data/raw").glob("*.xls"))
        candidates += list((root / "data/interim").glob("*.csv"))
        candidates += list((root / "runs/workbench/datasets").glob("*/dataset.csv"))
        sources = [p for p in sorted(candidates) if p.is_file() and p.resolve().is_relative_to(root) and file_digest(p) == expected]
        if not sources:
            raise ValueError("No allowlisted source matches the recorded file fingerprint")
        for source_path in sources:
            source = read_posts(source_path, metadata.get("text_column"))
            # Preserve exact numeric-looking text too; pandas must not normalize it.
            column = source.attrs["text_column"]
            exact = pd.read_csv(source_path, dtype=str, keep_default_na=False, usecols=[column])[column]
            source["text"] = exact
            source["text_sha256"] = exact.map(text_hash)
            if "user_id" not in source:
                continue
            by_identity = {}
            for row in source.to_dict("records"):
                key = (str(row["source_row"]), str(row["post_id"]), row["text_sha256"])
                if key in by_identity:
                    raise ValueError("Source alignment identity is not unique")
                by_identity[key] = str(row["user_id"])
            matched = []
            for row in result.to_dict("records"):
                key = (str(row["source_row"]), str(row["post_id"]), row["text_sha256"])
                if key not in by_identity:
                    break
                matched.append(by_identity[key])
            if len(matched) != len(result):
                continue
            if "user_id" not in result:
                result["user_id"] = ""
            existing = result.user_id.fillna("").astype(str)
            if any(not is_missing and value != recovered for is_missing, value, recovered in zip(missing, existing, matched)):
                raise ValueError("Existing user identities conflict with verified source identities")
            for index, recovered, is_missing in zip(result.index, matched, missing):
                if is_missing and recovered.strip():
                    result.at[index, "user_id"] = recovered
            remaining = result.user_id.isna() | result.user_id.astype(str).str.strip().eq("")
            recovered = int(missing.sum() - remaining.sum())
            warnings = ["Some verified source user IDs are blank; report the missing-user fallback."] if remaining.any() else []
            return _diagnostic(result, "partial" if remaining.any() else "verified", warnings,
                               recovered_rows=recovered, missing_rows=int(remaining.sum()),
                               source=source_path.relative_to(root).as_posix(), source_sha256=expected)
        raise ValueError("No matching source contains user IDs with exact row/post/text alignment")
    except (ValueError, OSError, KeyError, TypeError) as exc:
        return _diagnostic(result, "unavailable", [fallback, str(exc)], recovered_rows=0, missing_rows=int(missing.sum()))


def prepare_calibration(root, review_path):
    """Materialize the same blinded practice set for the same source and rubric."""
    root, source_path = _review_path(root, review_path)
    source_id = source_path.relative_to(root).as_posix()
    existing_metadata = source_path.parent / "metadata.json"
    if existing_metadata.is_file():
        existing = json.loads(existing_metadata.read_text(encoding="utf-8"))
        if existing.get("purpose") == "calibration":
            if file_digest(source_path) != existing.get("review_sha256"):
                raise ValueError("Calibration CSV changed since preparation; preserve the original practice artifact")
            return {"review_id": source_id, "rows": len(read_posts(source_path)), "existing": True,
                    "source_review_id": existing["source_review_id"], "warning": existing.get("warning", "")}
    rubric, rubric_hash = annotation_rubric(root)
    source_hash = file_digest(source_path)
    folder = root / "runs/workbench/calibration" / (source_hash + "_" + rubric_hash)
    output = folder / "review.csv"
    metadata_path = folder / "metadata.json"

    def reused():
        until = time.monotonic() + 3
        while not metadata_path.exists() and time.monotonic() < until:
            time.sleep(.05)
        if not metadata_path.is_file() or not output.is_file():
            raise ValueError("Calibration preparation is incomplete; retry after it finishes")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("source_sha256") != source_hash or metadata.get("rubric_sha256") != rubric_hash or file_digest(output) != metadata.get("review_sha256"):
            raise ValueError("Calibration artifacts changed since creation; preserve them and prepare a fresh source version")
        return {"review_id": output.relative_to(root).as_posix(), "rows": metadata["rows"],
                "existing": True, "source_review_id": source_id, "warning": metadata.get("warning", "")}

    if folder.exists():
        return reused()
    frame = read_posts(source_path)
    if "split" not in frame:
        raise ValueError("Review sample needs development/test split assignments")
    frame = enrich_users(frame, source_path, root)
    eligible = frame[(frame.split == "development") & frame.text.str.strip().ne("")].copy()
    if eligible.empty:
        raise ValueError("No unique nonblank development posts are available for calibration")
    eligible["_selection_key"] = eligible.apply(lambda row: hashlib.sha256(
        ("calibration-v1\0" + row.post_id + "\0" + row.text_sha256).encode("utf-8")).hexdigest(), axis=1)
    eligible = eligible.sort_values("_selection_key").drop_duplicates("text_sha256")
    if eligible.empty:
        raise ValueError("No unique nonblank development posts are available for calibration")
    selected = eligible.head(30)
    columns = [name for name in ("post_id", "source_post_id", "source_row", "text_sha256", "text", "user_id", "split") if name in selected]
    review = selected[columns].copy()
    review["human_labels"] = ""
    review["review_status"] = "pending"
    review["notes"] = ""
    warning = f"Only {len(review)} eligible posts are available; calibration target is 30." if len(review) < 30 else ""
    try:
        folder.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        return reused()
    review.to_csv(output, index=False, mode="x")
    metadata = {"schema_version": 1, "purpose": "calibration", "rows": len(review),
                "source_review_id": source_id, "source_sha256": source_hash,
                "rubric_version": rubric.get("version", rubric.get("rubric_version")),
                "rubric_sha256": rubric_hash, "review_sha256": file_digest(output),
                "selected_ids": review.post_id.tolist(),
                "selected_identity": review[["post_id", "text_sha256"]].to_dict("records"),
                "test_rows_selected": 0, "excluded_test_rows": int(frame.split.eq("test").sum()),
                "excluded_blank_development_rows": int(((frame.split == "development") & frame.text.str.strip().eq("")).sum()),
                "eligible_unique_development_rows": len(eligible), "warning": warning,
                "user_enrichment": frame.attrs.get("user_enrichment", {}),
                "note": "Practice on development text only. No AI or source annotations are reference labels. Test text was excluded."}
    with metadata_path.open("x", encoding="utf-8") as destination:
        json.dump(metadata, destination, indent=2)
    return {"review_id": output.relative_to(root).as_posix(), "rows": len(review),
            "existing": False, "source_review_id": source_id, "warning": warning}
