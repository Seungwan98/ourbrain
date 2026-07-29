"""Import human-reviewed normal patches into a training manifest."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from ourbrain_cv.manifest import MANIFEST_FIELDS, read_manifest, write_manifest

NEGATIVE_LABELS = {"0", "negative", "no_crack", "normal"}
EXCLUDED_REVIEW_LABELS = {"1", "crack", "positive", "uncertain", "unsure"}
SPREADSHEET_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def restore_spreadsheet_safe_cell(value: str) -> str:
    """Undo the review UI's formula-injection prefix for internal import."""

    if (
        len(value) >= 2
        and value.startswith("'")
        and value[1:].startswith(SPREADSHEET_FORMULA_PREFIXES)
    ):
        return value[1:]
    return value


def deterministic_split(
    group_id: str,
    *,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> str:
    """Assign a new group without reshuffling existing manifest groups."""
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("split ratios must sum to a positive value")
    digest = hashlib.sha256(f"{seed}:{group_id}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    train_cutoff = train_ratio / total
    val_cutoff = train_cutoff + val_ratio / total
    if value < train_cutoff:
        return "train"
    if value < val_cutoff:
        return "val"
    return "test"


def file_sha256(path: str | Path) -> str:
    """Return a stable SHA-256 digest for a review or manifest artifact."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def review_audit_path(manifest_csv: str | Path) -> Path:
    """Return the provenance sidecar path for an imported manifest."""

    return Path(manifest_csv).with_suffix(".review.json")


def validate_review_audit(manifest_csv: str | Path) -> dict[str, Any]:
    """Validate complete-review provenance and manifest integrity."""

    manifest_path = Path(manifest_csv).expanduser().resolve()
    audit_path = review_audit_path(manifest_path)
    if not audit_path.is_file():
        raise RuntimeError(
            f"Complete-review provenance is missing: {audit_path}. "
            "Run import-negatives with a fully reviewed CSV."
        )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("schema_version") != 1 or audit.get("review_complete") is not True:
        raise RuntimeError(
            f"Review provenance is incomplete or unsupported: {audit_path}"
        )
    expected_digest = audit.get("output_manifest_sha256", "")
    actual_digest = file_sha256(manifest_path)
    if not expected_digest or actual_digest != expected_digest:
        raise RuntimeError(
            "Training manifest changed after reviewed-negative import; "
            "run import-negatives again."
        )
    return audit


def import_reviewed_negatives(
    review_csv: str | Path,
    manifest_csv: str | Path,
    output_csv: str | Path,
    *,
    seed: int = 42,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Append explicit negatives and record complete-review provenance.

    Every review row must have a recognized decision before a release manifest
    is written. ``crack`` and ``uncertain`` are valid review decisions but are
    deliberately excluded from negative training data.
    """

    review_path = Path(review_csv).expanduser().resolve()
    manifest_path = Path(manifest_csv).expanduser().resolve()
    output_path = Path(output_csv).expanduser().resolve()
    audit_path = review_audit_path(output_path)
    if len({review_path, manifest_path, output_path, audit_path}) != 4:
        raise ValueError("review, base manifest, output manifest, and audit paths must differ")

    existing = read_manifest(manifest_path)
    with review_path.open(newline="", encoding="utf-8") as handle:
        review_rows = list(csv.DictReader(handle))

    # A new import attempt supersedes any previous result at this output path.
    # Invalidate provenance first so a crash or validation failure cannot leave
    # a stale manifest that still passes the training gate.
    audit_path.unlink(missing_ok=True)
    output_path.unlink(missing_ok=True)

    existing_paths = {row["image_path"] for row in existing}
    split_by_group = {row["group_id"]: row["split"] for row in existing}
    added: list[dict[str, str]] = []
    skipped_unreviewed = 0
    skipped_excluded = 0
    invalid_labels = 0
    invalid_label_values: Counter[str] = Counter()
    skipped_missing = 0
    skipped_duplicate = 0
    negative_decisions = 0

    for review in review_rows:
        label = restore_spreadsheet_safe_cell(
            review.get("review_label", "")
        ).strip().lower()
        if not label:
            skipped_unreviewed += 1
            continue
        if label in EXCLUDED_REVIEW_LABELS:
            skipped_excluded += 1
            continue
        if label not in NEGATIVE_LABELS:
            invalid_labels += 1
            invalid_label_values[label] += 1
            continue
        negative_decisions += 1
        candidate = (
            Path(
                restore_spreadsheet_safe_cell(
                    review.get("candidate_path", "")
                )
            )
            .expanduser()
            .resolve()
        )
        if not candidate.is_file():
            skipped_missing += 1
            continue
        if str(candidate) in existing_paths:
            skipped_duplicate += 1
            continue
        group_id = (
            restore_spreadsheet_safe_cell(review.get("group_id", "")).strip()
            or candidate.stem.split("_", 1)[0]
        )
        split = split_by_group.get(group_id) or deterministic_split(group_id, seed=seed)
        with Image.open(candidate) as image:
            width, height = image.size
        added.append(
            {
                "image_path": str(candidate),
                "mask_path": "",
                "group_id": group_id,
                "split": split,
                "width": str(width),
                "height": str(height),
                "mask_width": "",
                "mask_height": "",
                "positive_pixels": "0",
                "source_kind": "reviewed_negative",
            }
        )
        existing_paths.add(str(candidate))

    review_complete = (
        skipped_unreviewed == 0 and invalid_labels == 0 and skipped_missing == 0
    )
    if require_complete and not review_complete:
        raise ValueError(
            "Review is incomplete; every candidate needs a recognized decision and "
            "every negative candidate file must exist. "
            f"unreviewed={skipped_unreviewed}, invalid_labels={invalid_labels}, "
            f"missing_files={skipped_missing}"
        )

    rows = existing + added
    write_manifest(rows, output_path)
    audit = {
        "schema_version": 1,
        "review_complete": review_complete,
        "review_csv": str(review_path),
        "review_csv_sha256": file_sha256(review_path),
        "base_manifest": str(manifest_path),
        "base_manifest_sha256": file_sha256(manifest_path),
        "output_manifest": str(output_path),
        "output_manifest_sha256": file_sha256(output_path),
        "review_rows": (
            negative_decisions + skipped_excluded + skipped_unreviewed + invalid_labels
        ),
        "decision_counts": {
            "negative": negative_decisions,
            "excluded_non_negative": skipped_excluded,
            "unreviewed": skipped_unreviewed,
            "invalid": invalid_labels,
        },
        "invalid_label_values": dict(invalid_label_values),
        "added_negatives": len(added),
        "missing_negative_files": skipped_missing,
        "duplicate_negatives": skipped_duplicate,
    }
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "output_manifest": str(output_path),
        "review_audit": str(audit_path),
        "review_complete": review_complete,
        "existing_rows": len(existing),
        "added_negatives": len(added),
        "total_rows": len(rows),
        "skipped_unreviewed": skipped_unreviewed,
        "skipped_excluded": skipped_excluded,
        "invalid_labels": invalid_labels,
        "skipped_missing": skipped_missing,
        "skipped_duplicate": skipped_duplicate,
    }


__all__ = [
    "MANIFEST_FIELDS",
    "EXCLUDED_REVIEW_LABELS",
    "SPREADSHEET_FORMULA_PREFIXES",
    "deterministic_split",
    "file_sha256",
    "import_reviewed_negatives",
    "review_audit_path",
    "restore_spreadsheet_safe_cell",
    "validate_review_audit",
]
