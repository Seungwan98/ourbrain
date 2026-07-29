"""Import human-reviewed normal patches into a training manifest."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from PIL import Image

from ourbrain_cv.manifest import MANIFEST_FIELDS, read_manifest, write_manifest

NEGATIVE_LABELS = {"0", "negative", "no_crack", "normal"}


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


def import_reviewed_negatives(
    review_csv: str | Path,
    manifest_csv: str | Path,
    output_csv: str | Path,
    *,
    seed: int = 42,
) -> dict[str, int | str]:
    """Append only explicitly reviewed negative patches to a new manifest."""
    existing = read_manifest(manifest_csv)
    existing_paths = {row["image_path"] for row in existing}
    split_by_group = {row["group_id"]: row["split"] for row in existing}
    added: list[dict[str, str]] = []
    skipped_unreviewed = 0
    skipped_missing = 0
    skipped_duplicate = 0

    with Path(review_csv).open(newline="", encoding="utf-8") as handle:
        for review in csv.DictReader(handle):
            label = review.get("review_label", "").strip().lower()
            if label not in NEGATIVE_LABELS:
                skipped_unreviewed += 1
                continue
            candidate = Path(review.get("candidate_path", "")).expanduser().resolve()
            if not candidate.is_file():
                skipped_missing += 1
                continue
            if str(candidate) in existing_paths:
                skipped_duplicate += 1
                continue
            group_id = review.get("group_id", "").strip() or candidate.stem.split("_", 1)[0]
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

    rows = existing + added
    write_manifest(rows, output_csv)
    return {
        "output_manifest": str(Path(output_csv).expanduser().resolve()),
        "existing_rows": len(existing),
        "added_negatives": len(added),
        "total_rows": len(rows),
        "skipped_unreviewed": skipped_unreviewed,
        "skipped_missing": skipped_missing,
        "skipped_duplicate": skipped_duplicate,
    }


__all__ = ["MANIFEST_FIELDS", "deterministic_split", "import_reviewed_negatives"]

