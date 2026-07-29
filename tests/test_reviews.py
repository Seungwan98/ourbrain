from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image

from ourbrain_cv.manifest import MANIFEST_FIELDS, read_manifest, write_manifest
from ourbrain_cv.reviews import deterministic_split, import_reviewed_negatives


def test_deterministic_split_is_stable() -> None:
    assert deterministic_split("0003", seed=42) == deterministic_split("0003", seed=42)


def test_import_reviewed_negatives_only_adds_explicit_normal_labels(tmp_path: Path) -> None:
    existing_image = tmp_path / "existing.png"
    Image.new("RGB", (8, 8), "gray").save(existing_image)
    manifest = tmp_path / "manifest.csv"
    write_manifest(
        [
            {
                "image_path": str(existing_image),
                "mask_path": "/mask.png",
                "group_id": "0001",
                "split": "val",
                "width": "8",
                "height": "8",
                "mask_width": "8",
                "mask_height": "8",
                "positive_pixels": "1",
                "source_kind": "paired",
            }
        ],
        manifest,
    )

    accepted = tmp_path / "accepted.png"
    ignored = tmp_path / "ignored.png"
    Image.new("RGB", (16, 12), "gray").save(accepted)
    Image.new("RGB", (16, 12), "gray").save(ignored)
    review = tmp_path / "review.csv"
    with review.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["candidate_path", "group_id", "review_label"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "candidate_path": str(accepted),
                "group_id": "0001",
                "review_label": "negative",
            }
        )
        writer.writerow(
            {
                "candidate_path": str(ignored),
                "group_id": "0002",
                "review_label": "",
            }
        )

    output = tmp_path / "with_negatives.csv"
    summary = import_reviewed_negatives(review, manifest, output)
    rows = read_manifest(output)

    assert summary["added_negatives"] == 1
    assert len(rows) == 2
    negative = rows[-1]
    assert list(negative) == MANIFEST_FIELDS
    assert negative["mask_path"] == ""
    assert negative["positive_pixels"] == "0"
    assert negative["source_kind"] == "reviewed_negative"
    assert negative["split"] == "val"

