from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from ourbrain_cv.manifest import MANIFEST_FIELDS, read_manifest, write_manifest
from ourbrain_cv.reviews import (
    deterministic_split,
    import_reviewed_negatives,
    restore_spreadsheet_safe_cell,
    review_audit_path,
    validate_review_audit,
)


def test_deterministic_split_is_stable() -> None:
    assert deterministic_split("0003", seed=42) == deterministic_split("0003", seed=42)


def test_restore_spreadsheet_safe_cell_only_unescapes_formula_prefixes() -> None:
    assert restore_spreadsheet_safe_cell("'=HYPERLINK(...)") == "=HYPERLINK(...)"
    assert restore_spreadsheet_safe_cell("'+SUM(1,2)") == "+SUM(1,2)"
    assert restore_spreadsheet_safe_cell("'normal") == "'normal"
    assert restore_spreadsheet_safe_cell("negative") == "negative"


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
                "review_label": "crack",
            }
        )

    output = tmp_path / "with_negatives.csv"
    summary = import_reviewed_negatives(review, manifest, output)
    rows = read_manifest(output)

    assert summary["added_negatives"] == 1
    assert summary["review_complete"] is True
    assert summary["skipped_excluded"] == 1
    assert len(rows) == 2
    negative = rows[-1]
    assert list(negative) == MANIFEST_FIELDS
    assert negative["mask_path"] == ""
    assert negative["positive_pixels"] == "0"
    assert negative["source_kind"] == "reviewed_negative"
    assert negative["split"] == "val"
    audit = json.loads(review_audit_path(output).read_text(encoding="utf-8"))
    assert audit["review_complete"] is True
    assert audit["decision_counts"] == {
        "negative": 1,
        "excluded_non_negative": 1,
        "unreviewed": 0,
        "invalid": 0,
    }
    assert validate_review_audit(output)["output_manifest"] == str(output.resolve())


def test_import_reviewed_negatives_rejects_partial_review(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.png"
    Image.new("RGB", (8, 8), "gray").save(candidate)
    manifest = tmp_path / "manifest.csv"
    write_manifest([], manifest)
    review = tmp_path / "review.csv"

    def write_decision(label: str) -> None:
        with review.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["candidate_path", "group_id", "review_label"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "candidate_path": str(candidate),
                    "group_id": "001",
                    "review_label": label,
                }
            )

    output = tmp_path / "with_negatives.csv"
    write_decision("negative")
    import_reviewed_negatives(review, manifest, output)
    assert output.exists()
    assert validate_review_audit(output)["review_complete"] is True

    write_decision("")
    with pytest.raises(ValueError, match="Review is incomplete"):
        import_reviewed_negatives(review, manifest, output)

    assert not output.exists()
    assert not review_audit_path(output).exists()
    with pytest.raises(RuntimeError, match="provenance is missing"):
        validate_review_audit(output)


def test_import_rejects_colliding_output_paths(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    write_manifest([], manifest)
    review = tmp_path / "review.csv"
    with review.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["candidate_path", "group_id", "review_label"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "candidate_path": "/candidate.png",
                "group_id": "001",
                "review_label": "negative",
            }
        )

    with pytest.raises(ValueError, match="paths must differ"):
        import_reviewed_negatives(review, manifest, manifest)
    with pytest.raises(ValueError, match="paths must differ"):
        import_reviewed_negatives(manifest, manifest, tmp_path / "output.csv")


def test_import_reviewed_negatives_restores_spreadsheet_safe_candidate_path(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    candidate = Path("=candidate.png")
    Image.new("RGB", (8, 8), "gray").save(candidate)
    manifest = Path("manifest.csv")
    write_manifest([], manifest)
    review = Path("review.csv")
    with review.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["candidate_path", "group_id", "review_label"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "candidate_path": "'=candidate.png",
                "group_id": "'=group",
                "review_label": "negative",
            }
        )

    output = Path("with_negatives.csv")
    summary = import_reviewed_negatives(review, manifest, output)
    row = read_manifest(output)[0]

    assert summary["added_negatives"] == 1
    assert row["image_path"] == str(candidate.resolve())
    assert row["group_id"] == "=group"
