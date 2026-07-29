from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from ourbrain_cv.cli import _validate_training_manifest, build_parser
from ourbrain_cv.manifest import write_manifest
from ourbrain_cv.reviews import import_reviewed_negatives


def _row(split: str, source_kind: str = "paired") -> dict[str, str]:
    return {
        "image_path": f"/{split}-{source_kind}.png",
        "mask_path": "/mask.png" if source_kind == "paired" else "",
        "group_id": f"{split}-{source_kind}",
        "split": split,
        "width": "8",
        "height": "8",
        "mask_width": "8" if source_kind == "paired" else "",
        "mask_height": "8" if source_kind == "paired" else "",
        "positive_pixels": "1" if source_kind == "paired" else "0",
        "source_kind": source_kind,
    }


def _build_reviewed_manifest(tmp_path: Path) -> Path:
    base_manifest = tmp_path / "base.csv"
    write_manifest([_row(split) for split in ("train", "val", "test")], base_manifest)
    review = tmp_path / "review.csv"
    with review.open("w", newline="", encoding="utf-8") as handle:
        import csv

        writer = csv.DictWriter(
            handle,
            fieldnames=["candidate_path", "group_id", "review_label"],
        )
        writer.writeheader()
        for split in ("train", "val", "test"):
            candidate = tmp_path / f"{split}.png"
            Image.new("RGB", (8, 8), "gray").save(candidate)
            writer.writerow(
                {
                    "candidate_path": str(candidate),
                    "group_id": f"{split}-paired",
                    "review_label": "negative",
                }
            )
    manifest = tmp_path / "manifest.csv"
    import_reviewed_negatives(review, base_manifest, manifest)
    return manifest


def test_training_manifest_requires_reviewed_negatives_in_every_split(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    write_manifest([_row(split) for split in ("train", "val", "test")], manifest)

    with pytest.raises(RuntimeError, match="train, val, test"):
        _validate_training_manifest(manifest, allow_positive_only=False)

    summary = _validate_training_manifest(manifest, allow_positive_only=True)
    assert summary["positive_only_override"] is True


def test_training_manifest_accepts_reviewed_negatives_in_every_split(tmp_path: Path):
    manifest = _build_reviewed_manifest(tmp_path)
    summary = _validate_training_manifest(manifest, allow_positive_only=False)
    assert summary["reviewed_negative_counts"] == {"train": 1, "val": 1, "test": 1}
    assert summary["positive_only_override"] is False
    assert summary["review_rows"] == 3
    assert summary["review_audit"].endswith("manifest.review.json")


def test_training_manifest_rejects_post_import_tampering(tmp_path: Path):
    manifest = _build_reviewed_manifest(tmp_path)
    manifest.write_text(manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="changed after reviewed-negative import"):
        _validate_training_manifest(manifest, allow_positive_only=False)


def test_training_manifest_rejects_missing_review_provenance(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    rows = [_row(split, "reviewed_negative") for split in ("train", "val", "test")]
    write_manifest(rows, manifest)

    with pytest.raises(RuntimeError, match="provenance is missing"):
        _validate_training_manifest(manifest, allow_positive_only=False)


def test_train_parser_supports_manifest_override_and_smoke_bypass():
    args = build_parser().parse_args(
        [
            "train",
            "--manifest",
            "custom.csv",
            "--allow-positive-only",
        ]
    )

    assert args.manifest == "custom.csv"
    assert args.allow_positive_only is True


def test_review_ui_parser_has_portable_defaults():
    args = build_parser().parse_args(["review-ui"])

    assert args.review == "data/negative_review/negative_review.csv"
    assert args.manifest == "artifacts/manifest.csv"
    assert args.output == "data/negative_review/review.html"
    assert args.serve is False
    assert args.host == "127.0.0.1"
    assert args.port == 8765
