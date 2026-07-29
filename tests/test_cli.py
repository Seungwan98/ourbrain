from __future__ import annotations

from pathlib import Path

import pytest

from ourbrain_cv.cli import _validate_training_manifest, build_parser
from ourbrain_cv.manifest import write_manifest


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


def test_training_manifest_requires_reviewed_negatives_in_every_split(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    write_manifest([_row(split) for split in ("train", "val", "test")], manifest)

    with pytest.raises(RuntimeError, match="train, val, test"):
        _validate_training_manifest(manifest, allow_positive_only=False)

    summary = _validate_training_manifest(manifest, allow_positive_only=True)
    assert summary["positive_only_override"] is True


def test_training_manifest_accepts_reviewed_negatives_in_every_split(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    rows = [_row(split) for split in ("train", "val", "test")]
    rows += [_row(split, "reviewed_negative") for split in ("train", "val", "test")]
    write_manifest(rows, manifest)

    summary = _validate_training_manifest(manifest, allow_positive_only=False)

    assert summary["reviewed_negative_counts"] == {"train": 1, "val": 1, "test": 1}
    assert summary["positive_only_override"] is False


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
