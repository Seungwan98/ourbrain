from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from ourbrain_cv.cli import (
    _checkpoint_sha256,
    _decision_threshold,
    _sha256_file,
    _validate_training_manifest,
    build_parser,
)
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


def test_decision_threshold_prefers_explicit_then_calibration(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"checkpoint")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("image_path,mask_path,split\n", encoding="utf-8")
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "selected_threshold": 0.73,
                "recall_constraint_met": True,
                "minimum_component_pixels": 8,
                "image_level_minimum_pixels": 16,
                "provenance": {
                    "selection_split": "val",
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": _checkpoint_sha256(checkpoint),
                    "manifest": str(manifest),
                    "manifest_sha256": _sha256_file(manifest),
                },
            }
        ),
        encoding="utf-8",
    )

    calibrated_args = build_parser().parse_args(
        [
            "evaluate",
            "--checkpoint",
            str(checkpoint),
            "--calibration",
            str(calibration),
        ]
    )
    explicit_args = build_parser().parse_args(
        [
            "evaluate",
            "--checkpoint",
            str(checkpoint),
            "--threshold",
            "0.61",
        ]
    )

    inference_cfg = {
        "threshold": 0.5,
        "minimum_component_pixels": 8,
        "image_level_minimum_pixels": 16,
    }
    assert _decision_threshold(calibrated_args, inference_cfg, manifest=manifest) == 0.73
    assert _decision_threshold(explicit_args, inference_cfg) == 0.61


def test_calibrate_parser_validates_threshold_grid():
    args = build_parser().parse_args(
        [
            "calibrate",
            "--checkpoint",
            "checkpoint",
            "--thresholds",
            "0.25,0.5,0.75",
        ]
    )

    assert args.thresholds == [0.25, 0.5, 0.75]


def test_threshold_arguments_reject_non_finite_and_out_of_range_values():
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            ["evaluate", "--checkpoint", "checkpoint", "--threshold", "nan"]
        )
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["calibrate", "--checkpoint", "checkpoint", "--thresholds", "0.5,1.1"]
        )


def test_calibration_rejects_checkpoint_or_postprocessing_mismatch(tmp_path: Path):
    expected_checkpoint = tmp_path / "expected-checkpoint"
    expected_checkpoint.mkdir()
    (expected_checkpoint / "model.safetensors").write_bytes(b"expected")
    other_checkpoint = tmp_path / "other-checkpoint"
    other_checkpoint.mkdir()
    (other_checkpoint / "model.safetensors").write_bytes(b"other")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("image_path,mask_path,split\n", encoding="utf-8")
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "selected_threshold": 0.7,
                "recall_constraint_met": True,
                "minimum_component_pixels": 8,
                "image_level_minimum_pixels": 16,
                "provenance": {
                    "selection_split": "val",
                    "checkpoint": str(expected_checkpoint),
                    "checkpoint_sha256": _checkpoint_sha256(expected_checkpoint),
                    "manifest": str(manifest),
                    "manifest_sha256": _sha256_file(manifest),
                },
            }
        ),
        encoding="utf-8",
    )
    args = build_parser().parse_args(
        [
            "evaluate",
            "--checkpoint",
            str(other_checkpoint),
            "--calibration",
            str(calibration),
        ]
    )
    inference_cfg = {
        "threshold": 0.5,
        "minimum_component_pixels": 8,
        "image_level_minimum_pixels": 16,
    }

    with pytest.raises(ValueError, match="checkpoint does not match"):
        _decision_threshold(args, inference_cfg, manifest=manifest)

    args.checkpoint = str(expected_checkpoint)
    inference_cfg["minimum_component_pixels"] = 9
    with pytest.raises(ValueError, match="minimum_component_pixels"):
        _decision_threshold(args, inference_cfg, manifest=manifest)

    inference_cfg["minimum_component_pixels"] = 8
    (expected_checkpoint / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="content hash"):
        _decision_threshold(args, inference_cfg, manifest=manifest)


def test_calibration_rejects_unmet_recall_constraint(tmp_path: Path):
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "selected_threshold": 0.3,
                "recall_constraint_met": False,
                "provenance": {"selection_split": "val"},
            }
        ),
        encoding="utf-8",
    )
    args = build_parser().parse_args(
        [
            "infer",
            "--checkpoint",
            "checkpoint",
            "--input",
            "scan.bmp",
            "--output",
            "outputs/scan",
            "--calibration",
            str(calibration),
        ]
    )

    with pytest.raises(ValueError, match="did not meet"):
        _decision_threshold(args, {"threshold": 0.5})


def test_calibration_requires_complete_provenance(tmp_path: Path):
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "selected_threshold": 0.7,
                "recall_constraint_met": True,
                "minimum_component_pixels": 8,
                "image_level_minimum_pixels": 16,
                "provenance": {"selection_split": "val"},
            }
        ),
        encoding="utf-8",
    )
    args = build_parser().parse_args(
        [
            "evaluate",
            "--checkpoint",
            str(tmp_path / "checkpoint"),
            "--calibration",
            str(calibration),
        ]
    )

    with pytest.raises(ValueError, match="checkpoint does not match"):
        _decision_threshold(args, {"threshold": 0.5}, manifest=tmp_path / "manifest.csv")
