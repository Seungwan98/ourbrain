from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from ourbrain_cv.cli import (
    _apply_training_overrides,
    _build_training_transform,
    _checkpoint_sha256,
    _decision_threshold,
    _sha256_file,
    _validate_training_manifest,
    _verify_training_files,
    _write_json_atomic,
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


def test_training_manifest_rejects_group_leakage(tmp_path: Path):
    rows = [
        {**_row("train"), "group_id": "shared"},
        {**_row("val"), "group_id": "shared"},
    ]
    manifest = tmp_path / "manifest.csv"
    write_manifest(rows, manifest)

    with pytest.raises(RuntimeError, match="leaking_groups"):
        _validate_training_manifest(manifest, allow_positive_only=True)


def test_verify_training_files_decodes_images_masks_and_negatives(tmp_path: Path):
    paired_image = tmp_path / "paired.png"
    paired_mask = tmp_path / "paired-mask.png"
    negative_image = tmp_path / "negative.png"
    Image.new("RGB", (8, 8), "gray").save(paired_image)
    Image.new("L", (8, 8), 255).save(paired_mask)
    Image.new("RGB", (8, 8), "gray").save(negative_image)
    manifest = tmp_path / "manifest.csv"
    write_manifest(
        [
            {
                **_row("train"),
                "image_path": str(paired_image),
                "mask_path": str(paired_mask),
            },
            {
                **_row("val", "reviewed_negative"),
                "image_path": str(negative_image),
            },
        ],
        manifest,
    )

    result = _verify_training_files(manifest)

    assert result["rows"] == 2
    assert result["decoded_images"] == 2
    assert result["decoded_masks"] == 1
    assert result["source_counts"] == {"paired": 1, "reviewed_negative": 1}


def test_verify_training_files_reports_missing_and_size_mismatch(tmp_path: Path):
    image = tmp_path / "wrong-size.png"
    Image.new("RGB", (4, 4), "gray").save(image)
    manifest = tmp_path / "manifest.csv"
    write_manifest(
        [
            {
                **_row("train"),
                "image_path": str(image),
                "mask_path": str(tmp_path / "missing-mask.png"),
            }
        ],
        manifest,
    )

    with pytest.raises(RuntimeError, match=r"(?s)image size.*missing mask"):
        _verify_training_files(manifest)


def test_train_parser_supports_manifest_override_and_smoke_bypass():
    args = build_parser().parse_args(
        [
            "train",
            "--manifest",
            "custom.csv",
            "--model-checkpoint",
            "source-checkpoint",
            "--output-dir",
            "round-output",
            "--epochs",
            "1",
            "--freeze-backbone-epochs",
            "0",
            "--max-train-samples",
            "10",
            "--max-val-samples",
            "4",
            "--allow-positive-only",
        ]
    )

    assert args.manifest == "custom.csv"
    assert args.model_checkpoint == "source-checkpoint"
    assert args.output_dir == "round-output"
    assert args.epochs == 1
    assert args.freeze_backbone_epochs == 0
    assert args.max_train_samples == 10
    assert args.max_val_samples == 4
    assert args.allow_positive_only is True


def test_training_overrides_apply_bounded_smoke_budget() -> None:
    raw = {
        "model": {"checkpoint": "source"},
        "training": {"output_dir": "output", "epochs": 30},
    }
    args = build_parser().parse_args(
        [
            "train",
            "--epochs",
            "1",
            "--freeze-backbone-epochs",
            "0",
            "--max-train-samples",
            "10",
            "--max-val-samples",
            "4",
        ]
    )

    _apply_training_overrides(raw, args)

    assert raw["training"]["epochs"] == 1
    assert raw["training"]["freeze_backbone_epochs"] == 0
    assert raw["training"]["max_train_samples"] == 10
    assert raw["training"]["max_val_samples"] == 4


def test_training_transform_reads_augmentation_config() -> None:
    transform = _build_training_transform(
        {
            "augmentation": {
                "horizontal_flip_probability": 0.4,
                "vertical_flip_probability": 0.1,
                "brightness_jitter": 0.2,
                "contrast_jitter": 0.25,
                "rotation_degrees": 7.0,
                "gamma_jitter": 0.12,
                "gaussian_blur_probability": 0.15,
                "gaussian_blur_radius": 0.8,
                "gaussian_noise_probability": 0.2,
                "gaussian_noise_std": 0.01,
            }
        },
        image_size=384,
        seed=7,
    )

    assert transform.image_size == 384
    assert transform.horizontal_flip_probability == 0.4
    assert transform.vertical_flip_probability == 0.1
    assert transform.brightness_jitter == 0.2
    assert transform.contrast_jitter == 0.25
    assert transform.rotation_degrees == 7.0
    assert transform.gamma_jitter == 0.12
    assert transform.gaussian_blur_probability == 0.15
    assert transform.gaussian_blur_radius == 0.8
    assert transform.gaussian_noise_probability == 0.2
    assert transform.gaussian_noise_std == 0.01


def test_training_transform_rejects_non_mapping_augmentation() -> None:
    with pytest.raises(ValueError, match="must be a mapping"):
        _build_training_transform(
            {"augmentation": []},
            image_size=512,
            seed=42,
        )


def test_training_overrides_reject_source_checkpoint_overwrite() -> None:
    raw = {
        "model": {"checkpoint": "original"},
        "training": {"output_dir": "output"},
    }
    args = build_parser().parse_args(
        [
            "train",
            "--model-checkpoint",
            "same-directory",
            "--output-dir",
            "same-directory",
        ]
    )

    with pytest.raises(ValueError, match="must be different"):
        _apply_training_overrides(raw, args)


def test_inference_and_evaluation_parsers_support_manifest_override() -> None:
    parser = build_parser()
    infer = parser.parse_args(
        [
            "infer",
            "--checkpoint",
            "checkpoint",
            "--input",
            "scan.bmp",
            "--output",
            "output",
            "--manifest",
            "round.csv",
        ]
    )
    evaluate = parser.parse_args(
        [
            "evaluate",
            "--checkpoint",
            "checkpoint",
            "--manifest",
            "round.csv",
        ]
    )
    calibrate = parser.parse_args(
        [
            "calibrate",
            "--checkpoint",
            "checkpoint",
            "--manifest",
            "round.csv",
        ]
    )

    assert infer.manifest == "round.csv"
    assert evaluate.manifest == "round.csv"
    assert calibrate.manifest == "round.csv"


def test_training_preflight_validates_review_provenance_and_local_checkpoint(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
):
    import yaml

    manifest = _build_reviewed_manifest(tmp_path)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    config = {
        "seed": 42,
        "model": {
            "checkpoint": str(checkpoint),
            "num_labels": 2,
            "crack_label": 1,
        },
        "data": {
            "manifest": str(manifest),
            "image_size": 512,
        },
        "training": {
            "output_dir": str(tmp_path / "output"),
        },
        "inference": {
            "tile_size": 512,
            "overlap": 96,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    args = build_parser().parse_args(
        [
            "training-preflight",
            "--config",
            str(config_path),
            "--require-local-checkpoint",
            "--device",
            "cpu",
        ]
    )

    assert args.handler(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ready"
    assert payload["manifest"]["reviewed_negative_counts"] == {
        "train": 1,
        "val": 1,
        "test": 1,
    }
    assert payload["model_checkpoint_has_weights"] is True
    assert payload["device"] == "cpu"
    assert payload["device_details"] == {"type": "cpu"}

    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    args.device = "cuda"
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        args.handler(args)


def test_training_preflight_rejects_missing_local_checkpoint(tmp_path: Path):
    import yaml

    manifest = _build_reviewed_manifest(tmp_path)
    config = {
        "model": {
            "checkpoint": str(tmp_path / "missing"),
            "num_labels": 2,
            "crack_label": 1,
        },
        "data": {
            "manifest": str(manifest),
            "image_size": 512,
        },
        "training": {
            "output_dir": str(tmp_path / "output"),
        },
        "inference": {
            "tile_size": 512,
            "overlap": 96,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    args = build_parser().parse_args(
        [
            "training-preflight",
            "--config",
            str(config_path),
            "--require-local-checkpoint",
        ]
    )

    with pytest.raises(RuntimeError, match="local model checkpoint"):
        args.handler(args)


def test_training_preflight_explains_missing_reviewed_manifest(tmp_path: Path):
    import yaml

    config = {
        "model": {
            "checkpoint": "openmmlab/upernet-swin-tiny",
            "num_labels": 2,
            "crack_label": 1,
        },
        "data": {
            "manifest": str(tmp_path / "manifest_with_negatives.csv"),
            "image_size": 512,
        },
        "training": {
            "output_dir": str(tmp_path / "output"),
        },
        "inference": {
            "tile_size": 512,
            "overlap": 96,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    args = build_parser().parse_args(["training-preflight", "--config", str(config_path)])

    with pytest.raises(RuntimeError, match="Complete remote human review"):
        args.handler(args)


def test_review_ui_parser_has_portable_defaults():
    args = build_parser().parse_args(["review-ui"])

    assert args.review == "data/negative_review/negative_review.csv"
    assert args.manifest == "artifacts/manifest.csv"
    assert args.output == "data/negative_review/review.html"
    assert args.serve is False
    assert args.host == "127.0.0.1"
    assert args.port == 8765


def test_remote_review_status_parser_supports_compact_summary() -> None:
    args = build_parser().parse_args(
        [
            "remote-review-status",
            "--url",
            "https://review.example.com",
            "--summary-only",
        ]
    )

    assert args.summary_only is True


def test_write_json_atomic_replaces_existing_document_without_temp_file(
    tmp_path: Path,
):
    output = tmp_path / "metrics.json"
    output.write_text('{"stale": true}', encoding="utf-8")

    written = _write_json_atomic(output, {"schema_version": 1, "value": "완료"})

    assert written == output.resolve()
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "value": "완료",
    }
    assert not (tmp_path / "metrics.json.tmp").exists()


def test_evaluate_cli_records_complete_artifact_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    import yaml

    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "image_path,mask_path,group_id,split,width,height,mask_width,"
        "mask_height,positive_pixels,source_kind\n",
        encoding="utf-8",
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"checkpoint")
    config = {
        "seed": 42,
        "model": {
            "checkpoint": str(checkpoint),
            "num_labels": 2,
            "crack_label": 1,
        },
        "data": {
            "manifest": str(manifest),
            "image_size": 512,
            "mask_threshold": 127,
        },
        "training": {"output_dir": str(tmp_path / "unused")},
        "inference": {
            "tile_size": 512,
            "overlap": 96,
            "threshold": 0.5,
            "minimum_component_pixels": 8,
            "image_level_minimum_pixels": 16,
            "maximum_positive_ratio": 0.25,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = tmp_path / "test-metrics.json"

    monkeypatch.setattr(
        "ourbrain_cv.data.TunnelCrackSegmentationDataset",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "ourbrain_cv.modeling.load_model_for_inference",
        lambda checkpoint_path: object(),
    )
    monkeypatch.setattr(
        "ourbrain_cv.evaluation.evaluate_dataset",
        lambda *args, **kwargs: {
            "samples": 1,
            "threshold": kwargs["threshold"],
            "image_level_recall": 1.0,
        },
    )
    args = build_parser().parse_args(
        [
            "evaluate",
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint),
            "--threshold",
            "0.6",
            "--split",
            "test",
            "--output",
            str(output),
            "--device",
            "cpu",
        ]
    )

    assert args.handler(args) == 0
    capsys.readouterr()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["evaluation_split"] == "test"
    assert payload["threshold"] == 0.6
    assert payload["provenance"]["threshold_source"] == "explicit"
    assert payload["provenance"]["config_sha256"] == _sha256_file(config_path)
    assert payload["provenance"]["checkpoint_sha256"] == _checkpoint_sha256(
        checkpoint
    )
    assert payload["provenance"]["manifest_sha256"] == _sha256_file(manifest)
    assert payload["provenance"]["review_audit"] is None
    assert not (tmp_path / "test-metrics.json.tmp").exists()


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
                "boundary_tolerance": 2,
                "provenance": {
                    "selection_split": "val",
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": _checkpoint_sha256(checkpoint),
                    "manifest": str(manifest),
                    "manifest_sha256": _sha256_file(manifest),
                    "positive_only_override": True,
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
                "boundary_tolerance": 2,
                "provenance": {
                    "selection_split": "val",
                    "checkpoint": str(expected_checkpoint),
                    "checkpoint_sha256": _checkpoint_sha256(expected_checkpoint),
                    "manifest": str(manifest),
                    "manifest_sha256": _sha256_file(manifest),
                    "positive_only_override": True,
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
    args.boundary_tolerance = 3
    with pytest.raises(ValueError, match="boundary_tolerance"):
        _decision_threshold(args, inference_cfg, manifest=manifest)

    args.boundary_tolerance = 2
    (expected_checkpoint / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="content hash"):
        _decision_threshold(args, inference_cfg, manifest=manifest)


def test_calibration_rejects_review_audit_tampering(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"checkpoint")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("image_path,mask_path,split\n", encoding="utf-8")
    review_audit = manifest.with_suffix(".review.json")
    review_audit.write_text('{"complete": true}', encoding="utf-8")
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "selected_threshold": 0.7,
                "recall_constraint_met": True,
                "minimum_component_pixels": 8,
                "image_level_minimum_pixels": 16,
                "boundary_tolerance": 2,
                "provenance": {
                    "selection_split": "val",
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": _checkpoint_sha256(checkpoint),
                    "manifest": str(manifest),
                    "manifest_sha256": _sha256_file(manifest),
                    "review_audit": str(review_audit),
                    "review_audit_sha256": _sha256_file(review_audit),
                    "positive_only_override": False,
                },
            }
        ),
        encoding="utf-8",
    )
    args = build_parser().parse_args(
        [
            "evaluate",
            "--checkpoint",
            str(checkpoint),
            "--calibration",
            str(calibration),
        ]
    )
    inference_cfg = {
        "threshold": 0.5,
        "minimum_component_pixels": 8,
        "image_level_minimum_pixels": 16,
    }

    assert _decision_threshold(args, inference_cfg, manifest=manifest) == 0.7
    review_audit.write_text('{"complete": false}', encoding="utf-8")
    with pytest.raises(ValueError, match="review audit content hash"):
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
