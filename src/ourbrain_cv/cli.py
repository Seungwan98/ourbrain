"""Command-line interface for the OurBrain tunnel crack pipeline."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ourbrain_cv.config import load_config
from ourbrain_cv.provenance import checkpoint_sha256 as _checkpoint_sha256
from ourbrain_cv.provenance import sha256_file as _sha256_file
from ourbrain_cv.provenance import write_json_atomic as _write_json_atomic


def _resolve_dataset_layout(data_root: Path) -> tuple[Path | None, Path | None]:
    """Support both a volume root and a direct ``train`` directory."""
    direct_images = data_root / "crack"
    direct_masks = data_root / "label" / "crack"
    if direct_images.is_dir() and direct_masks.is_dir():
        return direct_images, direct_masks
    return None, None


def _prepare(args: argparse.Namespace) -> int:
    from ourbrain_cv.manifest import build_manifest, write_manifest

    data_root = Path(args.data_root).expanduser().resolve()
    image_dir, mask_dir = _resolve_dataset_layout(data_root)
    result = build_manifest(
        data_root,
        image_dir=image_dir,
        mask_dir=mask_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        mask_threshold=args.mask_threshold,
    )
    manifest_path = write_manifest(result.rows, args.manifest)
    audit_path = Path(args.audit).expanduser().resolve()
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(result.audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "manifest": str(manifest_path.resolve()),
                "audit": str(audit_path),
                "paired": result.audit["paired"],
                "groups": result.audit["groups"],
                "split_counts": result.audit["split_counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _negative_candidates(args: argparse.Namespace) -> int:
    from ourbrain_cv.negative_candidates import generate_review_set

    review_path = generate_review_set(
        raw_root=args.raw_root,
        manifest_path=args.manifest,
        output_dir=args.output,
        tile_size=args.tile_size,
        stride=args.stride,
        exclusion_margin=args.exclusion_margin,
        max_candidates=args.max_candidates,
        max_raw_images=args.max_raw_images,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "review_csv": str(review_path),
                "status": "human_review_required",
                "accepted_negative_labels": ["negative", "0"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _pilot_hard_negatives(args: argparse.Namespace) -> int:
    from ourbrain_cv.pilot_hard_negatives import build_pilot_hard_negative_review

    result = build_pilot_hard_negative_review(
        args.review,
        args.output,
        tile_size=args.tile_size,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _training_config(raw: dict[str, Any]) -> dict[str, Any]:
    training = dict(raw["training"])
    training.setdefault("seed", int(raw.get("seed", 42)))
    return training


def _apply_training_overrides(
    raw: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    model_checkpoint = getattr(args, "model_checkpoint", None)
    output_dir = getattr(args, "output_dir", None)
    if model_checkpoint:
        raw["model"]["checkpoint"] = model_checkpoint
    if output_dir:
        raw["training"]["output_dir"] = output_dir
    for field in (
        "epochs",
        "freeze_backbone_epochs",
        "max_train_samples",
        "max_val_samples",
    ):
        value = getattr(args, field, None)
        if value is not None:
            raw["training"][field] = value
    if Path(str(raw["model"]["checkpoint"])).expanduser().resolve() == Path(
        str(raw["training"]["output_dir"])
    ).expanduser().resolve():
        raise ValueError(
            "model checkpoint and training output directory must be different"
        )


def _parse_thresholds(value: str) -> list[float]:
    try:
        thresholds = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("thresholds must be comma-separated numbers") from exc
    if not thresholds or any(
        not math.isfinite(item) or item < 0 or item > 1 for item in thresholds
    ):
        raise argparse.ArgumentTypeError("thresholds must be between 0 and 1")
    return thresholds


def _probability(value: str) -> float:
    try:
        probability = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a number between 0 and 1") from exc
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise argparse.ArgumentTypeError("value must be between 0 and 1")
    return probability


def _validate_decision_threshold(value: Any, source: str) -> float:
    threshold = float(value)
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError(f"{source} threshold must be between 0 and 1")
    return threshold


def _decision_threshold(
    args: argparse.Namespace,
    inference_cfg: dict[str, Any],
    *,
    manifest: str | Path | None = None,
) -> float:
    if getattr(args, "threshold", None) is not None:
        return _validate_decision_threshold(args.threshold, "explicit")
    calibration = getattr(args, "calibration", None)
    if calibration:
        payload = json.loads(Path(calibration).expanduser().read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported or missing calibration schema_version")
        if "selected_threshold" not in payload:
            raise ValueError("calibration JSON has no selected_threshold")
        if payload.get("recall_constraint_met") is not True:
            raise ValueError(
                "calibration did not meet its image-recall constraint; "
                "do not use its fallback threshold for evaluation or inference"
            )
        provenance = payload.get("provenance")
        if not isinstance(provenance, dict):
            raise ValueError("calibration JSON has no provenance object")
        if provenance.get("selection_split") != "val":
            raise ValueError("calibration threshold must have been selected on the val split")
        calibrated_checkpoint = provenance.get("checkpoint")
        if not calibrated_checkpoint or Path(calibrated_checkpoint).resolve() != Path(
            args.checkpoint
        ).expanduser().resolve():
            raise ValueError("calibration checkpoint does not match --checkpoint")
        calibrated_checkpoint_sha = provenance.get("checkpoint_sha256")
        if (
            not calibrated_checkpoint_sha
            or calibrated_checkpoint_sha != _checkpoint_sha256(args.checkpoint)
        ):
            raise ValueError("calibration checkpoint content hash does not match --checkpoint")
        calibrated_manifest = provenance.get("manifest")
        calibrated_manifest_sha = provenance.get("manifest_sha256")
        if manifest is None or not calibrated_manifest or not calibrated_manifest_sha:
            raise ValueError("calibration JSON has incomplete manifest provenance")
        current_manifest = Path(manifest).expanduser().resolve()
        if Path(calibrated_manifest).resolve() != current_manifest:
            raise ValueError("calibration manifest does not match the configured manifest")
        if calibrated_manifest_sha != _sha256_file(current_manifest):
            raise ValueError("calibration manifest content hash does not match")
        postprocessing_defaults = {
            "image_level_minimum_pixels": 16,
            "minimum_component_pixels": 8,
        }
        for field, default in postprocessing_defaults.items():
            if field not in payload or int(payload[field]) != int(
                inference_cfg.get(field, default)
            ):
                raise ValueError(
                    f"calibration {field} does not match the inference config"
                )
        if hasattr(args, "boundary_tolerance") and (
            "boundary_tolerance" not in payload
            or int(payload["boundary_tolerance"]) != int(args.boundary_tolerance)
        ):
            raise ValueError(
                "calibration boundary_tolerance does not match evaluation"
            )
        positive_only_override = provenance.get("positive_only_override")
        if positive_only_override is not True:
            calibrated_review_audit = provenance.get("review_audit")
            calibrated_review_audit_sha = provenance.get("review_audit_sha256")
            current_review_audit = current_manifest.with_suffix(".review.json")
            if (
                not calibrated_review_audit
                or not calibrated_review_audit_sha
                or Path(calibrated_review_audit).resolve() != current_review_audit
                or not current_review_audit.is_file()
            ):
                raise ValueError(
                    "calibration JSON has incomplete review audit provenance"
                )
            if calibrated_review_audit_sha != _sha256_file(current_review_audit):
                raise ValueError("calibration review audit content hash does not match")
        return _validate_decision_threshold(
            payload["selected_threshold"], "calibration selected"
        )
    return _validate_decision_threshold(
        inference_cfg.get("threshold", 0.5), "config"
    )


def _import_negatives(args: argparse.Namespace) -> int:
    from ourbrain_cv.reviews import import_reviewed_negatives

    result = import_reviewed_negatives(
        args.review,
        args.manifest,
        args.output,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _review_ui(args: argparse.Namespace) -> int:
    from ourbrain_cv.review_ui import build_negative_review_ui, serve_review_ui

    result = build_negative_review_ui(
        args.review,
        args.output,
        manifest_csv=args.manifest,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.serve:
        serve_review_ui(
            result["review_html"],
            host=args.host,
            port=args.port,
            open_browser=not args.no_open_browser,
        )
    return 0


def _model_config(raw: dict[str, Any]) -> dict[str, Any]:
    allowed = {"architecture", "checkpoint", "num_labels", "id2label", "label2id"}
    return {key: value for key, value in raw["model"].items() if key in allowed}


def _validate_training_manifest(
    manifest: str | Path, *, allow_positive_only: bool
) -> dict[str, Any]:
    """Require reviewed negatives in every split before a real training run."""

    from ourbrain_cv.manifest import read_manifest
    from ourbrain_cv.reviews import review_audit_path, validate_review_audit

    rows = read_manifest(manifest)
    split_by_group: defaultdict[str, set[str]] = defaultdict(set)
    invalid_split_rows: list[int] = []
    blank_group_rows: list[int] = []
    for index, row in enumerate(rows, start=2):
        split = row.get("split", "").strip()
        group_id = row.get("group_id", "").strip()
        if split not in {"train", "val", "test"}:
            invalid_split_rows.append(index)
        if not group_id:
            blank_group_rows.append(index)
        elif split:
            split_by_group[group_id].add(split)
    leaking_groups = {
        group_id: sorted(splits)
        for group_id, splits in split_by_group.items()
        if len(splits) > 1
    }
    if invalid_split_rows or blank_group_rows or leaking_groups:
        raise RuntimeError(
            "Training manifest split integrity failed. "
            f"invalid_split_rows={invalid_split_rows[:10]}, "
            f"blank_group_rows={blank_group_rows[:10]}, "
            f"leaking_groups={dict(list(leaking_groups.items())[:10])}"
        )
    negative_counts = Counter(
        row.get("split", "")
        for row in rows
        if row.get("source_kind", "").strip() == "reviewed_negative"
    )
    required_splits = ("train", "val", "test")
    missing_splits = [split for split in required_splits if negative_counts[split] == 0]
    if missing_splits and not allow_positive_only:
        raise RuntimeError(
            "Reviewed negative examples are required in train/val/test before full training. "
            f"Missing reviewed negatives in: {', '.join(missing_splits)}. "
            "Run negative-candidates, complete human review, and import-negatives. "
            "Use --allow-positive-only only for pipeline smoke tests."
        )
    review_audit = None
    if not allow_positive_only:
        review_audit = validate_review_audit(manifest)
    return {
        "manifest": str(Path(manifest).expanduser().resolve()),
        "rows": len(rows),
        "reviewed_negative_counts": {
            split: negative_counts[split] for split in required_splits
        },
        "review_audit": (
            str(review_audit_path(Path(manifest).expanduser().resolve()))
            if review_audit is not None
            else None
        ),
        "review_rows": review_audit.get("review_rows") if review_audit else None,
        "positive_only_override": bool(missing_splits and allow_positive_only),
        "groups": len(split_by_group),
        "group_leakage_count": len(leaking_groups),
    }


def _verify_training_files(manifest: str | Path) -> dict[str, Any]:
    """Fully decode every manifest image and mask before a GPU training run."""

    from PIL import Image

    from ourbrain_cv.manifest import read_manifest

    rows = read_manifest(manifest)
    errors: list[str] = []
    decoded_images = 0
    decoded_masks = 0
    total_bytes = 0
    source_counts: Counter[str] = Counter()
    mask_size_counts: Counter[str] = Counter()
    masks_resized_to_image = 0

    def expected_size(
        row: dict[str, str],
        prefix: str,
        *,
        row_number: int,
        kind: str,
    ) -> tuple[int, int] | None:
        width = row.get(f"{prefix}width", "").strip()
        height = row.get(f"{prefix}height", "").strip()
        if not width or not height:
            errors.append(f"row {row_number}: manifest {kind} dimensions are missing")
            return None
        try:
            size = int(width), int(height)
        except ValueError:
            errors.append(
                f"row {row_number}: manifest {kind} dimensions are not integers"
            )
            return None
        if size[0] <= 0 or size[1] <= 0:
            errors.append(
                f"row {row_number}: manifest {kind} dimensions must be positive"
            )
            return None
        return size

    def decode(path_value: str, *, row_number: int, kind: str) -> tuple[int, int] | None:
        nonlocal total_bytes
        path = Path(path_value).expanduser()
        if not path.is_file():
            errors.append(f"row {row_number}: missing {kind}: {path}")
            return None
        try:
            total_bytes += path.stat().st_size
            with Image.open(path) as image:
                image.load()
                return image.size
        except (OSError, ValueError) as exc:
            errors.append(f"row {row_number}: cannot decode {kind} {path}: {exc}")
            return None

    for row_number, row in enumerate(rows, start=2):
        image_size: tuple[int, int] | None
        source_kind = row.get("source_kind", "").strip()
        source_counts[source_kind] += 1
        if source_kind not in {"paired", "reviewed_negative"}:
            errors.append(
                f"row {row_number}: unsupported source_kind {source_kind!r}"
            )
        image_size = decode(
            row.get("image_path", ""),
            row_number=row_number,
            kind="image",
        )
        if image_size is not None:
            decoded_images += 1
            recorded = expected_size(
                row,
                "",
                row_number=row_number,
                kind="image",
            )
            if recorded is not None and image_size != recorded:
                errors.append(
                    f"row {row_number}: image size {image_size} != manifest {recorded}"
                )

        mask_value = row.get("mask_path", "").strip()
        if source_kind == "reviewed_negative":
            if mask_value:
                errors.append(
                    f"row {row_number}: reviewed negative unexpectedly has a mask"
                )
            if row.get("positive_pixels", "").strip() != "0":
                errors.append(
                    f"row {row_number}: reviewed negative positive_pixels must be 0"
                )
            continue
        if not mask_value:
            errors.append(f"row {row_number}: paired sample has no mask")
            continue
        mask_size = decode(mask_value, row_number=row_number, kind="mask")
        if mask_size is not None:
            decoded_masks += 1
            mask_size_counts[f"{mask_size[0]}x{mask_size[1]}"] += 1
            recorded_mask = expected_size(
                row,
                "mask_",
                row_number=row_number,
                kind="mask",
            )
            if recorded_mask is not None and mask_size != recorded_mask:
                errors.append(
                    f"row {row_number}: mask size {mask_size} != manifest {recorded_mask}"
                )
            if image_size is not None and mask_size != image_size:
                # The source dataset intentionally contains 682x682/711x711 masks
                # for some 512x512 images. The dataset loader aligns categorical
                # masks to the image using nearest-neighbor interpolation.
                masks_resized_to_image += 1

    if errors:
        preview = "\n".join(errors[:20])
        suffix = f"\n... and {len(errors) - 20} more" if len(errors) > 20 else ""
        raise RuntimeError(
            f"Training file verification failed with {len(errors)} error(s):\n"
            f"{preview}{suffix}"
        )
    return {
        "rows": len(rows),
        "decoded_images": decoded_images,
        "decoded_masks": decoded_masks,
        "total_bytes": total_bytes,
        "source_counts": dict(source_counts),
        "mask_size_counts": dict(mask_size_counts),
        "masks_resized_to_image": masks_resized_to_image,
    }


def _build_training_transform(
    data_cfg: dict[str, Any],
    *,
    image_size: int,
    seed: int,
) -> Any:
    from ourbrain_cv.transforms import JointSegmentationTransform

    augmentation_cfg = data_cfg.get("augmentation", {})
    if not isinstance(augmentation_cfg, dict):
        raise ValueError("data.augmentation must be a mapping")
    return JointSegmentationTransform(
        image_size=image_size,
        train=True,
        horizontal_flip_probability=float(
            augmentation_cfg.get("horizontal_flip_probability", 0.5)
        ),
        vertical_flip_probability=float(
            augmentation_cfg.get("vertical_flip_probability", 0.25)
        ),
        brightness_jitter=float(augmentation_cfg.get("brightness_jitter", 0.15)),
        contrast_jitter=float(augmentation_cfg.get("contrast_jitter", 0.15)),
        rotation_degrees=float(augmentation_cfg.get("rotation_degrees", 0.0)),
        gamma_jitter=float(augmentation_cfg.get("gamma_jitter", 0.0)),
        gaussian_blur_probability=float(
            augmentation_cfg.get("gaussian_blur_probability", 0.0)
        ),
        gaussian_blur_radius=float(
            augmentation_cfg.get("gaussian_blur_radius", 0.0)
        ),
        gaussian_noise_probability=float(
            augmentation_cfg.get("gaussian_noise_probability", 0.0)
        ),
        gaussian_noise_std=float(
            augmentation_cfg.get("gaussian_noise_std", 0.0)
        ),
        seed=seed,
    )


def _train(args: argparse.Namespace) -> int:
    from ourbrain_cv.data import TunnelCrackSegmentationDataset
    from ourbrain_cv.training import train_model
    from ourbrain_cv.transforms import JointSegmentationTransform

    raw = load_config(args.config)
    _apply_training_overrides(raw, args)
    data_cfg = raw["data"]
    image_size = int(data_cfg.get("image_size", 512))
    seed = int(raw.get("seed", 42))
    manifest = args.manifest or data_cfg["manifest"]
    data_summary = _validate_training_manifest(
        manifest, allow_positive_only=args.allow_positive_only
    )

    train_dataset = TunnelCrackSegmentationDataset(
        manifest,
        split="train",
        mask_threshold=int(data_cfg.get("mask_threshold", 127)),
        transform=_build_training_transform(
            data_cfg,
            image_size=image_size,
            seed=seed,
        ),
        synthetic_negative_probability=float(
            data_cfg.get("synthetic_negative_probability", 0.0)
        ),
        synthetic_negative_crop_size=int(data_cfg.get("synthetic_negative_crop_size", 256)),
        crack_centered_probability=float(data_cfg.get("crack_centered_probability", 0.0)),
        crack_centered_crop_size=int(data_cfg.get("crack_centered_crop_size", 384)),
        seed=seed,
    )
    val_dataset = TunnelCrackSegmentationDataset(
        manifest,
        split="val",
        mask_threshold=int(data_cfg.get("mask_threshold", 127)),
        transform=JointSegmentationTransform(image_size=image_size, train=False),
        seed=seed,
    )
    if not train_dataset:
        raise RuntimeError("Training split is empty. Run `ourbrain-cv prepare` first.")
    if not val_dataset:
        raise RuntimeError("Validation split is empty. Check group split ratios.")

    result = train_model(
        train_dataset,
        val_dataset,
        model_config=_model_config(raw),
        config=_training_config(raw),
        device=args.device,
    )
    result["data_summary"] = data_summary
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _training_preflight(args: argparse.Namespace) -> int:
    import torch

    raw = load_config(args.config)
    _apply_training_overrides(raw, args)
    manifest = args.manifest or raw["data"]["manifest"]
    manifest_path = Path(manifest).expanduser()
    if not manifest_path.is_file():
        raise RuntimeError(
            "Training manifest does not exist. Complete remote human review, download the "
            "reviewed CSV, and run import-negatives first: "
            f"{manifest_path.resolve()}"
        )
    data_summary = _validate_training_manifest(
        manifest_path, allow_positive_only=args.allow_positive_only
    )
    file_verification = (
        _verify_training_files(manifest_path) if args.verify_files else None
    )
    checkpoint_value = str(raw["model"]["checkpoint"])
    checkpoint_path = Path(checkpoint_value).expanduser()
    checkpoint_is_local = checkpoint_path.exists()
    checkpoint_has_model = checkpoint_is_local and any(
        (
            *checkpoint_path.glob("*.safetensors"),
            *checkpoint_path.glob("*.bin"),
        )
    )
    if args.require_local_checkpoint and not checkpoint_has_model:
        raise RuntimeError(
            "A local model checkpoint with .safetensors or .bin weights is required: "
            f"{checkpoint_path.resolve()}"
        )
    device_details: dict[str, Any] | None = None
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but torch.cuda.is_available() is false."
            )
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        device_details = {
            "type": "cuda",
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
        }
    elif args.device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS was requested but torch.backends.mps.is_available() is false."
            )
        device_details = {"type": "mps"}
    elif args.device == "cpu":
        device_details = {"type": "cpu"}
    result = {
        "status": "ready",
        "config": str(Path(args.config).expanduser().resolve()),
        "manifest": data_summary,
        "file_verification": file_verification,
        "model_checkpoint": checkpoint_value,
        "model_checkpoint_is_local": checkpoint_is_local,
        "model_checkpoint_has_weights": checkpoint_has_model,
        "output_dir": str(raw["training"]["output_dir"]),
        "device": args.device,
        "device_details": device_details,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _infer(args: argparse.Namespace) -> int:
    from ourbrain_cv.inference import (
        TiledInferenceConfig,
        load_checkpoint_adapter,
        run_tiled_inference,
    )
    from ourbrain_cv.training import auto_device

    raw = load_config(args.config)
    manifest = args.manifest or raw["data"]["manifest"]
    inference_cfg = dict(raw["inference"])
    inference_cfg["memmap_dir"] = args.memmap_dir
    inference_cfg["threshold"] = _decision_threshold(
        args,
        inference_cfg,
        manifest=manifest,
    )
    config = TiledInferenceConfig(**inference_cfg)
    device = args.device or str(auto_device())
    adapter = load_checkpoint_adapter(
        args.checkpoint,
        crack_class=int(raw["model"].get("crack_label", 1)),
        device=device,
    )
    result = run_tiled_inference(args.input, adapter, args.output, config=config)
    print(json.dumps(result.summary, ensure_ascii=False, indent=2))
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    from ourbrain_cv.data import TunnelCrackSegmentationDataset
    from ourbrain_cv.evaluation import evaluate_dataset
    from ourbrain_cv.modeling import enable_mps_compatibility, load_model_for_inference
    from ourbrain_cv.training import auto_device
    from ourbrain_cv.transforms import JointSegmentationTransform

    raw = load_config(args.config)
    device = args.device or str(auto_device())
    data_cfg = dict(raw["data"])
    if args.manifest:
        data_cfg["manifest"] = args.manifest
    inference_cfg = raw["inference"]
    threshold = _decision_threshold(
        args,
        inference_cfg,
        manifest=data_cfg["manifest"],
    )
    dataset = TunnelCrackSegmentationDataset(
        data_cfg["manifest"],
        split=args.split,
        mask_threshold=int(data_cfg.get("mask_threshold", 127)),
        transform=JointSegmentationTransform(
            image_size=int(data_cfg.get("image_size", 512)),
            train=False,
        ),
        seed=int(raw.get("seed", 42)),
    )
    if args.max_samples is not None:
        from torch.utils.data import Subset

        dataset = Subset(dataset, range(min(len(dataset), max(0, args.max_samples))))
    model = load_model_for_inference(args.checkpoint)
    if str(device).startswith("mps"):
        enable_mps_compatibility(model)
    result = evaluate_dataset(
        model,
        dataset,
        device=device,
        threshold=threshold,
        image_level_minimum_pixels=int(
            inference_cfg.get("image_level_minimum_pixels", 16)
        ),
        minimum_component_pixels=int(
            inference_cfg.get("minimum_component_pixels", 8)
        ),
        boundary_tolerance=args.boundary_tolerance,
        output_json=None,
    )
    config_path = Path(args.config).expanduser().resolve()
    manifest = Path(data_cfg["manifest"]).expanduser().resolve()
    review_audit = manifest.with_suffix(".review.json")
    provenance: dict[str, Any] = {
        "config": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "checkpoint_sha256": _checkpoint_sha256(args.checkpoint),
        "manifest": str(manifest),
        "manifest_sha256": _sha256_file(manifest),
        "review_audit": str(review_audit) if review_audit.is_file() else None,
        "review_audit_sha256": (
            _sha256_file(review_audit) if review_audit.is_file() else None
        ),
        "threshold_source": (
            "calibration"
            if args.calibration
            else "explicit"
            if args.threshold is not None
            else "config"
        ),
        "calibration": (
            str(Path(args.calibration).expanduser().resolve())
            if args.calibration
            else None
        ),
        "calibration_sha256": (
            _sha256_file(args.calibration) if args.calibration else None
        ),
    }
    result.update(
        {
            "schema_version": 1,
            "evaluation_split": args.split,
            "provenance": provenance,
        }
    )
    output = _write_json_atomic(args.output, result)
    result["output_json"] = str(output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _calibrate(args: argparse.Namespace) -> int:
    from ourbrain_cv.data import TunnelCrackSegmentationDataset
    from ourbrain_cv.evaluation import calibrate_threshold
    from ourbrain_cv.modeling import enable_mps_compatibility, load_model_for_inference
    from ourbrain_cv.training import auto_device
    from ourbrain_cv.transforms import JointSegmentationTransform

    raw = load_config(args.config)
    device = args.device or str(auto_device())
    data_cfg = dict(raw["data"])
    if args.manifest:
        data_cfg["manifest"] = args.manifest
    inference_cfg = raw["inference"]
    manifest = Path(data_cfg["manifest"]).expanduser().resolve()
    data_summary = _validate_training_manifest(
        manifest,
        allow_positive_only=args.allow_positive_only,
    )
    dataset = TunnelCrackSegmentationDataset(
        manifest,
        split="val",
        mask_threshold=int(data_cfg.get("mask_threshold", 127)),
        transform=JointSegmentationTransform(
            image_size=int(data_cfg.get("image_size", 512)),
            train=False,
        ),
        seed=int(raw.get("seed", 42)),
    )
    if args.max_samples is not None:
        from torch.utils.data import Subset

        dataset = Subset(dataset, range(min(len(dataset), max(0, args.max_samples))))
    if len(dataset) == 0:
        raise RuntimeError("Validation split is empty; threshold calibration cannot run.")
    model = load_model_for_inference(args.checkpoint)
    if str(device).startswith("mps"):
        enable_mps_compatibility(model)
    result = calibrate_threshold(
        model,
        dataset,
        thresholds=args.thresholds,
        device=device,
        minimum_image_recall=args.minimum_image_recall,
        image_level_minimum_pixels=int(
            inference_cfg.get("image_level_minimum_pixels", 16)
        ),
        minimum_component_pixels=int(
            inference_cfg.get("minimum_component_pixels", 8)
        ),
        boundary_tolerance=args.boundary_tolerance,
        require_both_image_classes=not args.allow_positive_only,
        provenance={
            "selection_split": "val",
            "config": str(Path(args.config).expanduser().resolve()),
            "config_sha256": _sha256_file(args.config),
            "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
            "checkpoint_sha256": _checkpoint_sha256(args.checkpoint),
            "manifest": str(manifest),
            "manifest_sha256": _sha256_file(manifest),
            "review_audit": data_summary["review_audit"],
            "review_audit_sha256": (
                _sha256_file(data_summary["review_audit"])
                if data_summary["review_audit"]
                else None
            ),
            "positive_only_override": data_summary["positive_only_override"],
        },
        output_json=None,
    )
    output = _write_json_atomic(args.output, result)
    result["output_json"] = str(output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _select_calibrated_model(args: argparse.Namespace) -> int:
    from ourbrain_cv.model_selection import select_calibrated_model

    result = select_calibrated_model(
        args.candidates,
        args.output,
        minimum_image_recall=args.minimum_image_recall,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _remote_review_bundle(args: argparse.Namespace) -> int:
    from ourbrain_cv.remote_review import build_remote_review_bundle

    result = build_remote_review_bundle(
        args.review,
        args.manifest,
        args.output,
        template_dir=args.template,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _remote_review_status(args: argparse.Namespace) -> int:
    from ourbrain_cv.remote_review import remote_review_status

    result = remote_review_status(
        args.url,
        token_env=args.token_env,
        timeout=args.timeout,
    )
    if args.summary_only:
        result = {
            "datasetId": result.get("datasetId"),
            "revision": result.get("revision"),
            "updatedAt": result.get("updatedAt"),
            "summary": result["summary"],
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _remote_review_download(args: argparse.Namespace) -> int:
    from ourbrain_cv.remote_review import download_remote_review_csv

    result = download_remote_review_csv(
        args.url,
        args.output,
        token_env=args.token_env,
        timeout=args.timeout,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ourbrain-cv",
        description="Tunnel crack segmentation training and tiled inference",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help="audit paired images/masks and write a leakage-safe manifest",
    )
    prepare.add_argument("--data-root", required=True)
    prepare.add_argument("--manifest", default="artifacts/manifest.csv")
    prepare.add_argument("--audit", default="artifacts/data_audit.json")
    prepare.add_argument("--train-ratio", type=float, default=0.70)
    prepare.add_argument("--val-ratio", type=float, default=0.15)
    prepare.add_argument("--test-ratio", type=float, default=0.15)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--mask-threshold", type=int, default=127)
    prepare.set_defaults(handler=_prepare)

    negatives = subparsers.add_parser(
        "negative-candidates",
        help="generate raw-image patches for human hard-negative review",
    )
    negatives.add_argument("--raw-root", required=True)
    negatives.add_argument("--manifest", default="artifacts/manifest.csv")
    negatives.add_argument("--output", default="data/negative_review")
    negatives.add_argument("--tile-size", type=int, default=512)
    negatives.add_argument("--stride", type=int, default=512)
    negatives.add_argument("--exclusion-margin", type=int, default=128)
    negatives.add_argument("--max-candidates", type=int, default=200)
    negatives.add_argument(
        "--max-raw-images",
        type=int,
        help="limit the number of sorted raw scans for a small review batch",
    )
    negatives.add_argument("--seed", type=int, default=42)
    negatives.set_defaults(handler=_negative_candidates)

    pilot_hard_negatives = subparsers.add_parser(
        "pilot-hard-negatives",
        help="crop human-confirmed pilot false positives for a second review",
    )
    pilot_hard_negatives.add_argument(
        "--review",
        default="runs/v0.2-pilot/pilot_review.csv",
    )
    pilot_hard_negatives.add_argument(
        "--output",
        default="data/pilot_hard_negatives",
    )
    pilot_hard_negatives.add_argument("--tile-size", type=int, default=512)
    pilot_hard_negatives.set_defaults(handler=_pilot_hard_negatives)

    import_negatives = subparsers.add_parser(
        "import-negatives",
        help="append explicitly reviewed normal patches to a new manifest",
    )
    import_negatives.add_argument("--review", required=True)
    import_negatives.add_argument("--manifest", default="artifacts/manifest.csv")
    import_negatives.add_argument(
        "--output",
        default="artifacts/manifest_with_negatives.csv",
    )
    import_negatives.add_argument("--seed", type=int, default=42)
    import_negatives.set_defaults(handler=_import_negatives)

    review_ui = subparsers.add_parser(
        "review-ui",
        help="build a local keyboard-driven UI for hard-negative review",
    )
    review_ui.add_argument(
        "--review",
        default="data/negative_review/negative_review.csv",
    )
    review_ui.add_argument("--manifest", default="artifacts/manifest.csv")
    review_ui.add_argument(
        "--output",
        default="data/negative_review/review.html",
    )
    review_ui.add_argument("--seed", type=int, default=42)
    review_ui.add_argument(
        "--serve",
        action="store_true",
        help="serve the UI over localhost for stable browser storage",
    )
    review_ui.add_argument(
        "--host",
        default="127.0.0.1",
        choices=["127.0.0.1", "localhost", "::1"],
        help="loopback address for --serve",
    )
    review_ui.add_argument("--port", type=int, default=8765)
    review_ui.add_argument(
        "--no-open-browser",
        action="store_true",
        help="do not open the default browser when --serve is used",
    )
    review_ui.set_defaults(handler=_review_ui)

    train = subparsers.add_parser("train", help="fine-tune a supported segmentation model")
    train.add_argument("--config", default="configs/upernet_swin_tiny.yaml")
    train.add_argument("--manifest", help="override data.manifest from the config")
    train.add_argument(
        "--model-checkpoint",
        help="override model.checkpoint without changing the source config",
    )
    train.add_argument(
        "--output-dir",
        help="override training.output_dir without changing the source config",
    )
    train.add_argument("--epochs", type=int, help="override the maximum epoch count")
    train.add_argument(
        "--freeze-backbone-epochs",
        type=int,
        help="override frozen-backbone epochs for a bounded smoke run",
    )
    train.add_argument(
        "--max-train-samples",
        type=int,
        help="limit training samples for a bounded smoke run",
    )
    train.add_argument(
        "--max-val-samples",
        type=int,
        help="limit validation samples for a bounded smoke run",
    )
    train.add_argument(
        "--allow-positive-only",
        action="store_true",
        help="bypass reviewed-negative gate for smoke testing only",
    )
    train.add_argument("--device", choices=["cpu", "mps", "cuda"])
    train.set_defaults(handler=_train)

    preflight = subparsers.add_parser(
        "training-preflight",
        help="validate training data provenance and checkpoint readiness without training",
    )
    preflight.add_argument("--config", default="configs/upernet_swin_tiny.yaml")
    preflight.add_argument("--manifest", help="override data.manifest from the config")
    preflight.add_argument(
        "--model-checkpoint",
        help="override model.checkpoint without changing the source config",
    )
    preflight.add_argument(
        "--output-dir",
        help="override training.output_dir without changing the source config",
    )
    preflight.add_argument(
        "--allow-positive-only",
        action="store_true",
        help="bypass reviewed-negative gate for smoke testing only",
    )
    preflight.add_argument(
        "--require-local-checkpoint",
        action="store_true",
        help="require model.checkpoint to be a local directory containing model weights",
    )
    preflight.add_argument(
        "--verify-files",
        action="store_true",
        help="fully decode every manifest image and mask before reporting readiness",
    )
    preflight.add_argument("--device", choices=["cpu", "mps", "cuda"])
    preflight.set_defaults(handler=_training_preflight)

    infer = subparsers.add_parser("infer", help="run tiled inference over one large image")
    infer.add_argument("--config", default="configs/upernet_swin_tiny.yaml")
    infer.add_argument("--checkpoint", required=True)
    infer.add_argument("--manifest", help="override data.manifest provenance")
    infer.add_argument("--input", required=True)
    infer.add_argument("--output", required=True)
    infer.add_argument("--device", choices=["cpu", "mps", "cuda"])
    infer.add_argument("--memmap-dir", default="outputs/.memmap")
    infer_threshold = infer.add_mutually_exclusive_group()
    infer_threshold.add_argument("--threshold", type=_probability)
    infer_threshold.add_argument("--calibration")
    infer.set_defaults(handler=_infer)

    evaluate = subparsers.add_parser("evaluate", help="evaluate a checkpoint on a held-out split")
    evaluate.add_argument("--config", default="configs/upernet_swin_tiny.yaml")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--manifest", help="override data.manifest from the config")
    evaluate.add_argument("--split", choices=["train", "val", "test"], default="test")
    evaluate.add_argument("--output", default="artifacts/test_metrics.json")
    evaluate.add_argument("--boundary-tolerance", type=int, default=2)
    evaluate.add_argument("--max-samples", type=int)
    evaluate.add_argument("--device", choices=["cpu", "mps", "cuda"])
    evaluate_threshold = evaluate.add_mutually_exclusive_group()
    evaluate_threshold.add_argument("--threshold", type=_probability)
    evaluate_threshold.add_argument("--calibration")
    evaluate.set_defaults(handler=_evaluate)

    calibrate = subparsers.add_parser(
        "calibrate",
        help="select a validation threshold under an image-recall constraint",
    )
    calibrate.add_argument("--config", default="configs/upernet_swin_tiny.yaml")
    calibrate.add_argument("--checkpoint", required=True)
    calibrate.add_argument("--manifest", help="override data.manifest from the config")
    calibrate.add_argument(
        "--thresholds",
        type=_parse_thresholds,
        default=_parse_thresholds("0.3,0.4,0.5,0.6,0.7,0.8,0.9"),
    )
    calibrate.add_argument("--minimum-image-recall", type=_probability, default=0.95)
    calibrate.add_argument("--output", default="artifacts/threshold_calibration.json")
    calibrate.add_argument("--max-samples", type=int)
    calibrate.add_argument("--boundary-tolerance", type=int, default=2)
    calibrate.add_argument("--device", choices=["cpu", "mps", "cuda"])
    calibrate.add_argument(
        "--allow-positive-only",
        action="store_true",
        help="bypass reviewed-negative gate for smoke testing only",
    )
    calibrate.set_defaults(handler=_calibrate)

    select_model = subparsers.add_parser(
        "select-calibrated-model",
        help="select a validation-calibrated model under the recall constraint",
    )
    select_model.add_argument("--candidates", required=True)
    select_model.add_argument(
        "--minimum-image-recall",
        type=_probability,
        default=0.95,
    )
    select_model.add_argument(
        "--output",
        default="artifacts/model_selection.json",
    )
    select_model.set_defaults(handler=_select_calibrated_model)

    remote_bundle = subparsers.add_parser(
        "remote-review-bundle",
        help="build a Vercel source bundle for authenticated remote review",
    )
    remote_bundle.add_argument(
        "--review",
        default="data/negative_review/negative_review.csv",
    )
    remote_bundle.add_argument("--manifest", default="artifacts/manifest.csv")
    remote_bundle.add_argument(
        "--template",
        default="web/remote-review-template",
    )
    remote_bundle.add_argument(
        "--output",
        default="build/remote-review-app",
    )
    remote_bundle.add_argument("--seed", type=int, default=42)
    remote_bundle.set_defaults(handler=_remote_review_bundle)

    remote_status = subparsers.add_parser(
        "remote-review-status",
        help="fetch progress from the authenticated remote review service",
    )
    remote_status.add_argument("--url", required=True)
    remote_status.add_argument("--token-env", default="OURBRAIN_REVIEW_TOKEN")
    remote_status.add_argument("--timeout", type=float, default=30)
    remote_status.add_argument(
        "--summary-only",
        action="store_true",
        help="omit the full candidate list and print only progress totals",
    )
    remote_status.set_defaults(handler=_remote_review_status)

    remote_download = subparsers.add_parser(
        "remote-review-download",
        help="download remote decisions as a strict-import-compatible CSV",
    )
    remote_download.add_argument("--url", required=True)
    remote_download.add_argument(
        "--output",
        default="data/negative_review/negative_review_reviewed.csv",
    )
    remote_download.add_argument("--token-env", default="OURBRAIN_REVIEW_TOKEN")
    remote_download.add_argument("--timeout", type=float, default=30)
    remote_download.set_defaults(handler=_remote_review_download)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
