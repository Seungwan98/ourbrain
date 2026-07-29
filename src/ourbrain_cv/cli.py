"""Command-line interface for the OurBrain tunnel crack pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from ourbrain_cv.config import load_config


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


def _training_config(raw: dict[str, Any]) -> dict[str, Any]:
    training = dict(raw["training"])
    training.setdefault("seed", int(raw.get("seed", 42)))
    return training


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


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_sha256(checkpoint: str | Path) -> str:
    root = Path(checkpoint).expanduser().resolve()
    if root.is_file():
        files = [root]
        relative_to = root.parent
    elif root.is_dir():
        files = sorted(
            {
                *root.glob("*.safetensors"),
                *root.glob("*.bin"),
                *root.glob("*.json"),
            }
        )
        relative_to = root
    else:
        raise ValueError(f"checkpoint does not exist: {root}")
    if not files:
        raise ValueError(f"checkpoint has no model/config files: {root}")

    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(relative_to).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


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
    allowed = {"checkpoint", "num_labels", "id2label", "label2id"}
    return {key: value for key, value in raw["model"].items() if key in allowed}


def _validate_training_manifest(
    manifest: str | Path, *, allow_positive_only: bool
) -> dict[str, Any]:
    """Require reviewed negatives in every split before a real training run."""

    from ourbrain_cv.manifest import read_manifest
    from ourbrain_cv.reviews import review_audit_path, validate_review_audit

    rows = read_manifest(manifest)
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
    }


def _train(args: argparse.Namespace) -> int:
    from ourbrain_cv.data import TunnelCrackSegmentationDataset
    from ourbrain_cv.training import train_model
    from ourbrain_cv.transforms import JointSegmentationTransform

    raw = load_config(args.config)
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
        transform=JointSegmentationTransform(image_size=image_size, train=True, seed=seed),
        synthetic_negative_probability=float(
            data_cfg.get("synthetic_negative_probability", 0.0)
        ),
        synthetic_negative_crop_size=int(data_cfg.get("synthetic_negative_crop_size", 256)),
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


def _infer(args: argparse.Namespace) -> int:
    from ourbrain_cv.inference import (
        TiledInferenceConfig,
        load_checkpoint_adapter,
        run_tiled_inference,
    )
    from ourbrain_cv.training import auto_device

    raw = load_config(args.config)
    inference_cfg = dict(raw["inference"])
    inference_cfg["memmap_dir"] = args.memmap_dir
    inference_cfg["threshold"] = _decision_threshold(
        args,
        inference_cfg,
        manifest=raw["data"]["manifest"],
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
    data_cfg = raw["data"]
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
        boundary_tolerance=args.boundary_tolerance,
        image_level_minimum_pixels=int(
            inference_cfg.get("image_level_minimum_pixels", 16)
        ),
        minimum_component_pixels=int(
            inference_cfg.get("minimum_component_pixels", 8)
        ),
        output_json=args.output,
    )
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
    data_cfg = raw["data"]
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
        require_both_image_classes=not args.allow_positive_only,
        provenance={
            "selection_split": "val",
            "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
            "checkpoint_sha256": _checkpoint_sha256(args.checkpoint),
            "manifest": str(manifest),
            "manifest_sha256": _sha256_file(manifest),
            "review_audit": data_summary["review_audit"],
            "positive_only_override": data_summary["positive_only_override"],
        },
        output_json=args.output,
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

    train = subparsers.add_parser("train", help="fine-tune UPerNet-Swin-Tiny")
    train.add_argument("--config", default="configs/upernet_swin_tiny.yaml")
    train.add_argument("--manifest", help="override data.manifest from the config")
    train.add_argument(
        "--allow-positive-only",
        action="store_true",
        help="bypass reviewed-negative gate for smoke testing only",
    )
    train.add_argument("--device", choices=["cpu", "mps", "cuda"])
    train.set_defaults(handler=_train)

    infer = subparsers.add_parser("infer", help="run tiled inference over one large image")
    infer.add_argument("--config", default="configs/upernet_swin_tiny.yaml")
    infer.add_argument("--checkpoint", required=True)
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
    calibrate.add_argument(
        "--thresholds",
        type=_parse_thresholds,
        default=_parse_thresholds("0.3,0.4,0.5,0.6,0.7,0.8,0.9"),
    )
    calibrate.add_argument("--minimum-image-recall", type=_probability, default=0.95)
    calibrate.add_argument("--output", default="artifacts/threshold_calibration.json")
    calibrate.add_argument("--max-samples", type=int)
    calibrate.add_argument("--device", choices=["cpu", "mps", "cuda"])
    calibrate.add_argument(
        "--allow-positive-only",
        action="store_true",
        help="bypass reviewed-negative gate for smoke testing only",
    )
    calibrate.set_defaults(handler=_calibrate)

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
