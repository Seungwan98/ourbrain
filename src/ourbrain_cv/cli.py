"""Command-line interface for the OurBrain tunnel crack pipeline."""

from __future__ import annotations

import argparse
import json
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
    return {
        "manifest": str(Path(manifest).expanduser().resolve()),
        "rows": len(rows),
        "reviewed_negative_counts": {
            split: negative_counts[split] for split in required_splits
        },
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
        threshold=float(inference_cfg.get("threshold", 0.5)),
        boundary_tolerance=args.boundary_tolerance,
        image_level_minimum_pixels=int(
            inference_cfg.get("image_level_minimum_pixels", 16)
        ),
        output_json=args.output,
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
    infer.set_defaults(handler=_infer)

    evaluate = subparsers.add_parser("evaluate", help="evaluate a checkpoint on a held-out split")
    evaluate.add_argument("--config", default="configs/upernet_swin_tiny.yaml")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--split", choices=["train", "val", "test"], default="test")
    evaluate.add_argument("--output", default="artifacts/test_metrics.json")
    evaluate.add_argument("--boundary-tolerance", type=int, default=2)
    evaluate.add_argument("--max-samples", type=int)
    evaluate.add_argument("--device", choices=["cpu", "mps", "cuda"])
    evaluate.set_defaults(handler=_evaluate)

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
