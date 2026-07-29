"""Held-out dataset evaluation for tunnel crack segmentation."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ourbrain_cv.metrics import compute_segmentation_metrics


def _safe_div(numerator: int, denominator: int, *, empty: float = 1.0) -> float:
    return empty if denominator == 0 else numerator / denominator


def evaluate_dataset(
    model: nn.Module,
    dataset: Any,
    *,
    device: str | torch.device = "cpu",
    threshold: float = 0.5,
    boundary_tolerance: int = 2,
    image_level_minimum_pixels: int = 16,
    minimum_component_pixels: int = 1,
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate a checkpoint and aggregate pixel/image confusion counts."""
    dev = torch.device(device)
    model.to(dev).eval()
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    with torch.no_grad():
        for sample in dataset:
            pixel_values = sample["pixel_values"].unsqueeze(0).to(dev)
            labels = sample["labels"].unsqueeze(0).to(dev)
            output = model(pixel_values=pixel_values, labels=labels)
            logits = output.logits if hasattr(output, "logits") else output["logits"]
            rows.append(
                compute_segmentation_metrics(
                    logits.detach().cpu(),
                    labels.detach().cpu(),
                    threshold=threshold,
                    boundary_tolerance=boundary_tolerance,
                    image_level_min_pixels=image_level_minimum_pixels,
                    minimum_component_pixels=minimum_component_pixels,
                )
            )

    elapsed = time.perf_counter() - started
    count_fields = [
        "tp",
        "fp",
        "tn",
        "fn",
        "image_tp",
        "image_fp",
        "image_tn",
        "image_fn",
    ]
    counts = {field: int(sum(int(row[field]) for row in rows)) for field in count_fields}
    tp, fp, tn, fn = counts["tp"], counts["fp"], counts["tn"], counts["fn"]
    image_tp = counts["image_tp"]
    image_fp = counts["image_fp"]
    image_tn = counts["image_tn"]
    image_fn = counts["image_fn"]
    result: dict[str, Any] = {
        "samples": len(rows),
        "threshold": threshold,
        "boundary_tolerance": boundary_tolerance,
        "image_level_minimum_pixels": image_level_minimum_pixels,
        "minimum_component_pixels": minimum_component_pixels,
        "crack_iou": _safe_div(tp, tp + fp + fn),
        "crack_dice": _safe_div(2 * tp, 2 * tp + fp + fn),
        "precision": _safe_div(tp, tp + fp),
        "recall": _safe_div(tp, tp + fn),
        "specificity": _safe_div(tn, tn + fp),
        "boundary_precision": float(
            np.mean([row["boundary_precision"] for row in rows])
        )
        if rows
        else 0.0,
        "boundary_recall": float(np.mean([row["boundary_recall"] for row in rows]))
        if rows
        else 0.0,
        "boundary_f1": float(np.mean([row["boundary_f1"] for row in rows]))
        if rows
        else 0.0,
        "image_level_recall": _safe_div(image_tp, image_tp + image_fn),
        "image_level_specificity": _safe_div(image_tn, image_tn + image_fp),
        **counts,
        "elapsed_seconds": elapsed,
        "samples_per_second": len(rows) / elapsed if elapsed > 0 else 0.0,
    }
    if output_json is not None:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        result["output_json"] = str(path.resolve())
    return result
