"""Held-out dataset evaluation for tunnel crack segmentation."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ourbrain_cv.losses import crack_probabilities
from ourbrain_cv.metrics import (
    compute_segmentation_metrics,
    confusion_counts,
    filter_small_components,
)


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


def calibrate_threshold(
    model: nn.Module,
    dataset: Any,
    *,
    thresholds: list[float],
    device: str | torch.device = "cpu",
    minimum_image_recall: float = 0.95,
    image_level_minimum_pixels: int = 16,
    minimum_component_pixels: int = 8,
    require_both_image_classes: bool = True,
    provenance: dict[str, Any] | None = None,
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    """Select the most specific threshold meeting an image-recall constraint."""

    candidates = sorted(set(float(value) for value in thresholds))
    if not candidates or any(
        not math.isfinite(value) or value < 0 or value > 1 for value in candidates
    ):
        raise ValueError("thresholds must contain values between 0 and 1")
    if not 0 <= minimum_image_recall <= 1:
        raise ValueError("minimum_image_recall must be between 0 and 1")
    if image_level_minimum_pixels < 1:
        raise ValueError("image_level_minimum_pixels must be at least 1")
    if minimum_component_pixels < 1:
        raise ValueError("minimum_component_pixels must be at least 1")

    count_fields = (
        "tp",
        "fp",
        "tn",
        "fn",
        "image_tp",
        "image_fp",
        "image_tn",
        "image_fn",
    )
    counts_by_threshold = {
        threshold: {field: 0 for field in count_fields} for threshold in candidates
    }
    dev = torch.device(device)
    model.to(dev).eval()
    sample_count = 0
    started = time.perf_counter()

    with torch.no_grad():
        for sample in dataset:
            pixel_values = sample["pixel_values"].unsqueeze(0).to(dev)
            labels = sample["labels"].unsqueeze(0).to(dev)
            output = model(pixel_values=pixel_values, labels=labels)
            logits = output.logits if hasattr(output, "logits") else output["logits"]
            probabilities = crack_probabilities(logits.detach().cpu())
            target = labels.detach().cpu().bool()
            sample_count += 1

            for threshold in candidates:
                prediction = filter_small_components(
                    probabilities > threshold,
                    minimum_component_pixels,
                )
                pixel_counts = confusion_counts(prediction, target)
                predicted_image = (
                    prediction.flatten(1).sum(dim=1)
                    >= image_level_minimum_pixels
                )
                target_image = (
                    target.flatten(1).sum(dim=1)
                    >= image_level_minimum_pixels
                )
                image_counts = confusion_counts(predicted_image, target_image)
                current = counts_by_threshold[threshold]
                current["tp"] += pixel_counts.tp
                current["fp"] += pixel_counts.fp
                current["tn"] += pixel_counts.tn
                current["fn"] += pixel_counts.fn
                current["image_tp"] += image_counts.tp
                current["image_fp"] += image_counts.fp
                current["image_tn"] += image_counts.tn
                current["image_fn"] += image_counts.fn

    if sample_count == 0:
        raise ValueError("threshold calibration requires at least one sample")

    curve: list[dict[str, Any]] = []
    for threshold in candidates:
        counts = counts_by_threshold[threshold]
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        image_tp = counts["image_tp"]
        image_fp = counts["image_fp"]
        image_tn = counts["image_tn"]
        image_fn = counts["image_fn"]
        curve.append(
            {
                "threshold": threshold,
                "crack_dice": _safe_div(2 * tp, 2 * tp + fp + fn),
                "image_level_recall": _safe_div(
                    image_tp, image_tp + image_fn
                ),
                "image_level_specificity": _safe_div(
                    image_tn, image_tn + image_fp
                ),
                **counts,
            }
        )

    image_positive_samples = curve[0]["image_tp"] + curve[0]["image_fn"]
    image_negative_samples = curve[0]["image_tn"] + curve[0]["image_fp"]
    if require_both_image_classes and (
        image_positive_samples == 0 or image_negative_samples == 0
    ):
        raise ValueError(
            "threshold calibration requires both crack and reviewed-negative "
            "validation images"
        )

    eligible = [
        row
        for row in curve
        if row["image_level_recall"] >= minimum_image_recall
    ]
    if eligible:
        selected = max(
            eligible,
            key=lambda row: (
                row["image_level_specificity"],
                row["crack_dice"],
                row["threshold"],
            ),
        )
        constraint_met = True
    else:
        selected = max(
            curve,
            key=lambda row: (
                row["image_level_recall"],
                row["image_level_specificity"],
                row["crack_dice"],
                row["threshold"],
            ),
        )
        constraint_met = False

    result: dict[str, Any] = {
        "schema_version": 1,
        "samples": sample_count,
        "image_positive_samples": image_positive_samples,
        "image_negative_samples": image_negative_samples,
        "selected_threshold": selected["threshold"],
        "selected_metrics": selected,
        "minimum_image_recall": minimum_image_recall,
        "recall_constraint_met": constraint_met,
        "image_level_minimum_pixels": image_level_minimum_pixels,
        "minimum_component_pixels": minimum_component_pixels,
        "selection_policy": (
            "maximize image specificity subject to minimum image recall; "
            "tie-break by crack Dice and higher threshold"
        ),
        "curve": curve,
        "elapsed_seconds": time.perf_counter() - started,
    }
    if provenance is not None:
        result["provenance"] = provenance
    if output_json is not None:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        result["output_json"] = str(path.resolve())
    return result
