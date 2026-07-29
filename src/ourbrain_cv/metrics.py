"""Segmentation and image-level metrics for crack detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .losses import crack_probabilities
from .postprocessing import remove_small_components


@dataclass(frozen=True)
class ConfusionCounts:
    tp: int
    fp: int
    tn: int
    fn: int

    def as_dict(self) -> dict[str, int]:
        return {"tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn}


def _safe_div(num: float, den: float, empty_value: float = 1.0) -> float:
    return empty_value if den == 0 else num / den


def masks_from_logits(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Threshold crack probability consistently with production inference."""

    return crack_probabilities(logits) > threshold


def filter_small_components(
    masks: torch.Tensor, minimum_pixels: int
) -> torch.Tensor:
    """Apply the same component-size filter used by tiled inference."""

    masks_bool = masks.bool()
    if minimum_pixels <= 1:
        return masks_bool
    device = masks_bool.device
    cleaned = [
        torch.from_numpy(remove_small_components(mask, minimum_pixels)[0])
        for mask in masks_bool.detach().cpu().numpy()
    ]
    return torch.stack(cleaned).to(device=device, dtype=torch.bool)


def confusion_counts(pred: torch.Tensor, target: torch.Tensor) -> ConfusionCounts:
    pred_b = pred.bool()
    target_b = target.bool()
    if pred_b.shape != target_b.shape:
        raise ValueError(
            f"pred/target shape mismatch: {tuple(pred_b.shape)} != {tuple(target_b.shape)}"
        )
    tp = int((pred_b & target_b).sum().item())
    fp = int((pred_b & ~target_b).sum().item())
    tn = int((~pred_b & ~target_b).sum().item())
    fn = int((~pred_b & target_b).sum().item())
    return ConfusionCounts(tp=tp, fp=fp, tn=tn, fn=fn)


def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask.bool()
    x = mask.float()
    if x.ndim == 2:
        x = x.unsqueeze(0)
    if x.ndim == 3:
        x = x.unsqueeze(1)
    k = 2 * radius + 1
    return (F.max_pool2d(x, kernel_size=k, stride=1, padding=radius)[:, 0] > 0)


def boundary_f1(
    pred: torch.Tensor, target: torch.Tensor, *, tolerance: int = 2
) -> dict[str, float]:
    """Boundary/centerline F1 with pixel tolerance, robust for empty masks."""

    pred_b = pred.bool()
    target_b = target.bool()
    if pred_b.shape != target_b.shape:
        raise ValueError(
            f"pred/target shape mismatch: {tuple(pred_b.shape)} != {tuple(target_b.shape)}"
        )

    pred_count = int(pred_b.sum().item())
    target_count = int(target_b.sum().item())
    if pred_count == 0 and target_count == 0:
        return {"boundary_precision": 1.0, "boundary_recall": 1.0, "boundary_f1": 1.0}
    if pred_count == 0 or target_count == 0:
        return {"boundary_precision": 0.0, "boundary_recall": 0.0, "boundary_f1": 0.0}

    target_d = _dilate(target_b, tolerance)
    pred_d = _dilate(pred_b, tolerance)
    precision = _safe_div(float((pred_b & target_d).sum().item()), float(pred_count), 0.0)
    recall = _safe_div(float((target_b & pred_d).sum().item()), float(target_count), 0.0)
    f1 = _safe_div(2.0 * precision * recall, precision + recall, 0.0)
    return {"boundary_precision": precision, "boundary_recall": recall, "boundary_f1": f1}


def compute_segmentation_metrics(
    logits_or_pred: torch.Tensor,
    labels: torch.Tensor,
    *,
    threshold: float = 0.5,
    boundary_tolerance: int = 2,
    image_level_min_pixels: int = 1,
    minimum_component_pixels: int = 1,
) -> dict[str, Any]:
    """Compute pixel, boundary, and image-level crack metrics.

    Accepts either logits ``[B,C,H,W]`` or boolean/probability masks ``[B,H,W]``.
    Empty-mask denominators are handled explicitly instead of producing NaN.
    """

    with torch.no_grad():
        if logits_or_pred.ndim == 4:
            pred = masks_from_logits(logits_or_pred, threshold=threshold)
        else:
            pred = (
                logits_or_pred > threshold
                if logits_or_pred.dtype.is_floating_point
                else logits_or_pred.bool()
            )
        target = labels[:, 0] if labels.ndim == 4 and labels.shape[1] == 1 else labels
        target = target.bool()
        pred = filter_small_components(
            pred.to(target.device), minimum_component_pixels
        )

        counts = confusion_counts(pred, target)
        tp, fp, tn, fn = counts.tp, counts.fp, counts.tn, counts.fn
        precision = _safe_div(tp, tp + fp, 1.0)
        recall = _safe_div(tp, tp + fn, 1.0)
        specificity = _safe_div(tn, tn + fp, 1.0)
        iou = _safe_div(tp, tp + fp + fn, 1.0)
        dice = _safe_div(2 * tp, 2 * tp + fp + fn, 1.0)

        pred_img = pred.flatten(1).sum(dim=1) >= image_level_min_pixels
        target_img = target.flatten(1).sum(dim=1) >= image_level_min_pixels
        img_counts = confusion_counts(pred_img, target_img)
        image_recall = _safe_div(img_counts.tp, img_counts.tp + img_counts.fn, 1.0)
        image_specificity = _safe_div(img_counts.tn, img_counts.tn + img_counts.fp, 1.0)

        bf = boundary_f1(pred, target, tolerance=boundary_tolerance)
        return {
            "crack_iou": iou,
            "crack_dice": dice,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            **bf,
            "image_level_recall": image_recall,
            "image_level_specificity": image_specificity,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "image_tp": img_counts.tp,
            "image_fp": img_counts.fp,
            "image_tn": img_counts.tn,
            "image_fn": img_counts.fn,
        }
