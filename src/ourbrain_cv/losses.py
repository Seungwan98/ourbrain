"""Loss functions for very sparse tunnel crack masks."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


def _binary_targets(labels: torch.Tensor) -> torch.Tensor:
    if labels.ndim == 4 and labels.shape[1] == 1:
        labels = labels[:, 0]
    if labels.ndim != 3:
        raise ValueError(f"labels must have shape [B,H,W] or [B,1,H,W], got {tuple(labels.shape)}")
    return (labels > 0).float()


def crack_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Return crack probabilities from binary or two-class logits."""

    if logits.ndim != 4:
        raise ValueError(f"logits must have shape [B,C,H,W], got {tuple(logits.shape)}")
    if logits.shape[1] == 1:
        return torch.sigmoid(logits[:, 0])
    if logits.shape[1] >= 2:
        return torch.softmax(logits, dim=1)[:, 1]
    raise ValueError("logits channel dimension must be >= 1")


def binary_focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    alpha: float = 0.75,
    gamma: float = 2.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Binary focal loss over the crack channel with 1- or 2-class logits."""

    targets = _binary_targets(labels).to(logits.device)
    probs = crack_probabilities(logits).clamp(eps, 1.0 - eps)
    ce = F.binary_cross_entropy(probs, targets, reduction="none")
    p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return (alpha_t * (1.0 - p_t).pow(gamma) * ce).mean()


def soft_dice_loss(
    logits: torch.Tensor, labels: torch.Tensor, *, smooth: float = 1.0
) -> torch.Tensor:
    """Soft Dice loss for sparse crack pixels; safe for empty masks."""

    targets = _binary_targets(labels).to(logits.device)
    probs = crack_probabilities(logits)
    dims = tuple(range(1, probs.ndim))
    intersection = (probs * targets).sum(dim=dims)
    denominator = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - dice.mean()


def _gradient_map(x: torch.Tensor) -> torch.Tensor:
    """Differentiable morphology-like edge map using finite differences/max pooling."""

    if x.ndim == 3:
        x = x.unsqueeze(1)
    dx = F.pad((x[:, :, :, 1:] - x[:, :, :, :-1]).abs(), (0, 1, 0, 0))
    dy = F.pad((x[:, :, 1:, :] - x[:, :, :-1, :]).abs(), (0, 0, 0, 1))
    grad = torch.clamp(dx + dy, 0.0, 1.0)
    return F.max_pool2d(grad, kernel_size=3, stride=1, padding=1)[:, 0]


def boundary_loss(
    logits: torch.Tensor, labels: torch.Tensor, *, smooth: float = 1.0
) -> torch.Tensor:
    """Dice-style boundary loss on differentiable prediction/target gradients."""

    targets = _binary_targets(labels).to(logits.device)
    probs = crack_probabilities(logits)
    pred_edges = _gradient_map(probs)
    target_edges = _gradient_map(targets)
    dims = tuple(range(1, pred_edges.ndim))
    intersection = (pred_edges * target_edges).sum(dim=dims)
    denominator = pred_edges.sum(dim=dims) + target_edges.sum(dim=dims)
    score = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - score.mean()


@dataclass(frozen=True)
class LossWeights:
    focal: float = 1.0
    dice: float = 1.0
    boundary: float = 0.25
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0


class CrackSegmentationLoss(nn.Module):
    """Weighted focal + Dice + boundary objective for thin cracks."""

    def __init__(self, weights: LossWeights | None = None) -> None:
        super().__init__()
        self.weights = weights or LossWeights()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        parts = self.components(logits, labels)
        return (
            self.weights.focal * parts["focal"]
            + self.weights.dice * parts["dice"]
            + self.weights.boundary * parts["boundary"]
        )

    def components(self, logits: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "focal": binary_focal_loss(
                logits, labels, alpha=self.weights.focal_alpha, gamma=self.weights.focal_gamma
            ),
            "dice": soft_dice_loss(logits, labels),
            "boundary": boundary_loss(logits, labels),
        }
