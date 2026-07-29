import math

import torch

from ourbrain_cv.metrics import (
    boundary_f1,
    compute_segmentation_metrics,
    confusion_counts,
    masks_from_logits,
)


def test_confusion_counts_and_pixel_metrics():
    pred = torch.tensor([[[1, 0], [1, 0]]], dtype=torch.bool)
    target = torch.tensor([[[1, 1], [0, 0]]], dtype=torch.bool)
    counts = confusion_counts(pred, target)
    assert counts.as_dict() == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}

    metrics = compute_segmentation_metrics(pred, target)
    assert metrics["crack_iou"] == 1 / 3
    assert metrics["crack_dice"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["specificity"] == 0.5


def test_empty_masks_are_perfect_not_nan():
    pred = torch.zeros(2, 4, 4, dtype=torch.bool)
    target = torch.zeros(2, 4, 4, dtype=torch.bool)
    metrics = compute_segmentation_metrics(pred, target)
    for key in [
        "crack_iou",
        "crack_dice",
        "precision",
        "recall",
        "specificity",
        "boundary_f1",
        "image_level_recall",
        "image_level_specificity",
    ]:
        assert metrics[key] == 1.0


def test_boundary_f1_uses_tolerance():
    pred = torch.zeros(1, 8, 8, dtype=torch.bool)
    target = torch.zeros(1, 8, 8, dtype=torch.bool)
    pred[0, 4, 2:6] = True
    target[0, 5, 2:6] = True
    assert boundary_f1(pred, target, tolerance=1)["boundary_f1"] == 1.0
    assert boundary_f1(pred, target, tolerance=0)["boundary_f1"] == 0.0


def test_logits_metrics_include_image_confusion():
    logits = torch.zeros(2, 2, 4, 4)
    logits[0, 1, 1, 1] = 10
    labels = torch.zeros(2, 4, 4, dtype=torch.long)
    labels[0, 1, 1] = 1
    metrics = compute_segmentation_metrics(logits, labels, image_level_min_pixels=1)
    assert metrics["image_tp"] == 1
    assert metrics["image_tn"] == 1
    assert metrics["image_level_recall"] == 1.0
    assert metrics["image_level_specificity"] == 1.0


def test_two_class_logits_respect_configured_probability_threshold():
    logits = torch.zeros(1, 2, 1, 1)
    logits[:, 1] = math.log(0.55 / 0.45)

    assert masks_from_logits(logits, threshold=0.5).item() is True
    assert masks_from_logits(logits, threshold=0.9).item() is False
