from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from ourbrain_cv.evaluation import calibrate_threshold, evaluate_dataset


class IdentityLogitModel(nn.Module):
    def forward(self, pixel_values, labels=None):
        crack = pixel_values[:, :1]
        background = 1.0 - crack
        return SimpleNamespace(logits=torch.cat([background, crack], dim=1) * 10)


class ProbabilityLogitModel(nn.Module):
    def forward(self, pixel_values, labels=None):
        probability = pixel_values[:, :1].clamp(1e-4, 1 - 1e-4)
        crack = torch.logit(probability)
        return SimpleNamespace(logits=torch.cat([torch.zeros_like(crack), crack], dim=1))


def test_evaluate_dataset_aggregates_positive_and_negative_samples(tmp_path) -> None:
    positive_labels = torch.zeros(4, 4, dtype=torch.long)
    positive_labels[1, 1] = 1
    positive_pixels = positive_labels.float().unsqueeze(0).repeat(3, 1, 1)
    negative_labels = torch.zeros(4, 4, dtype=torch.long)
    negative_pixels = torch.zeros(3, 4, 4)
    dataset = [
        {"pixel_values": positive_pixels, "labels": positive_labels},
        {"pixel_values": negative_pixels, "labels": negative_labels},
    ]

    result = evaluate_dataset(
        IdentityLogitModel(),
        dataset,
        image_level_minimum_pixels=1,
        output_json=tmp_path / "metrics.json",
    )

    assert result["samples"] == 2
    assert result["crack_dice"] == 1.0
    assert result["image_level_recall"] == 1.0
    assert result["image_level_specificity"] == 1.0
    assert (tmp_path / "metrics.json").exists()


def test_evaluate_dataset_uses_inference_component_filter() -> None:
    labels = torch.zeros(4, 4, dtype=torch.long)
    pixels = torch.zeros(3, 4, 4)
    pixels[:, 1, 1] = 1
    dataset = [
        {
            "pixel_values": pixels,
            "labels": labels,
            "image_path": "normal-hard-negative.png",
            "group_id": "normal-001",
            "metadata": {
                "split": "test",
                "source_kind": "reviewed_negative",
            },
        }
    ]

    unfiltered = evaluate_dataset(
        IdentityLogitModel(),
        dataset,
        image_level_minimum_pixels=1,
        minimum_component_pixels=1,
    )
    filtered = evaluate_dataset(
        IdentityLogitModel(),
        dataset,
        image_level_minimum_pixels=1,
        minimum_component_pixels=2,
    )

    assert unfiltered["fp"] == 1
    assert unfiltered["image_level_specificity"] == 0.0
    assert unfiltered["error_case_count"] == 1
    assert unfiltered["error_cases"][0] == {
        "sample_index": 0,
        "image_path": "normal-hard-negative.png",
        "mask_path": None,
        "group_id": "normal-001",
        "split": "test",
        "source_kind": "reviewed_negative",
        "error_types": ["image_false_positive"],
        "false_positive_pixels": 1,
        "false_negative_pixels": 0,
        "crack_dice": 0.0,
        "boundary_f1": 0.0,
    }
    assert filtered["fp"] == 0
    assert filtered["image_level_specificity"] == 1.0
    assert filtered["error_case_count"] == 0


def test_calibrate_threshold_preserves_required_image_recall(tmp_path) -> None:
    def sample(probability: float, positive: bool):
        pixels = torch.zeros(3, 4, 4)
        pixels[:, 1, 1] = probability
        labels = torch.zeros(4, 4, dtype=torch.long)
        if positive:
            labels[1, 1] = 1
        return {"pixel_values": pixels, "labels": labels}

    dataset = [
        sample(0.8, True),
        sample(0.6, True),
        sample(0.55, False),
        sample(0.2, False),
    ]
    result = calibrate_threshold(
        ProbabilityLogitModel(),
        dataset,
        thresholds=[0.5, 0.7],
        minimum_image_recall=0.9,
        image_level_minimum_pixels=1,
        minimum_component_pixels=1,
        output_json=tmp_path / "calibration.json",
    )

    assert result["selected_threshold"] == 0.5
    assert result["recall_constraint_met"] is True
    assert result["selected_metrics"]["image_level_recall"] == 1.0
    assert result["selected_metrics"]["image_level_specificity"] == 0.5
    assert (tmp_path / "calibration.json").exists()


def test_calibrate_threshold_rejects_empty_or_single_class_validation() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        calibrate_threshold(
            ProbabilityLogitModel(),
            [],
            thresholds=[float("nan")],
        )

    with pytest.raises(ValueError, match="at least one sample"):
        calibrate_threshold(
            ProbabilityLogitModel(),
            [],
            thresholds=[0.5],
        )

    negative = {
        "pixel_values": torch.zeros(3, 4, 4),
        "labels": torch.zeros(4, 4, dtype=torch.long),
    }
    with pytest.raises(ValueError, match="both crack and reviewed-negative"):
        calibrate_threshold(
            ProbabilityLogitModel(),
            [negative],
            thresholds=[0.5],
            image_level_minimum_pixels=1,
        )
