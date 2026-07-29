from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from ourbrain_cv.evaluation import evaluate_dataset


class IdentityLogitModel(nn.Module):
    def forward(self, pixel_values, labels=None):
        crack = pixel_values[:, :1]
        background = 1.0 - crack
        return SimpleNamespace(logits=torch.cat([background, crack], dim=1) * 10)


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
    dataset = [{"pixel_values": pixels, "labels": labels}]

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
    assert filtered["fp"] == 0
    assert filtered["image_level_specificity"] == 1.0
