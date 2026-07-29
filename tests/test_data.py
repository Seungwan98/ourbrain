from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ourbrain_cv.data import (
    TunnelCrackSegmentationDataset,
    load_crack_mask,
    synthetic_negative_crop,
)


def _write_rgb(path: Path, size: tuple[int, int] = (8, 8)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (10, 20, 30)).save(path)


def _write_mask(path: Path, size: tuple[int, int] = (4, 4)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = Image.new("L", size, 255)
    mask.putpixel((1, 1), 0)
    mask.save(path)


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "image_path",
        "mask_path",
        "group_id",
        "split",
        "width",
        "height",
        "mask_width",
        "mask_height",
        "positive_pixels",
        "source_kind",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_load_crack_mask_inverts_black_to_positive_and_resizes_nearest(tmp_path: Path) -> None:
    mask_path = tmp_path / "mask.bmp"
    _write_mask(mask_path, size=(4, 4))

    mask = load_crack_mask(mask_path, image_size=(8, 8), threshold=127)

    assert mask.size == (8, 8)
    tensor = torch.from_numpy(np.array(mask, copy=True))
    assert int(tensor.sum()) == 4
    assert tensor[2, 2].item() == 1
    assert tensor[0, 0].item() == 0


def test_dataset_returns_paired_segmentation_and_explicit_negative(tmp_path: Path) -> None:
    image_path = tmp_path / "img.bmp"
    mask_path = tmp_path / "img-L.bmp"
    negative_path = tmp_path / "negative.bmp"
    _write_rgb(image_path, size=(8, 8))
    _write_mask(mask_path, size=(4, 4))
    _write_rgb(negative_path, size=(6, 5))
    manifest_path = tmp_path / "manifest.csv"
    _write_manifest(
        manifest_path,
        [
            {
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "group_id": "001",
                "split": "train",
                "width": "8",
                "height": "8",
                "mask_width": "4",
                "mask_height": "4",
                "positive_pixels": "1",
                "source_kind": "paired",
            },
            {
                "image_path": str(negative_path),
                "mask_path": "",
                "group_id": "neg",
                "split": "train",
                "width": "6",
                "height": "5",
                "mask_width": "0",
                "mask_height": "0",
                "positive_pixels": "0",
                "source_kind": "negative",
            },
        ],
    )

    dataset = TunnelCrackSegmentationDataset(manifest_path, split="train")

    paired = dataset[0]
    assert paired["image"].shape == (3, 8, 8)
    assert paired["pixel_values"].shape == (3, 8, 8)
    assert paired["mask"].shape == (8, 8)
    assert paired["labels"].shape == (8, 8)
    assert paired["mask"].dtype == torch.int64
    assert paired["group_id"] == "001"
    assert int(paired["mask"].sum()) == 4

    negative = dataset[1]
    assert negative["image"].shape == (3, 5, 6)
    assert negative["mask"].shape == (5, 6)
    assert int(negative["mask"].sum()) == 0


def test_dataset_can_resize_without_external_transform(tmp_path: Path) -> None:
    image_path = tmp_path / "img.bmp"
    mask_path = tmp_path / "img-L.bmp"
    _write_rgb(image_path, size=(8, 8))
    _write_mask(mask_path, size=(4, 4))
    manifest_path = tmp_path / "manifest.csv"
    _write_manifest(
        manifest_path,
        [
            {
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "group_id": "001",
                "split": "train",
                "width": "8",
                "height": "8",
                "mask_width": "4",
                "mask_height": "4",
                "positive_pixels": "1",
                "source_kind": "paired",
            },
        ],
    )

    sample = TunnelCrackSegmentationDataset(manifest_path, image_size=16)[0]

    assert sample["pixel_values"].shape == (3, 16, 16)
    assert sample["labels"].shape == (16, 16)
    assert int(sample["labels"].sum()) == 16


def test_synthetic_negative_crop_selects_region_far_from_positive_or_returns_original() -> None:
    image = Image.new("RGB", (10, 10), (0, 0, 0))
    mask = Image.new("L", (10, 10), 0)
    for x in range(0, 5):
        for y in range(0, 5):
            mask.putpixel((x, y), 1)

    cropped_image, cropped_mask = synthetic_negative_crop(
        image,
        mask,
        crop_size=3,
        min_distance=0,
        max_attempts=200,
    )

    assert cropped_image.size == (3, 3)
    assert cropped_mask.size == (3, 3)
    assert int(np.asarray(cropped_mask).sum()) == 0

    full_positive = Image.new("L", (5, 5), 1)
    original_image, original_mask = synthetic_negative_crop(
        Image.new("RGB", (5, 5)),
        full_positive,
        crop_size=3,
        min_distance=0,
        max_attempts=10,
    )
    assert original_image.size == (5, 5)
    assert original_mask is full_positive
