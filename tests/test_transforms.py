from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from ourbrain_cv.transforms import JointSegmentationTransform


def test_transform_resizes_and_keeps_mask_categorical() -> None:
    image = Image.new("RGB", (16, 8), "gray")
    mask_array = np.zeros((8, 16), dtype=np.uint8)
    mask_array[:, 7:9] = 1
    mask = Image.fromarray(mask_array, mode="L")

    result = JointSegmentationTransform(image_size=32)(image=image, mask=mask)

    assert result["image"].shape == (3, 32, 32)
    assert result["mask"].shape == (32, 32)
    assert set(torch.unique(result["mask"]).tolist()) == {0, 1}


def test_transform_applies_forced_flip_to_image_and_mask() -> None:
    image_array = np.zeros((4, 4, 3), dtype=np.uint8)
    image_array[:, 0] = 255
    mask_array = np.zeros((4, 4), dtype=np.uint8)
    mask_array[:, 0] = 1

    result = JointSegmentationTransform(
        image_size=4,
        train=True,
        horizontal_flip_probability=1.0,
        vertical_flip_probability=0.0,
        brightness_jitter=0.0,
        contrast_jitter=0.0,
        seed=1,
    )(
        image=Image.fromarray(image_array),
        mask=Image.fromarray(mask_array, mode="L"),
    )

    assert result["mask"][:, -1].all()
    assert not result["mask"][:, 0].any()


def test_transform_rotation_keeps_image_and_mask_spatially_aligned() -> None:
    image_array = np.zeros((32, 32, 3), dtype=np.uint8)
    image_array[10:22, 14:18] = 255
    mask_array = np.zeros((32, 32), dtype=np.uint8)
    mask_array[10:22, 14:18] = 1

    result = JointSegmentationTransform(
        image_size=32,
        train=True,
        horizontal_flip_probability=0.0,
        vertical_flip_probability=0.0,
        brightness_jitter=0.0,
        contrast_jitter=0.0,
        rotation_degrees=30.0,
        seed=3,
    )(
        image=Image.fromarray(image_array),
        mask=Image.fromarray(mask_array, mode="L"),
    )

    positive = result["mask"].bool()
    negative = ~positive
    assert positive.any()
    assert result["image"][0][positive].mean() > result["image"][0][negative].mean()


def test_advanced_photometric_augmentation_is_seeded_and_keeps_mask() -> None:
    image_array = np.full((24, 24, 3), 128, dtype=np.uint8)
    image_array[:, ::3, 0] = 220
    mask_array = np.zeros((24, 24), dtype=np.uint8)
    mask_array[8:16, 8:16] = 1
    image = Image.fromarray(image_array)
    mask = Image.fromarray(mask_array, mode="L")
    kwargs = {
        "image_size": 24,
        "train": True,
        "horizontal_flip_probability": 0.0,
        "vertical_flip_probability": 0.0,
        "brightness_jitter": 0.2,
        "contrast_jitter": 0.2,
        "gamma_jitter": 0.15,
        "gaussian_blur_probability": 1.0,
        "gaussian_blur_radius": 1.0,
        "gaussian_noise_probability": 1.0,
        "gaussian_noise_std": 0.015,
        "seed": 42,
    }

    first = JointSegmentationTransform(**kwargs)(image=image, mask=mask)
    second = JointSegmentationTransform(**kwargs)(image=image, mask=mask)
    baseline = JointSegmentationTransform(image_size=24)(image=image, mask=mask)

    torch.testing.assert_close(first["image"], second["image"])
    torch.testing.assert_close(first["mask"], second["mask"])
    torch.testing.assert_close(first["mask"], baseline["mask"])
    assert not torch.equal(first["image"], baseline["image"])
    assert set(torch.unique(first["mask"]).tolist()) == {0, 1}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("horizontal_flip_probability", 1.1),
        ("gaussian_noise_probability", -0.1),
        ("rotation_degrees", -1.0),
        ("gamma_jitter", 1.0),
        ("gaussian_blur_radius", -0.1),
        ("gaussian_noise_std", -0.1),
    ],
)
def test_transform_rejects_invalid_augmentation_values(
    field: str,
    value: float,
) -> None:
    with pytest.raises(ValueError):
        JointSegmentationTransform(**{field: value})
