from __future__ import annotations

import numpy as np
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

