"""Joint image/mask transforms for thin crack segmentation."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from PIL import Image, ImageEnhance, ImageOps
from torchvision.transforms import functional as F


@dataclass
class JointSegmentationTransform:
    """Resize and jointly augment a PIL image/mask pair.

    Geometric operations are applied identically to the image and mask.
    Photometric operations only affect the RGB image. Masks remain categorical.
    """

    image_size: int = 512
    train: bool = False
    horizontal_flip_probability: float = 0.5
    vertical_flip_probability: float = 0.25
    brightness_jitter: float = 0.15
    contrast_jitter: float = 0.15
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        self._rng = random.Random(self.seed)

    def __call__(self, *, image: Image.Image, mask: Image.Image) -> dict[str, torch.Tensor]:
        image = image.convert("RGB")
        mask = mask.convert("L")

        if self.train:
            if self._rng.random() < self.horizontal_flip_probability:
                image = ImageOps.mirror(image)
                mask = ImageOps.mirror(mask)
            if self._rng.random() < self.vertical_flip_probability:
                image = ImageOps.flip(image)
                mask = ImageOps.flip(mask)

            brightness = 1.0 + self._rng.uniform(-self.brightness_jitter, self.brightness_jitter)
            contrast = 1.0 + self._rng.uniform(-self.contrast_jitter, self.contrast_jitter)
            image = ImageEnhance.Brightness(image).enhance(brightness)
            image = ImageEnhance.Contrast(image).enhance(contrast)

        size = [self.image_size, self.image_size]
        image_tensor = F.resize(
            F.pil_to_tensor(image).float() / 255.0,
            size,
            interpolation=F.InterpolationMode.BILINEAR,
            antialias=True,
        )
        image_tensor = F.normalize(
            image_tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        mask_tensor = F.resize(
            F.pil_to_tensor(mask),
            size,
            interpolation=F.InterpolationMode.NEAREST,
            antialias=False,
        )[0]
        mask_tensor = (mask_tensor > 0).long()
        return {"image": image_tensor.contiguous(), "mask": mask_tensor.contiguous()}

