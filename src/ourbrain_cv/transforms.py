"""Joint image/mask transforms for thin crack segmentation."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageStat
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
    rotation_degrees: float = 0.0
    gamma_jitter: float = 0.0
    gaussian_blur_probability: float = 0.0
    gaussian_blur_radius: float = 0.0
    gaussian_noise_probability: float = 0.0
    gaussian_noise_std: float = 0.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        for name in (
            "horizontal_flip_probability",
            "vertical_flip_probability",
            "gaussian_blur_probability",
            "gaussian_noise_probability",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        for name in (
            "brightness_jitter",
            "contrast_jitter",
            "rotation_degrees",
            "gamma_jitter",
            "gaussian_blur_radius",
            "gaussian_noise_std",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.gamma_jitter >= 1.0:
            raise ValueError("gamma_jitter must be less than 1")
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

            if self.rotation_degrees > 0.0:
                angle = self._rng.uniform(-self.rotation_degrees, self.rotation_degrees)
                fill = tuple(round(value) for value in ImageStat.Stat(image).mean[:3])
                image = image.rotate(
                    angle,
                    resample=Image.Resampling.BILINEAR,
                    fillcolor=fill,
                )
                mask = mask.rotate(
                    angle,
                    resample=Image.Resampling.NEAREST,
                    fillcolor=0,
                )

            brightness = 1.0 + self._rng.uniform(-self.brightness_jitter, self.brightness_jitter)
            contrast = 1.0 + self._rng.uniform(-self.contrast_jitter, self.contrast_jitter)
            image = ImageEnhance.Brightness(image).enhance(brightness)
            image = ImageEnhance.Contrast(image).enhance(contrast)

            if self.gamma_jitter > 0.0:
                gamma = self._rng.uniform(
                    1.0 - self.gamma_jitter,
                    1.0 + self.gamma_jitter,
                )
                gamma_table = [
                    round(255.0 * ((value / 255.0) ** gamma))
                    for value in range(256)
                ]
                image = image.point(gamma_table * 3)

            if (
                self.gaussian_blur_radius > 0.0
                and self._rng.random() < self.gaussian_blur_probability
            ):
                radius = self._rng.uniform(0.1, self.gaussian_blur_radius)
                image = image.filter(ImageFilter.GaussianBlur(radius=radius))

            if (
                self.gaussian_noise_std > 0.0
                and self._rng.random() < self.gaussian_noise_probability
            ):
                array = np.asarray(image, dtype=np.float32)
                noise_rng = np.random.default_rng(self._rng.getrandbits(64))
                noise = noise_rng.normal(
                    0.0,
                    self.gaussian_noise_std * 255.0,
                    size=array.shape,
                )
                image = Image.fromarray(
                    np.clip(array + noise, 0.0, 255.0).astype(np.uint8),
                    mode="RGB",
                )

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
