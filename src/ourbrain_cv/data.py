"""Dataset and mask utilities for tunnel crack segmentation."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

try:  # Torch is a project dependency, but keep import errors clear for tooling.
    import torch
    from torch.utils.data import Dataset
except ImportError as exc:  # pragma: no cover - exercised only in broken envs
    raise RuntimeError("ourbrain_cv.data requires PyTorch to be installed") from exc


def load_image(path: str | Path) -> Image.Image:
    """Load an image as RGB without mutating the source file."""

    return Image.open(path).convert("RGB")


def load_crack_mask(
    path: str | Path,
    *,
    image_size: tuple[int, int] | None = None,
    threshold: int = 127,
) -> Image.Image:
    """Load a mask where dark pixels are crack=1 and background=0.

    Source labels encode cracks as black lines on a light background. The returned
    PIL image is mode ``L`` with values 0 or 1. If ``image_size`` is provided and
    the label size differs, nearest-neighbor resizing preserves categorical mask
    labels.
    """

    with Image.open(path) as raw:
        mask = raw.convert("L")
        if image_size is not None and mask.size != image_size:
            mask = mask.resize(image_size, Image.Resampling.NEAREST)
        array = np.asarray(mask)
    binary = (array < threshold).astype(np.uint8)
    return Image.fromarray(binary, mode="L")


def mask_positive_pixels(mask: Image.Image | str | Path, *, threshold: int = 127) -> int:
    """Count crack-positive pixels in a source mask using dark-pixel semantics."""

    if isinstance(mask, Image.Image):
        grayscale = mask.convert("L")
        array = np.asarray(grayscale)
    else:
        with Image.open(mask) as loaded:
            array = np.asarray(loaded.convert("L"))
    return int((array < threshold).sum())


def _read_manifest_rows(manifest_csv: str | Path) -> list[dict[str, str]]:
    with Path(manifest_csv).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _to_image_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _to_mask_tensor(mask: Image.Image) -> torch.Tensor:
    array = np.asarray(mask, dtype=np.int64)
    return torch.from_numpy(array).contiguous()


def _dilated_positive_mask(mask_array: np.ndarray, min_distance: int) -> np.ndarray:
    if min_distance <= 0:
        return mask_array.astype(bool)
    pil = Image.fromarray((mask_array.astype(np.uint8) * 255), mode="L")
    # MaxFilter size must be odd; radius min_distance gives approx Chebyshev gap.
    size = max(3, min_distance * 2 + 1)
    if size % 2 == 0:
        size += 1
    return np.asarray(pil.filter(ImageFilter.MaxFilter(size=size))) > 0


def _window_has_positive(
    integral: np.ndarray, left: int, top: int, width: int, height: int
) -> bool:
    right = left + width
    bottom = top + height
    total = (
        integral[bottom, right]
        - integral[top, right]
        - integral[bottom, left]
        + integral[top, left]
    )
    return bool(total > 0)


def synthetic_negative_crop(
    image: Image.Image,
    mask: Image.Image,
    *,
    crop_size: int | tuple[int, int] = 256,
    min_distance: int = 8,
    max_attempts: int = 128,
    rng: random.Random | None = None,
) -> tuple[Image.Image, Image.Image]:
    """Return a crack-free crop sufficiently far from positives, else original.

    The mask is expected to contain 0/1 categorical values. A candidate crop is
    accepted only when the crop window does not intersect a dilated positive mask.
    If no such crop is found, the original image and mask are returned unchanged.
    """

    rng = rng or random.Random()
    image_width, image_height = image.size
    if isinstance(crop_size, int):
        crop_width = crop_height = crop_size
    else:
        crop_width, crop_height = crop_size

    if crop_width <= 0 or crop_height <= 0:
        raise ValueError("crop_size must be positive")
    if crop_width > image_width or crop_height > image_height:
        return image, mask

    mask_array = np.asarray(mask, dtype=np.uint8) > 0
    avoid = _dilated_positive_mask(mask_array, min_distance).astype(np.int64)
    integral = (
        np.pad(avoid, ((1, 0), (1, 0)), mode="constant")
        .cumsum(axis=0, dtype=np.int64)
        .cumsum(axis=1, dtype=np.int64)
    )

    max_left = image_width - crop_width
    max_top = image_height - crop_height
    for _ in range(max_attempts):
        left = rng.randint(0, max_left)
        top = rng.randint(0, max_top)
        if not _window_has_positive(integral, left, top, crop_width, crop_height):
            box = (left, top, left + crop_width, top + crop_height)
            cropped_image = image.crop(box)
            negative_mask = Image.new("L", cropped_image.size, 0)
            return cropped_image, negative_mask
    return image, mask


class TunnelCrackSegmentationDataset(Dataset):
    """PyTorch dataset backed by an OurBrain segmentation manifest.

    Manifest rows with a mask path return paired segmentation samples. Rows with
    an empty mask path are treated as explicit negatives and receive all-zero
    masks. ``split`` filters rows when provided.
    """

    def __init__(
        self,
        manifest_csv: str | Path,
        *,
        split: str | None = None,
        mask_threshold: int = 127,
        transform: Any | None = None,
        return_pil: bool = False,
        image_size: int | tuple[int, int] | None = None,
        synthetic_negative_probability: float = 0.0,
        synthetic_negative_crop_size: int = 256,
        synthetic_negative_min_distance: int = 8,
        seed: int = 42,
    ) -> None:
        rows = _read_manifest_rows(manifest_csv)
        if split is not None:
            rows = [row for row in rows if row.get("split") == split]
        self.rows = rows
        self.mask_threshold = mask_threshold
        self.transform = transform
        self.return_pil = return_pil
        if isinstance(image_size, int):
            self.image_size: tuple[int, int] | None = (image_size, image_size)
        else:
            self.image_size = image_size
        self.synthetic_negative_probability = synthetic_negative_probability
        self.synthetic_negative_crop_size = synthetic_negative_crop_size
        self.synthetic_negative_min_distance = synthetic_negative_min_distance
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image = load_image(row["image_path"])
        mask_path = row.get("mask_path", "")
        if mask_path:
            mask = load_crack_mask(mask_path, image_size=image.size, threshold=self.mask_threshold)
        else:
            mask = Image.new("L", image.size, 0)

        if self.image_size is not None and image.size != self.image_size:
            image = image.resize(self.image_size, Image.Resampling.BILINEAR)
            mask = mask.resize(self.image_size, Image.Resampling.NEAREST)

        if (
            mask_path
            and self.synthetic_negative_probability > 0
            and self.rng.random() < self.synthetic_negative_probability
        ):
            image, mask = synthetic_negative_crop(
                image,
                mask,
                crop_size=self.synthetic_negative_crop_size,
                min_distance=self.synthetic_negative_min_distance,
                rng=self.rng,
            )

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            if transformed is not None:
                image = transformed.get("image", image)
                mask = transformed.get("mask", mask)

        group = row.get("group", row.get("group_id", row.get("source_id", str(index))))
        sample: dict[str, Any] = {
            "metadata": row,
            "image_path": row["image_path"],
            "mask_path": mask_path,
            "group_id": group,
            "group": group,
        }
        if self.return_pil:
            sample["image"] = image
            sample["mask"] = mask
            sample["pixel_values"] = image
            sample["labels"] = mask
        else:
            image_tensor = image if torch.is_tensor(image) else _to_image_tensor(image)
            mask_tensor = mask if torch.is_tensor(mask) else _to_mask_tensor(mask)
            sample["image"] = image_tensor
            sample["mask"] = mask_tensor
            sample["pixel_values"] = image_tensor
            sample["labels"] = mask_tensor
        return sample
