"""Manifest generation for the OurBrain tunnel crack dataset.

The source dataset is treated as read-only. This module only scans image and
label files, derives deterministic group-level splits, and writes optional CSV
manifests/audit dictionaries to caller-selected locations.
"""

from __future__ import annotations

import csv
import random
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from ourbrain_cv.data import mask_positive_pixels

MANIFEST_FIELDS = [
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

IMAGE_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


@dataclass(frozen=True)
class ManifestResult:
    """Manifest rows plus audit statistics from a dataset scan."""

    rows: list[dict[str, str]]
    audit: dict[str, object]


def default_image_dir(dataset_root: str | Path) -> Path:
    return Path(dataset_root) / "train" / "crack"


def default_mask_dir(dataset_root: str | Path) -> Path:
    return Path(dataset_root) / "train" / "label" / "crack"


def group_id_from_stem(stem: str) -> str:
    """Return the group id: everything before the first underscore."""

    return stem.split("_", 1)[0]


def image_stem_for_label(label_path: str | Path) -> str:
    """Map a label filename such as ``abc-L.bmp`` to image stem ``abc``."""

    stem = Path(label_path).stem
    return stem[:-2] if stem.endswith("-L") else stem


def _iter_images(directory: Path) -> Iterable[Path]:
    if not directory.exists():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _split_groups(
    group_ids: Iterable[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, str]:
    """Assign each group id to exactly one split deterministically."""

    ratios_total = train_ratio + val_ratio + test_ratio
    if ratios_total <= 0:
        raise ValueError("split ratios must sum to a positive value")

    groups = sorted(set(group_ids))
    rng = random.Random(seed)
    rng.shuffle(groups)

    n_groups = len(groups)
    train_end = round(n_groups * train_ratio / ratios_total)
    val_end = train_end + round(n_groups * val_ratio / ratios_total)

    split_by_group: dict[str, str] = {}
    for idx, group_id in enumerate(groups):
        if idx < train_end:
            split = "train"
        elif idx < val_end:
            split = "val"
        else:
            split = "test"
        split_by_group[group_id] = split
    return split_by_group


def build_manifest(
    dataset_root: str | Path,
    *,
    image_dir: str | Path | None = None,
    mask_dir: str | Path | None = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    mask_threshold: int = 127,
) -> ManifestResult:
    """Scan a dataset and return paired segmentation manifest rows and audit stats.

    Expected default layout::

        <dataset_root>/train/crack/*.bmp
        <dataset_root>/train/label/crack/*-L.bmp

    Files are never modified. Labels are paired to images by stem, stripping the
    ``-L`` suffix from label stems when present.
    """

    image_root = Path(image_dir) if image_dir is not None else default_image_dir(dataset_root)
    label_root = Path(mask_dir) if mask_dir is not None else default_mask_dir(dataset_root)

    images = list(_iter_images(image_root))
    labels = list(_iter_images(label_root))

    images_by_stem = {path.stem: path for path in images}
    labels_by_image_stem = {image_stem_for_label(path): path for path in labels}

    paired_stems = sorted(set(images_by_stem) & set(labels_by_image_stem))
    missing_label_stems = sorted(set(images_by_stem) - set(labels_by_image_stem))
    orphan_label_stems = sorted(set(labels_by_image_stem) - set(images_by_stem))

    split_by_group = _split_groups(
        (group_id_from_stem(stem) for stem in paired_stems),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    rows: list[dict[str, str]] = []
    original_mask_sizes: Counter[str] = Counter()
    positive_pixels: list[int] = []
    invalid_pairs: list[dict[str, str]] = []

    for stem in paired_stems:
        image_path = images_by_stem[stem]
        mask_path = labels_by_image_stem[stem]
        try:
            with Image.open(image_path) as image:
                width, height = image.size
            with Image.open(mask_path) as mask:
                mask_width, mask_height = mask.size
                positives = mask_positive_pixels(mask, threshold=mask_threshold)
        except (OSError, ValueError) as exc:
            invalid_pairs.append(
                {
                    "stem": stem,
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                    "error": str(exc),
                }
            )
            continue

        group_id = group_id_from_stem(stem)
        original_mask_sizes[f"{mask_width}x{mask_height}"] += 1
        positive_pixels.append(positives)
        rows.append(
            {
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "group_id": group_id,
                "split": split_by_group[group_id],
                "width": str(width),
                "height": str(height),
                "mask_width": str(mask_width),
                "mask_height": str(mask_height),
                "positive_pixels": str(positives),
                "source_kind": "paired",
            }
        )

    split_counts = Counter(row["split"] for row in rows)
    used_groups = {row["group_id"] for row in rows}
    group_split_counts = Counter(
        split for group, split in split_by_group.items() if group in used_groups
    )
    audit: dict[str, object] = {
        "image_dir": str(image_root),
        "mask_dir": str(label_root),
        "total_images": len(images),
        "total_labels": len(labels),
        "paired": len(rows),
        "missing_labels": len(missing_label_stems),
        "orphan_labels": len(orphan_label_stems),
        "missing_label_stems": missing_label_stems,
        "orphan_label_stems": orphan_label_stems,
        "groups": len(used_groups),
        "split_counts": dict(split_counts),
        "group_split_counts": dict(group_split_counts),
        "mask_size_counts": dict(original_mask_sizes),
        "positive_pixels_min": min(positive_pixels) if positive_pixels else 0,
        "positive_pixels_max": max(positive_pixels) if positive_pixels else 0,
        "positive_pixels_total": sum(positive_pixels),
        "invalid_pairs": len(invalid_pairs),
        "invalid_pair_details": invalid_pairs,
    }
    return ManifestResult(rows=rows, audit=audit)


def write_manifest(rows: list[dict[str, str]], output_csv: str | Path) -> Path:
    """Write manifest rows to CSV and return the output path."""

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in MANIFEST_FIELDS})
    return output_path


def read_manifest(manifest_csv: str | Path) -> list[dict[str, str]]:
    """Read a manifest CSV into a list of row dictionaries."""

    with Path(manifest_csv).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
