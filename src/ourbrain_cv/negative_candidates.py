"""Utilities for proposing human-review negative tunnel patches.

These helpers operate on copied/exported raw images and write review artifacts to
caller-selected output paths. They never modify the source dataset.
"""

from __future__ import annotations

import csv
import random
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from ourbrain_cv.image_io import crop_trusted_image, open_trusted_large_image

IMAGE_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


@dataclass(frozen=True)
class Box:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


@dataclass(frozen=True)
class Candidate:
    group_id: str
    image_path: Path
    candidate_path: Path | None
    box: Box


_GRID_RE = re.compile(r"^(?P<group>[^_]+)_(?P<x>\d+)_(?P<y>\d+)(?:-.+)?$")


def parse_patch_box(stem: str, *, tile_size: int = 512) -> tuple[str, Box]:
    """Parse patch stems into source group id and absolute box.

    Two coordinate forms are supported after the group prefix:
    - small values are interpreted as grid indices multiplied by ``tile_size``;
    - larger values are interpreted as absolute top/left pixels. Historical
      OurBrain stems use ``<group>_<top>_<left>-suffix`` for absolute patches.
    """

    match = _GRID_RE.match(stem)
    if not match:
        raise ValueError(f"cannot parse patch stem: {stem}")
    group = match.group("group")
    first_token = match.group("x")
    second_token = match.group("y")
    first = int(first_token)
    second = int(second_token)

    if len(first_token) <= 3 and len(second_token) <= 3:
        left = second * tile_size
        top = first * tile_size
    else:
        top = first
        left = second
    return group, Box(left, top, left + tile_size, top + tile_size)


def boxes_intersect(left: Box, right: Box, *, margin: int = 0) -> bool:
    """Return whether boxes intersect after expanding both by ``margin``."""

    return not (
        left.right + margin <= right.left - margin
        or right.right + margin <= left.left - margin
        or left.bottom + margin <= right.top - margin
        or right.bottom + margin <= left.top - margin
    )


def _iter_raw_images(raw_root: Path) -> list[Path]:
    return sorted(
        path
        for path in raw_root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def collect_candidates(
    raw_root: str | Path,
    positives_by_group: dict[str, list[Box]],
    *,
    tile_size: int = 512,
    stride: int = 512,
    exclusion_margin: int = 0,
    max_candidates: int | None = None,
    max_raw_images: int | None = None,
    seed: int | None = None,
) -> list[Candidate]:
    """Collect raw-image tiles that do not overlap known positive boxes."""

    raw_path = Path(raw_root)
    candidates: list[Candidate] = []
    rng = random.Random(seed)
    seen = 0
    image_paths = _iter_raw_images(raw_path)
    if max_raw_images is not None:
        if max_raw_images <= 0:
            return []
        image_paths = image_paths[:max_raw_images]
    for image_path in image_paths:
        group_id = image_path.stem.split("_", 1)[0]
        positives = positives_by_group.get(group_id, [])
        with open_trusted_large_image(image_path) as image:
            width, height = image.size
        for top in range(0, max(0, height - tile_size) + 1, stride):
            for left in range(0, max(0, width - tile_size) + 1, stride):
                box = Box(left, top, left + tile_size, top + tile_size)
                if any(
                    boxes_intersect(box, positive, margin=exclusion_margin)
                    for positive in positives
                ):
                    continue
                candidate = Candidate(group_id, image_path, None, box)
                if max_candidates is None:
                    candidates.append(candidate)
                    continue
                if max_candidates <= 0:
                    return []
                seen += 1
                if len(candidates) < max_candidates:
                    candidates.append(candidate)
                else:
                    replacement = rng.randrange(seen)
                    if replacement < max_candidates:
                        candidates[replacement] = candidate
    return candidates


def _positives_from_manifest(manifest_path: str | Path, *, tile_size: int) -> dict[str, list[Box]]:
    positives: dict[str, list[Box]] = {}
    with Path(manifest_path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            image_path = Path(row["image_path"])
            try:
                group, box = parse_patch_box(image_path.stem, tile_size=tile_size)
            except ValueError:
                continue
            positives.setdefault(group, []).append(box)
    return positives


def _write_contact_sheets(
    candidate_paths: list[Path],
    output_path: Path,
    *,
    columns: int = 4,
    rows: int = 4,
    thumbnail_size: int = 192,
) -> None:
    per_sheet = columns * rows
    label_height = 22
    for offset in range(0, len(candidate_paths), per_sheet):
        sheet_paths = candidate_paths[offset : offset + per_sheet]
        sheet = Image.new(
            "RGB",
            (columns * thumbnail_size, rows * (thumbnail_size + label_height)),
            "white",
        )
        draw = ImageDraw.Draw(sheet)
        for index, path in enumerate(sheet_paths):
            with Image.open(path) as image:
                thumbnail = image.convert("RGB")
                thumbnail.thumbnail((thumbnail_size, thumbnail_size))
            left = (index % columns) * thumbnail_size
            top = (index // columns) * (thumbnail_size + label_height)
            sheet.paste(thumbnail, (left, top))
            draw.text((left + 3, top + thumbnail_size + 3), path.stem, fill="black")
        sheet.save(output_path / f"contact_sheet_{offset // per_sheet:03d}.jpg", quality=90)


def generate_review_set(
    *,
    raw_root: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    tile_size: int = 512,
    stride: int = 512,
    exclusion_margin: int = 0,
    max_candidates: int | None = None,
    max_raw_images: int | None = None,
    seed: int | None = None,
) -> Path:
    """Write candidate negative crops plus a CSV with blank review labels."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    positives = _positives_from_manifest(manifest_path, tile_size=tile_size)
    candidates = collect_candidates(
        raw_root,
        positives,
        tile_size=tile_size,
        stride=stride,
        exclusion_margin=exclusion_margin,
        max_candidates=max_candidates,
        max_raw_images=max_raw_images,
        seed=seed,
    )

    rows: list[dict[str, str]] = []
    candidate_paths: list[Path] = []
    candidates_by_image: dict[Path, list[tuple[int, Candidate]]] = {}
    for idx, candidate in enumerate(candidates):
        candidates_by_image.setdefault(candidate.image_path, []).append((idx, candidate))

    for image_path, image_candidates in candidates_by_image.items():
        for idx, candidate in image_candidates:
            name = (
                f"{candidate.group_id}_{candidate.box.top:06d}_{candidate.box.left:06d}"
                f"_neg_{idx:05d}.png"
            )
            candidate_path = output_path / name
            crop_trusted_image(
                image_path,
                (
                    candidate.box.left,
                    candidate.box.top,
                    candidate.box.right,
                    candidate.box.bottom,
                )
            ).save(candidate_path)
            candidate_paths.append(candidate_path)
            rows.append(
                {
                    "candidate_path": str(candidate_path),
                    "source_image_path": str(candidate.image_path),
                    "group_id": candidate.group_id,
                    "left": str(candidate.box.left),
                    "top": str(candidate.box.top),
                    "right": str(candidate.box.right),
                    "bottom": str(candidate.box.bottom),
                    "review_label": "",
                }
            )

    review_csv = output_path / "negative_review.csv"
    with review_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "candidate_path",
            "source_image_path",
            "group_id",
            "left",
            "top",
            "right",
            "bottom",
            "review_label",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _write_contact_sheets(candidate_paths, output_path)
    return review_csv
