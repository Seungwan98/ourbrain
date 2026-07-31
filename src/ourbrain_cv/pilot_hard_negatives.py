"""Build re-reviewable hard-negative crops from human-reviewed pilot errors."""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image

from ourbrain_cv.manifest import group_id_from_stem
from ourbrain_cv.reviews import file_sha256

PILOT_REVIEW_LABELS = {
    "correct_crack",
    "correct_normal",
    "false_positive",
    "false_negative",
    "uncertain",
}
OUTPUT_FIELDS = (
    "candidate_path",
    "source_image_path",
    "group_id",
    "left",
    "top",
    "right",
    "bottom",
    "review_label",
    "pilot_review_label",
    "pilot_error_category",
    "pilot_note",
)


def _integer_coordinate(row: dict[str, str], field: str, row_number: int) -> int:
    value = row.get(field, "").strip()
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(
            f"pilot review row {row_number} has invalid {field}: {value!r}"
        ) from exc


def _centered_crop_box(
    image_size: tuple[int, int],
    error_box: tuple[int, int, int, int],
    tile_size: int,
) -> tuple[int, int, int, int]:
    width, height = image_size
    left, top, right, bottom = error_box
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError(
            f"false-positive coordinates {error_box} are outside image {image_size}"
        )
    crop_width = min(tile_size, width)
    crop_height = min(tile_size, height)
    center_x = (left + right) // 2
    center_y = (top + bottom) // 2
    crop_left = min(max(center_x - crop_width // 2, 0), width - crop_width)
    crop_top = min(max(center_y - crop_height // 2, 0), height - crop_height)
    return (
        crop_left,
        crop_top,
        crop_left + crop_width,
        crop_top + crop_height,
    )


def build_pilot_hard_negative_review(
    pilot_review_csv: str | Path,
    output_dir: str | Path,
    *,
    tile_size: int = 512,
) -> dict[str, Any]:
    """Crop confirmed pilot false positives into a second human-review batch."""

    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    review_path = Path(pilot_review_csv).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    if not review_path.is_file():
        raise FileNotFoundError(f"pilot review CSV does not exist: {review_path}")
    if output_path.exists():
        raise FileExistsError(
            f"hard-negative output already exists; archive it before rerunning: {output_path}"
        )
    with review_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        rows = list(reader)
    required = {"input", "review_label", "left", "top", "right", "bottom"}
    missing = sorted(required - fields)
    if missing:
        raise ValueError(
            f"pilot review CSV is missing fields: {', '.join(missing)}"
        )
    if not rows:
        raise ValueError("pilot review CSV has no rows")

    invalid_labels: list[tuple[int, str]] = []
    for row_number, row in enumerate(rows, start=2):
        label = row.get("review_label", "").strip().lower()
        if label not in PILOT_REVIEW_LABELS:
            invalid_labels.append((row_number, label))
    if invalid_labels:
        raise ValueError(
            "pilot review must be complete before hard-negative extraction; "
            f"invalid_or_blank_labels={invalid_labels[:20]}"
        )

    false_positive_rows = [
        (row_number, row)
        for row_number, row in enumerate(rows, start=2)
        if row.get("review_label", "").strip().lower() == "false_positive"
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    crop_records: list[dict[str, str]] = []
    crop_hashes: dict[str, str] = {}
    seen_identity: set[tuple[str, tuple[int, int, int, int]]] = set()

    with tempfile.TemporaryDirectory(
        dir=output_path.parent,
        prefix=f".{output_path.name}-",
    ) as temporary:
        temporary_path = Path(temporary)
        for crop_index, (row_number, row) in enumerate(false_positive_rows):
            source = Path(row.get("input", "")).expanduser().resolve()
            if not source.is_file():
                raise FileNotFoundError(
                    f"pilot review row {row_number} source image is missing: {source}"
                )
            error_box = tuple(
                _integer_coordinate(row, field, row_number)
                for field in ("left", "top", "right", "bottom")
            )
            with Image.open(source) as image:
                crop_box = _centered_crop_box(image.size, error_box, tile_size)
                identity = (str(source), crop_box)
                if identity in seen_identity:
                    raise ValueError(
                        f"duplicate hard-negative crop at pilot row {row_number}: "
                        f"{source} {crop_box}"
                    )
                seen_identity.add(identity)
                crop = image.convert("RGB").crop(crop_box)
                file_name = (
                    f"{source.stem}_{crop_box[1]:06d}_{crop_box[0]:06d}"
                    f"_pilot_neg_{crop_index:05d}.png"
                )
                temporary_crop = temporary_path / file_name
                crop.save(temporary_crop)

            final_crop = output_path / file_name
            crop_records.append(
                {
                    "candidate_path": str(final_crop),
                    "source_image_path": str(source),
                    "group_id": group_id_from_stem(source.stem),
                    "left": str(crop_box[0]),
                    "top": str(crop_box[1]),
                    "right": str(crop_box[2]),
                    "bottom": str(crop_box[3]),
                    "review_label": "",
                    "pilot_review_label": "false_positive",
                    "pilot_error_category": row.get("error_category", "").strip(),
                    "pilot_note": row.get("note", "").strip(),
                }
            )
            crop_hashes[file_name] = file_sha256(temporary_crop)

        review_output = temporary_path / "hard_negative_review.csv"
        with review_output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()
            writer.writerows(crop_records)
        metadata = {
            "schema_version": 1,
            "status": (
                "human_re_review_required"
                if crop_records
                else "no_false_positive_crops"
            ),
            "pilot_review_csv": str(review_path),
            "pilot_review_sha256": file_sha256(review_path),
            "pilot_rows": len(rows),
            "false_positive_rows": len(false_positive_rows),
            "crop_count": len(crop_records),
            "tile_size": tile_size,
            "review_csv": str(output_path / review_output.name),
            "crop_sha256": crop_hashes,
            "accepted_import_labels_after_re_review": ["negative", "0"],
        }
        (temporary_path / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(output_path)
    return metadata


__all__ = [
    "PILOT_REVIEW_LABELS",
    "build_pilot_hard_negative_review",
]
