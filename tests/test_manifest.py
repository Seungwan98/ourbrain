from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image

from ourbrain_cv.manifest import build_manifest, group_id_from_stem, write_manifest


def _write_image(path: Path, size: tuple[int, int] = (8, 6), color: int = 255) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, color).save(path)


def _write_mask(
    path: Path, size: tuple[int, int] = (8, 6), crack_xy: tuple[int, int] = (1, 1)
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("L", size, 255)
    image.putpixel(crack_xy, 0)
    image.save(path)


def test_build_manifest_pairs_labels_by_stem_and_audits_missing_orphans(tmp_path: Path) -> None:
    image_dir = tmp_path / "train" / "crack"
    label_dir = tmp_path / "train" / "label" / "crack"
    _write_image(image_dir / "001_a.bmp")
    _write_image(image_dir / "001_b.bmp")
    _write_image(image_dir / "002_missing.bmp")
    _write_mask(label_dir / "001_a-L.bmp")
    _write_mask(label_dir / "001_b-L.bmp", crack_xy=(2, 2))
    _write_mask(label_dir / "003_orphan-L.bmp")

    result = build_manifest(tmp_path, seed=7)

    assert len(result.rows) == 2
    assert {Path(row["image_path"]).name for row in result.rows} == {"001_a.bmp", "001_b.bmp"}
    assert {Path(row["mask_path"]).name for row in result.rows} == {"001_a-L.bmp", "001_b-L.bmp"}
    assert result.audit["total_images"] == 3
    assert result.audit["total_labels"] == 3
    assert result.audit["paired"] == 2
    assert result.audit["missing_labels"] == 1
    assert result.audit["orphan_labels"] == 1
    assert result.audit["missing_label_stems"] == ["002_missing"]
    assert result.audit["orphan_label_stems"] == ["003_orphan"]
    assert all(row["positive_pixels"] == "1" for row in result.rows)


def test_group_id_uses_prefix_before_first_underscore() -> None:
    assert group_id_from_stem("ABC_001_002") == "ABC"
    assert group_id_from_stem("NO_UNDERSCORE") == "NO"
    assert group_id_from_stem("SINGLE") == "SINGLE"


def test_group_level_split_has_no_group_leakage_and_is_written(tmp_path: Path) -> None:
    for group in range(10):
        stem = f"{group:03d}_sample"
        _write_image(tmp_path / "train" / "crack" / f"{stem}.bmp")
        _write_mask(tmp_path / "train" / "label" / "crack" / f"{stem}-L.bmp")

    result = build_manifest(tmp_path, seed=123)
    split_by_group: dict[str, str] = {}
    for row in result.rows:
        previous = split_by_group.setdefault(row["group_id"], row["split"])
        assert row["split"] == previous
    assert set(split_by_group.values()) <= {"train", "val", "test"}
    assert set(split_by_group.values()) == {"train", "val", "test"}

    manifest_path = write_manifest(result.rows, tmp_path / "manifest.csv")
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(result.rows)
    assert list(rows[0]) == [
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


def test_build_manifest_records_and_skips_truncated_pairs(tmp_path: Path) -> None:
    image_dir = tmp_path / "train" / "crack"
    label_dir = tmp_path / "train" / "label" / "crack"
    _write_image(image_dir / "001_valid.bmp")
    _write_mask(label_dir / "001_valid-L.bmp")
    _write_image(image_dir / "002_broken.bmp")
    label_dir.mkdir(parents=True, exist_ok=True)
    (label_dir / "002_broken-L.bmp").write_bytes(b"not-a-valid-bitmap")

    result = build_manifest(tmp_path, seed=7)

    assert len(result.rows) == 1
    assert result.audit["invalid_pairs"] == 1
    details = result.audit["invalid_pair_details"]
    assert isinstance(details, list)
    assert details[0]["stem"] == "002_broken"
