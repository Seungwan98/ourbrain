from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image

from ourbrain_cv.negative_candidates import (
    Box,
    boxes_intersect,
    collect_candidates,
    generate_review_set,
    parse_patch_box,
)


def test_parse_patch_box_supports_grid_and_absolute_coordinates() -> None:
    group, grid = parse_patch_box("0030_002_008", tile_size=512)  # type: ignore[misc]
    assert group == "0030"
    assert grid == Box(4096, 1024, 4608, 1536)

    group, absolute = parse_patch_box("0006_11264_00512-c", tile_size=512)  # type: ignore[misc]
    assert group == "0006"
    assert absolute == Box(512, 11264, 1024, 11776)

    _, absolute_small = parse_patch_box("0006_00512_00512-c", tile_size=512)
    assert absolute_small == Box(512, 512, 1024, 1024)


def test_box_intersection_respects_margin() -> None:
    left = Box(0, 0, 100, 100)
    right = Box(110, 0, 210, 100)
    assert not boxes_intersect(left, right)
    assert boxes_intersect(left, right, margin=11)


def test_collect_candidates_excludes_known_positive_region(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    Image.new("RGB", (1024, 512), "gray").save(raw_root / "0001.png")

    candidates = collect_candidates(
        raw_root,
        {"0001": [Box(0, 0, 512, 512)]},
        tile_size=512,
        stride=512,
        exclusion_margin=0,
        max_candidates=10,
    )
    assert [candidate.box for candidate in candidates] == [Box(512, 0, 1024, 512)]


def test_generate_review_set_writes_blank_review_labels(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    Image.new("RGB", (1024, 512), "gray").save(raw_root / "0001.png")
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path"])
        writer.writeheader()
        writer.writerow({"image_path": "/source/0001_000_000.png"})

    review_path = generate_review_set(
        raw_root=raw_root,
        manifest_path=manifest,
        output_dir=tmp_path / "review",
        tile_size=512,
        stride=512,
        exclusion_margin=0,
        max_candidates=10,
    )

    with review_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["review_label"] == ""
    assert Path(rows[0]["candidate_path"]).exists()
    assert (tmp_path / "review" / "contact_sheet_000.jpg").exists()
