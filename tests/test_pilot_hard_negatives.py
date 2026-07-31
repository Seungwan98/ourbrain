from __future__ import annotations

import csv
from pathlib import Path

import pytest
from PIL import Image

from ourbrain_cv.pilot_hard_negatives import build_pilot_hard_negative_review

FIELDS = (
    "input",
    "overlay",
    "model_presence",
    "quality_gate_passed",
    "review_label",
    "left",
    "top",
    "right",
    "bottom",
    "error_category",
    "note",
)


def _write_review(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_build_pilot_hard_negatives_crops_only_false_positives(tmp_path: Path):
    source = tmp_path / "0042_scan.bmp"
    Image.new("RGB", (1024, 768), "gray").save(source)
    review = tmp_path / "pilot_review.csv"
    _write_review(
        review,
        [
            {
                "input": str(source),
                "review_label": "false_positive",
                "left": "700",
                "top": "300",
                "right": "740",
                "bottom": "340",
                "error_category": "cable",
                "note": "케이블을 균열로 오인",
            },
            {
                "input": str(source),
                "review_label": "correct_normal",
                "left": "",
                "top": "",
                "right": "",
                "bottom": "",
            },
        ],
    )
    output = tmp_path / "hard-negatives"

    result = build_pilot_hard_negative_review(review, output)

    assert result["status"] == "human_re_review_required"
    assert result["crop_count"] == 1
    with (output / "hard_negative_review.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["review_label"] == ""
    assert rows[0]["pilot_review_label"] == "false_positive"
    assert rows[0]["pilot_error_category"] == "cable"
    assert rows[0]["group_id"] == "0042"
    crop = Path(rows[0]["candidate_path"])
    assert crop.parent == output
    with Image.open(crop) as image:
        assert image.size == (512, 512)
    assert result["crop_sha256"][crop.name]


def test_build_pilot_hard_negatives_requires_complete_human_review(
    tmp_path: Path,
):
    review = tmp_path / "pilot_review.csv"
    _write_review(
        review,
        [
            {
                "input": str(tmp_path / "scan.bmp"),
                "review_label": "",
                "left": "",
                "top": "",
                "right": "",
                "bottom": "",
            }
        ],
    )
    output = tmp_path / "hard-negatives"

    with pytest.raises(ValueError, match="must be complete"):
        build_pilot_hard_negative_review(review, output)

    assert not output.exists()


def test_build_pilot_hard_negatives_rejects_out_of_bounds_coordinates(
    tmp_path: Path,
):
    source = tmp_path / "scan.bmp"
    Image.new("RGB", (128, 128), "gray").save(source)
    review = tmp_path / "pilot_review.csv"
    _write_review(
        review,
        [
            {
                "input": str(source),
                "review_label": "false_positive",
                "left": "120",
                "top": "120",
                "right": "140",
                "bottom": "140",
            }
        ],
    )

    with pytest.raises(ValueError, match="outside image"):
        build_pilot_hard_negative_review(review, tmp_path / "output")


def test_pilot_hard_negatives_refuses_to_overwrite_existing_output(
    tmp_path: Path,
):
    review = tmp_path / "pilot_review.csv"
    _write_review(
        review,
        [
            {
                "input": "unused.bmp",
                "review_label": "correct_normal",
                "left": "",
                "top": "",
                "right": "",
                "bottom": "",
            }
        ],
    )
    output = tmp_path / "existing"
    output.mkdir()

    with pytest.raises(FileExistsError, match="archive it"):
        build_pilot_hard_negative_review(review, output)
