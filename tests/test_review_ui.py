from __future__ import annotations

import csv
import threading
from pathlib import Path
from urllib.request import urlopen

import pytest
from PIL import Image

from ourbrain_cv.manifest import write_manifest
from ourbrain_cv.review_ui import (
    build_negative_review_ui,
    create_review_server,
    normalize_review_label,
)


def _manifest_row(image: Path, group: str, split: str) -> dict[str, str]:
    return {
        "image_path": str(image),
        "mask_path": "/mask.png",
        "group_id": group,
        "split": split,
        "width": "8",
        "height": "8",
        "mask_width": "8",
        "mask_height": "8",
        "positive_pixels": "1",
        "source_kind": "paired",
    }


def test_normalize_review_label_is_conservative() -> None:
    assert normalize_review_label("normal") == "negative"
    assert normalize_review_label("1") == "crack"
    assert normalize_review_label("maybe") == "uncertain"
    assert normalize_review_label("  ") == ""


def test_build_review_ui_embeds_candidates_and_split_summary(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates"
    candidates.mkdir()
    train_image = candidates / "train.png"
    test_image = candidates / "test.png"
    Image.new("RGB", (8, 8), "gray").save(train_image)
    Image.new("RGB", (8, 8), "gray").save(test_image)

    manifest = tmp_path / "manifest.csv"
    write_manifest(
        [
            _manifest_row(train_image, "001", "train"),
            _manifest_row(test_image, "002", "test"),
        ],
        manifest,
    )
    review = candidates / "negative_review.csv"
    with review.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "candidate_path",
                "source_image_path",
                "group_id",
                "left",
                "top",
                "right",
                "bottom",
                "review_label",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "candidate_path": str(train_image),
                "source_image_path": "/raw/001.bmp",
                "group_id": "001",
                "left": "0",
                "top": "0",
                "right": "8",
                "bottom": "8",
                "review_label": "normal",
            }
        )
        writer.writerow(
            {
                "candidate_path": str(test_image),
                "source_image_path": "/raw/002.bmp",
                "group_id": "002",
                "left": "0",
                "top": "0",
                "right": "8",
                "bottom": "8",
                "review_label": "",
            }
        )

    output = candidates / "review.html"
    result = build_negative_review_ui(
        review,
        output,
        manifest_csv=manifest,
    )
    document = output.read_text(encoding="utf-8")

    assert result["total"] == 2
    assert result["reviewed"] == 1
    assert result["negative"] == 1
    assert result["missing_candidate_files"] == 0
    assert result["by_split"]["train"]["negative"] == 1
    assert result["by_split"]["test"]["unreviewed"] == 1
    assert "train.png" in document
    assert "test.png" in document
    assert "검수 CSV 내보내기" in document
    assert "target_split" in document


def test_create_review_server_serves_generated_document(tmp_path: Path) -> None:
    review_html = tmp_path / "review page.html"
    review_html.write_text("<!doctype html><title>review-ready</title>", encoding="utf-8")
    server, url = create_review_server(review_html, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urlopen(url, timeout=5) as response:  # noqa: S310 - loopback test server
            document = response.read().decode()
        assert response.status == 200
        assert "review-ready" in document
        assert url.endswith("/review%20page.html")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_create_review_server_rejects_non_loopback_binding(tmp_path: Path) -> None:
    review_html = tmp_path / "review.html"
    review_html.write_text("<!doctype html>", encoding="utf-8")

    with pytest.raises(ValueError, match="loopback-only"):
        create_review_server(review_html, host="0.0.0.0")


def test_build_review_ui_rejects_candidates_outside_served_root(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates"
    html_dir = tmp_path / "html"
    candidates.mkdir()
    html_dir.mkdir()
    candidate = candidates / "candidate.png"
    Image.new("RGB", (8, 8), "gray").save(candidate)
    review = candidates / "negative_review.csv"
    with review.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["candidate_path", "group_id", "review_label"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "candidate_path": str(candidate),
                "group_id": "001",
                "review_label": "",
            }
        )

    with pytest.raises(ValueError, match="outside the review server root"):
        build_negative_review_ui(review, html_dir / "review.html")
