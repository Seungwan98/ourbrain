from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from ourbrain_cv.manifest import write_manifest
from ourbrain_cv.remote_review import (
    _remote_review_endpoint,
    build_remote_review_bundle,
    remote_review_status,
)


def test_build_remote_review_bundle_copies_assets_and_binds_dataset(tmp_path: Path):
    review_dir = tmp_path / "review"
    review_dir.mkdir()
    candidate = review_dir / "candidate.png"
    Image.new("RGB", (8, 8), "gray").save(candidate)
    review = review_dir / "negative_review.csv"
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
                "candidate_path": str(candidate),
                "source_image_path": "/raw/001.bmp",
                "group_id": "001",
                "left": "0",
                "top": "0",
                "right": "8",
                "bottom": "8",
                "review_label": "",
            }
        )

    manifest = tmp_path / "manifest.csv"
    write_manifest(
        [
            {
                "image_path": "/image.png",
                "mask_path": "/mask.png",
                "group_id": "001",
                "split": "val",
                "width": "8",
                "height": "8",
                "mask_width": "8",
                "mask_height": "8",
                "positive_pixels": "1",
                "source_kind": "paired",
            }
        ],
        manifest,
    )
    template = tmp_path / "template"
    (template / "public").mkdir(parents=True)
    (template / "lib").mkdir()
    (template / "public" / "index.html").write_text("review", encoding="utf-8")
    output = tmp_path / "build" / "remote"

    result = build_remote_review_bundle(
        review,
        manifest,
        output,
        template_dir=template,
    )
    metadata = json.loads((output / "bundle-metadata.json").read_text())
    dataset_module = (output / "lib" / "dataset.js").read_text()
    deployed = next((output / "private-candidates").iterdir())

    assert result["candidates"] == 1
    assert len(result["dataset_id"]) == 64
    assert metadata["dataset_id"] == result["dataset_id"]
    assert deployed.read_bytes() == candidate.read_bytes()
    assert result["review_csv_sha256"]
    assert f"review-images/{result['dataset_id']}/" in dataset_module
    assert '"imageSha256":' in dataset_module
    assert "/api/candidate?id=" in dataset_module
    assert '"target_split": "val"' in dataset_module
    assert "EXPORT_FIELDS" in dataset_module
    assert not (output / "public" / "candidates.json").exists()

    first_dataset_id = result["dataset_id"]
    first_deployed_name = deployed.name
    Image.new("RGB", (8, 8), "black").save(candidate)
    changed = build_remote_review_bundle(
        review,
        manifest,
        output,
        template_dir=template,
    )
    changed_deployed = next((output / "private-candidates").iterdir())
    assert changed["dataset_id"] != first_dataset_id
    assert changed_deployed.name != first_deployed_name


def test_remote_review_template_bundles_private_candidate_fallback():
    template = Path("web/remote-review-template")
    vercel_config = json.loads((template / "vercel.json").read_text())

    assert (
        vercel_config["functions"]["api/candidate.js"]["includeFiles"]
        == "private-candidates/**"
    )
    assert (template / "lib" / "candidate-file.js").is_file()


def test_remote_review_endpoint_requires_https_except_loopback(monkeypatch):
    monkeypatch.setenv("OURBRAIN_REVIEW_TOKEN", "secret")
    with pytest.raises(ValueError, match="must use HTTPS"):
        remote_review_status("http://review.example.com")

    assert (
        _remote_review_endpoint("https://review.example.com", "api/review-state")
        == "https://review.example.com/api/review-state"
    )
    assert (
        _remote_review_endpoint("http://127.0.0.1:3000", "api/review-state")
        == "http://127.0.0.1:3000/api/review-state"
    )
