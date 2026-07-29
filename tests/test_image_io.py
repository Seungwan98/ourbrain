from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ourbrain_cv.image_io import crop_trusted_image, open_trusted_large_image


def test_open_trusted_large_image_temporarily_lifts_and_restores_limit(tmp_path: Path) -> None:
    path = tmp_path / "large.png"
    Image.new("RGB", (32, 32), "gray").save(path)
    original = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = 10
    try:
        with open_trusted_large_image(path) as image:
            assert image.size == (32, 32)
        assert Image.MAX_IMAGE_PIXELS == 10
    finally:
        Image.MAX_IMAGE_PIXELS = original


def test_crop_trusted_image_reads_only_requested_bmp_region(tmp_path: Path) -> None:
    array = np.zeros((9, 11, 3), dtype=np.uint8)
    yy, xx = np.indices(array.shape[:2])
    array[..., 0] = xx * 11
    array[..., 1] = yy * 17
    array[..., 2] = (xx + yy) * 7
    path = tmp_path / "scan.bmp"
    Image.fromarray(array, mode="RGB").save(path)

    crop = crop_trusted_image(path, (3, 2, 9, 8))

    np.testing.assert_array_equal(np.asarray(crop), array[2:8, 3:9])


def test_crop_trusted_image_rejects_out_of_bounds_bmp_crop(tmp_path: Path) -> None:
    path = tmp_path / "scan.bmp"
    Image.new("RGB", (11, 9), "gray").save(path)

    with pytest.raises(ValueError, match="outside BMP dimensions"):
        crop_trusted_image(path, (3, 2, 12, 8))


def test_crop_trusted_image_rejects_truncated_bmp_pixels(tmp_path: Path) -> None:
    path = tmp_path / "truncated.bmp"
    Image.new("RGB", (11, 9), "gray").save(path)
    path.write_bytes(path.read_bytes()[:-20])

    with pytest.raises(ValueError, match="truncated BMP pixel data"):
        crop_trusted_image(path, (0, 0, 11, 9))
