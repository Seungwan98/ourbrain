import numpy as np
import pytest

from ourbrain_cv.tiling import axis_starts, blend_window, coverage_mask, iter_tiles


def test_axis_starts_pin_last_tile_for_full_coverage():
    assert axis_starts(1000, 256, 64) == [0, 192, 384, 576, 744]
    assert axis_starts(128, 256, 64) == [0]


def test_iter_tiles_covers_every_pixel_and_clips_small_images():
    image_size = (101, 77)
    tiles = list(iter_tiles(image_size, tile_size=32, overlap=8))

    assert tiles[0].box == (0, 0, 32, 32)
    assert tiles[-1].box == (69, 45, 101, 77)
    assert coverage_mask(image_size, 32, 8).all()


def test_invalid_overlap_rejected():
    with pytest.raises(ValueError):
        axis_starts(100, 32, 32)


def test_blend_window_is_nonzero_float32_and_peaks_at_one():
    window = blend_window(32, 48, minimum=0.05)

    assert window.dtype == np.float32
    assert window.shape == (32, 48)
    assert window.min() >= 0.05
    assert window.max() == pytest.approx(1.0)
