"""Tile coordinate and overlap-blending utilities for large tunnel images."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Tile:
    """A crop window in image coordinates."""

    x: int
    y: int
    width: int
    height: int

    @property
    def box(self) -> tuple[int, int, int, int]:
        """Return a Pillow-compatible ``(left, upper, right, lower)`` crop box."""

        return (self.x, self.y, self.x + self.width, self.y + self.height)


def axis_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    """Return start coordinates that fully cover one image axis.

    The last tile is pinned to ``length - tile_size`` when needed, so border pixels are
    always included even when the stride does not divide the axis length exactly.
    """

    if length <= 0:
        raise ValueError("length must be positive")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= tile_size:
        raise ValueError("overlap must be smaller than tile_size")

    if length <= tile_size:
        return [0]

    step = tile_size - overlap
    last_start = length - tile_size
    starts = list(range(0, last_start + 1, step))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def iter_tiles(
    image_size: tuple[int, int],
    tile_size: int | tuple[int, int],
    overlap: int | tuple[int, int],
) -> Iterator[Tile]:
    """Yield tiles covering ``image_size`` without loading image pixels.

    Args:
        image_size: ``(width, height)``.
        tile_size: scalar or ``(tile_width, tile_height)``.
        overlap: scalar or ``(overlap_x, overlap_y)``.
    """

    width, height = image_size
    tile_w, tile_h = _pair(tile_size, "tile_size")
    overlap_x, overlap_y = _pair(overlap, "overlap")

    xs = axis_starts(width, tile_w, overlap_x)
    ys = axis_starts(height, tile_h, overlap_y)
    for y in ys:
        for x in xs:
            yield Tile(x=x, y=y, width=min(tile_w, width - x), height=min(tile_h, height - y))


def blend_window(height: int, width: int, minimum: float = 0.05) -> np.ndarray:
    """Create a non-zero Hann-style 2D blending window as ``float32``.

    A pure Hann window is zero at tile borders, which can leave uncovered pixels when a
    tile touches the image edge. This function floors the weights to ``minimum`` and
    normalizes the peak to 1.0.
    """

    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if not 0 < minimum <= 1:
        raise ValueError("minimum must be in (0, 1]")

    wy = _nonzero_hann(height, minimum)
    wx = _nonzero_hann(width, minimum)
    window = np.outer(wy, wx).astype(np.float32, copy=False)
    peak = float(window.max())
    if peak > 0:
        window /= peak
    return np.clip(window, minimum, 1.0, out=window)


def coverage_mask(
    image_size: tuple[int, int], tile_size: int | tuple[int, int], overlap: int | tuple[int, int]
) -> np.ndarray:
    """Return a boolean mask showing pixels covered by generated tiles.

    Intended for tests and diagnostics; avoid for production-size images.
    """

    width, height = image_size
    covered = np.zeros((height, width), dtype=bool)
    for tile in iter_tiles(image_size, tile_size, overlap):
        covered[tile.y : tile.y + tile.height, tile.x : tile.x + tile.width] = True
    return covered


def _pair(value: int | tuple[int, int], name: str) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    if len(value) != 2:
        raise ValueError(f"{name} must be an int or a pair")
    return int(value[0]), int(value[1])


def _nonzero_hann(size: int, minimum: float) -> np.ndarray:
    if size == 1:
        return np.ones(1, dtype=np.float32)
    values = np.hanning(size).astype(np.float32)
    max_value = float(values.max())
    if max_value > 0:
        values /= max_value
    return np.clip(values, minimum, 1.0)
