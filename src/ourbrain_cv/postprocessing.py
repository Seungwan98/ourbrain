"""Shared binary-mask post-processing for evaluation and inference."""

from __future__ import annotations

import numpy as np


def remove_small_components(
    mask: np.ndarray, minimum_pixels: int, *, connectivity: int = 8
) -> tuple[np.ndarray, list[int]]:
    """Remove connected components smaller than ``minimum_pixels`` without cv2."""

    if mask.ndim != 2:
        raise ValueError("mask must be 2D")
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    if minimum_pixels <= 1:
        return mask.astype(bool, copy=True), _component_sizes(
            mask.astype(bool, copy=False), connectivity
        )

    source = mask.astype(bool, copy=False)
    visited = np.zeros(source.shape, dtype=bool)
    cleaned = np.zeros(source.shape, dtype=bool)
    kept_sizes: list[int] = []
    height, width = source.shape
    neighbors = _neighbors(connectivity)

    for start_y, start_x in zip(*np.nonzero(source), strict=False):
        if visited[start_y, start_x]:
            continue
        stack = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        pixels: list[tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            pixels.append((y, x))
            for dy, dx in neighbors:
                ny = y + dy
                nx = x + dx
                if (
                    0 <= ny < height
                    and 0 <= nx < width
                    and source[ny, nx]
                    and not visited[ny, nx]
                ):
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        if len(pixels) >= minimum_pixels:
            kept_sizes.append(len(pixels))
            ys, xs = zip(*pixels, strict=False)
            cleaned[ys, xs] = True
    return cleaned, kept_sizes


def _component_sizes(mask: np.ndarray, connectivity: int) -> list[int]:
    source = mask.astype(bool, copy=False)
    visited = np.zeros(source.shape, dtype=bool)
    sizes_out: list[int] = []
    height, width = source.shape
    neighbors = _neighbors(connectivity)
    for start_y, start_x in zip(*np.nonzero(source), strict=False):
        if visited[start_y, start_x]:
            continue
        stack = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        size = 0
        while stack:
            y, x = stack.pop()
            size += 1
            for dy, dx in neighbors:
                ny = y + dy
                nx = x + dx
                if (
                    0 <= ny < height
                    and 0 <= nx < width
                    and source[ny, nx]
                    and not visited[ny, nx]
                ):
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        sizes_out.append(size)
    return sizes_out


def _neighbors(connectivity: int) -> tuple[tuple[int, int], ...]:
    if connectivity == 4:
        return ((-1, 0), (0, -1), (0, 1), (1, 0))
    return (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )


__all__ = ["remove_small_components"]
