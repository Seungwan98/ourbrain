"""Safe image-opening helpers for trusted, very large local scans."""

from __future__ import annotations

import mmap
import struct
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import Image


class UnsupportedBmpLayoutError(ValueError):
    """Raised when a BMP cannot use the direct uncompressed 24-bit crop path."""


@contextmanager
def open_trusted_large_image(path: str | Path) -> Iterator[Image.Image]:
    """Open a trusted local scan while temporarily lifting Pillow's pixel cap.

    Pillow's decompression-bomb guard is valuable for untrusted uploads, but the
    known tunnel BMP files are intentionally about 237 million pixels. The
    global limit is restored immediately after the image context closes.
    """
    previous_limit = Image.MAX_IMAGE_PIXELS
    try:
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(path) as image:
            yield image
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit


def crop_trusted_image(
    path: str | Path, box: tuple[int, int, int, int]
) -> Image.Image:
    """Crop a trusted local image, fast-pathing uncompressed 24-bit BMP files.

    Pillow decodes an entire BMP before cropping it. Our raw tunnel scans are
    roughly 600–700MB each, while review patches are only 512×512. For the known
    uncompressed 24-bit BMP layout, this function memory-maps the file and reads
    only the requested BGR rows. Other image layouts use Pillow's normal path.
    """

    image_path = Path(path)
    if image_path.suffix.lower() == ".bmp":
        try:
            return _crop_uncompressed_bmp24(image_path, box)
        except UnsupportedBmpLayoutError:
            # A different BMP layout is still a valid trusted input; let Pillow
            # handle it rather than weakening format checks in the fast path.
            pass
    with open_trusted_large_image(image_path) as image:
        return image.crop(box).convert("RGB")


def _crop_uncompressed_bmp24(
    path: Path, box: tuple[int, int, int, int]
) -> Image.Image:
    left, top, right, bottom = box
    with path.open("rb") as handle:
        header = handle.read(54)
        if len(header) < 54 or header[:2] != b"BM":
            raise UnsupportedBmpLayoutError("not a Windows BMP")
        pixel_offset = struct.unpack_from("<I", header, 10)[0]
        dib_size = struct.unpack_from("<I", header, 14)[0]
        width = struct.unpack_from("<i", header, 18)[0]
        signed_height = struct.unpack_from("<i", header, 22)[0]
        planes, bits_per_pixel = struct.unpack_from("<HH", header, 26)
        compression = struct.unpack_from("<I", header, 30)[0]
        if (
            dib_size < 40
            or width <= 0
            or signed_height == 0
            or planes != 1
            or bits_per_pixel != 24
            or compression != 0
        ):
            raise UnsupportedBmpLayoutError("unsupported BMP layout")

        height = abs(signed_height)
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            raise ValueError(
                f"crop box {box} is outside BMP dimensions {(width, height)}"
            )
        crop_width = right - left
        crop_height = bottom - top
        row_stride = ((width * 3 + 3) // 4) * 4
        rgb = np.empty((crop_height, crop_width, 3), dtype=np.uint8)

        with mmap.mmap(handle.fileno(), length=0, access=mmap.ACCESS_READ) as mapped:
            for output_y in range(crop_height):
                source_y = top + output_y
                stored_y = source_y if signed_height < 0 else height - 1 - source_y
                start = pixel_offset + stored_y * row_stride + left * 3
                stop = start + crop_width * 3
                if stop > len(mapped):
                    raise ValueError("truncated BMP pixel data")
                bgr = np.frombuffer(mapped, dtype=np.uint8, count=crop_width * 3, offset=start)
                rgb[output_y] = bgr.reshape(crop_width, 3)[:, ::-1]
                del bgr
    return Image.fromarray(rgb, mode="RGB")
