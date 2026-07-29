"""Tiled inference and post-processing for tunnel crack segmentation."""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image

from ourbrain_cv.image_io import open_trusted_large_image
from ourbrain_cv.tiling import Tile, blend_window, iter_tiles

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ProbabilityPredictor(Protocol):
    """Protocol for tile-level crack probability predictors."""

    def predict_tile_probability(self, tile: Image.Image) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class TiledInferenceConfig:
    tile_size: int = 512
    overlap: int = 96
    threshold: float = 0.5
    minimum_component_pixels: int = 8
    image_level_minimum_pixels: int = 16
    crack_class: int = 1
    blend_minimum: float = 0.05
    memmap_dir: str | None = None
    preview_max_size: int = 2048
    maximum_positive_ratio: float = 0.25


@dataclass(frozen=True, slots=True)
class InferenceOutputs:
    probability_path: str
    mask_path: str
    overlay_path: str
    summary_path: str
    summary: dict[str, Any]


class SegmentationModelAdapter:
    """Adapter from a torch segmentation model to per-tile crack probabilities.

    The adapter intentionally does not require a Hugging Face processor. Tiles are
    converted from PIL RGB to ImageNet-normalized tensors. Model logits are upsampled
    back to the input tile size and softmaxed; ``crack_class`` is returned.
    """

    def __init__(self, model: Any, *, crack_class: int = 1, device: str | None = None) -> None:
        self.model = model
        self.crack_class = crack_class
        self.device = device
        if hasattr(self.model, "eval"):
            self.model.eval()
        if device is not None and hasattr(self.model, "to"):
            self.model.to(device)
            if str(device).startswith("mps"):
                from ourbrain_cv.modeling import enable_mps_compatibility

                enable_mps_compatibility(self.model)

    def predict_tile_probability(self, tile: Image.Image) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        width, height = tile.size
        array = np.asarray(tile.convert("RGB"), dtype=np.float32) / 255.0
        array = (array - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)
        if self.device is not None:
            tensor = tensor.to(self.device)

        with torch.no_grad():
            output = self.model(pixel_values=tensor)
            logits = _extract_logits(output)
            logits = F.interpolate(
                logits, size=(height, width), mode="bilinear", align_corners=False
            )
            probability = torch.softmax(logits, dim=1)[0, self.crack_class]
        return probability.detach().cpu().numpy().astype(np.float32, copy=False)


def load_checkpoint_adapter(
    checkpoint: str | Path, *, crack_class: int = 1, device: str | None = None
) -> SegmentationModelAdapter:
    """Load a checkpoint via ``ourbrain_cv.modeling`` using a lazy import.

    ``modeling.py`` may evolve while inference stays import-light. Supported helper
    names are tried in order to keep this boundary stable.
    """

    from ourbrain_cv import modeling  # type: ignore[attr-defined]

    for helper_name in ("load_model_for_inference", "load_segmentation_model", "load_model"):
        helper = getattr(modeling, helper_name, None)
        if helper is not None:
            model = helper(checkpoint)
            return SegmentationModelAdapter(model, crack_class=crack_class, device=device)
    raise AttributeError("ourbrain_cv.modeling has no supported checkpoint loading helper")


def run_tiled_inference(
    image_path: str | Path,
    predictor: ProbabilityPredictor | Any,
    output_dir: str | Path,
    config: TiledInferenceConfig | None = None,
) -> InferenceOutputs:
    """Run tiled crack inference over a large image and save production artifacts."""

    config = config or TiledInferenceConfig()
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predict = _as_predict_function(predictor)
    with open_trusted_large_image(image_path) as image:
        width, height = image.size
        accumulator, weights, tmpdir = _allocate_accumulators((height, width), config.memmap_dir)
        try:
            for tile in iter_tiles((width, height), config.tile_size, config.overlap):
                tile_image = image.crop(tile.box)
                probability = _coerce_probability(predict(tile_image), tile)
                window = blend_window(tile.height, tile.width, config.blend_minimum)
                region = np.s_[tile.y : tile.y + tile.height, tile.x : tile.x + tile.width]
                accumulator[region] += probability * window
                weights[region] += window

            stem = image_path.stem
            probability_path = output_dir / f"{stem}_probability.npy"
            mask_path = output_dir / f"{stem}_mask.png"
            overlay_path = output_dir / f"{stem}_overlay.png"
            summary_path = output_dir / f"{stem}_summary.json"

            _, binary_mask = _save_probability_and_threshold(
                probability_path,
                accumulator,
                weights,
                config.threshold,
                block_rows=config.tile_size,
            )
            pixel_count = width * height
            raw_positive_pixels = int(binary_mask.sum())
            raw_positive_ratio = raw_positive_pixels / float(pixel_count)
            quality_gate_passed = raw_positive_ratio <= config.maximum_positive_ratio
            if quality_gate_passed:
                cleaned_mask, component_sizes = remove_small_components(
                    binary_mask, config.minimum_component_pixels
                )
                postprocessing_applied = True
                quality_gate_reason = None
            else:
                # Python connected-component traversal is intentionally skipped for
                # implausibly dense predictions. This prevents an uncalibrated model
                # from turning a 240MP tunnel scan into millions of Python objects.
                cleaned_mask = binary_mask
                component_sizes = []
                postprocessing_applied = False
                quality_gate_reason = (
                    "raw positive ratio exceeds maximum_positive_ratio; "
                    "presence decision and connected-component analysis were withheld"
                )
            crack_pixels = int(cleaned_mask.sum())
            presence = (
                crack_pixels >= config.image_level_minimum_pixels
                if quality_gate_passed
                else None
            )
            _save_mask(mask_path, cleaned_mask)
            _save_overlay(overlay_path, image, cleaned_mask, max_size=config.preview_max_size)
            summary = {
                "image_path": str(image_path),
                "image_size": {"width": width, "height": height},
                "config": asdict(config),
                "threshold": config.threshold,
                "raw_positive_pixels": raw_positive_pixels,
                "raw_positive_ratio": raw_positive_ratio,
                "crack_pixels": crack_pixels,
                "crack_ratio": crack_pixels / float(pixel_count),
                "presence": presence,
                "quality_gate": {
                    "passed": quality_gate_passed,
                    "reason": quality_gate_reason,
                    "maximum_positive_ratio": config.maximum_positive_ratio,
                },
                "postprocessing_applied": postprocessing_applied,
                "components": {
                    "count": len(component_sizes) if postprocessing_applied else None,
                    "sizes": component_sizes,
                },
                "outputs": {
                    "probability": str(probability_path),
                    "mask": str(mask_path),
                    "overlay": str(overlay_path),
                },
            }
            summary_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            return InferenceOutputs(
                probability_path=str(probability_path),
                mask_path=str(mask_path),
                overlay_path=str(overlay_path),
                summary_path=str(summary_path),
                summary=summary,
            )
        finally:
            if tmpdir is not None:
                tmpdir.cleanup()


def remove_small_components(
    mask: np.ndarray, minimum_pixels: int, *, connectivity: int = 8
) -> tuple[np.ndarray, list[int]]:
    """Remove connected components smaller than ``minimum_pixels`` without cv2."""

    if mask.ndim != 2:
        raise ValueError("mask must be 2D")
    if minimum_pixels <= 1:
        return mask.astype(bool, copy=True), _component_sizes(
            mask.astype(bool, copy=False), connectivity
        )
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")

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
                if 0 <= ny < height and 0 <= nx < width and source[ny, nx] and not visited[ny, nx]:
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
                if 0 <= ny < height and 0 <= nx < width and source[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        sizes_out.append(size)
    return sizes_out


def _neighbors(connectivity: int) -> tuple[tuple[int, int], ...]:
    if connectivity == 4:
        return ((-1, 0), (0, -1), (0, 1), (1, 0))
    return ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def _extract_logits(output: Any) -> Any:
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, dict) and "logits" in output:
        return output["logits"]
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def _as_predict_function(predictor: ProbabilityPredictor | Any):
    if hasattr(predictor, "predict_tile_probability"):
        return predictor.predict_tile_probability
    if callable(predictor):
        return predictor
    raise TypeError("predictor must be callable or implement predict_tile_probability")


def _coerce_probability(probability: np.ndarray, tile: Tile) -> np.ndarray:
    probability = np.asarray(probability, dtype=np.float32)
    expected = (tile.height, tile.width)
    if probability.shape != expected:
        raise ValueError(f"predictor returned shape {probability.shape}, expected {expected}")
    return probability


def _allocate_accumulators(shape: tuple[int, int], memmap_dir: str | None):
    if memmap_dir is None:
        return np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32), None

    directory = Path(memmap_dir)
    directory.mkdir(parents=True, exist_ok=True)
    tmpdir = tempfile.TemporaryDirectory(dir=directory)
    tmp_path = Path(tmpdir.name)
    accumulator = np.memmap(tmp_path / "accumulator.dat", dtype=np.float32, mode="w+", shape=shape)
    weights = np.memmap(tmp_path / "weights.dat", dtype=np.float32, mode="w+", shape=shape)
    accumulator[:] = 0
    weights[:] = 0
    return accumulator, weights, tmpdir


def _save_probability_and_threshold(
    probability_path: Path,
    accumulator: np.ndarray,
    weights: np.ndarray,
    threshold: float,
    *,
    block_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Persist probability and create threshold mask without an extra full float copy.

    For in-memory arrays this still returns a normal ndarray. For memmap-backed
    accumulators it writes the final probability directly to a .npy memmap in row
    blocks, avoiding the additional ~0.9GB float32 copy required by 10k×24k images.
    """

    is_memmap = isinstance(accumulator, np.memmap) or isinstance(weights, np.memmap)
    if is_memmap:
        probability = np.lib.format.open_memmap(
            probability_path, mode="w+", dtype=np.float32, shape=accumulator.shape
        )
        binary = np.zeros(accumulator.shape, dtype=bool)
        rows = max(1, int(block_rows))
        for y in range(0, accumulator.shape[0], rows):
            region = np.s_[y : y + rows, :]
            np.divide(
                accumulator[region],
                weights[region],
                out=probability[region],
                where=weights[region] > 0,
            )
            binary[region] = probability[region] > threshold
        probability.flush()
        return probability, binary

    probability = np.zeros(accumulator.shape, dtype=np.float32)
    np.divide(accumulator, weights, out=probability, where=weights > 0)
    np.save(probability_path, probability)
    return probability, probability > threshold


def _save_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(path)


def _save_overlay(path: Path, image: Image.Image, mask: np.ndarray, *, max_size: int) -> None:
    preview = image.convert("RGB").copy()
    if max_size > 0:
        preview.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    mask_image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L").resize(
        preview.size, Image.Resampling.NEAREST
    )
    preview_arr = np.asarray(preview, dtype=np.float32)
    mask_arr = np.asarray(mask_image, dtype=bool)
    red = np.zeros_like(preview_arr)
    red[..., 0] = 255
    preview_arr[mask_arr] = preview_arr[mask_arr] * 0.45 + red[mask_arr] * 0.55
    Image.fromarray(np.clip(preview_arr, 0, 255).astype(np.uint8), mode="RGB").save(path)
