import json
from pathlib import Path

import numpy as np
from PIL import Image

from ourbrain_cv.inference import TiledInferenceConfig, remove_small_components, run_tiled_inference


class RedChannelPredictor:
    def predict_tile_probability(self, tile: Image.Image) -> np.ndarray:
        return np.asarray(tile.convert("RGB"), dtype=np.float32)[..., 0] / 255.0


class ConstantPredictor:
    def __init__(self, value: float):
        self.value = value

    def predict_tile_probability(self, tile: Image.Image) -> np.ndarray:
        width, height = tile.size
        return np.full((height, width), self.value, dtype=np.float32)


def test_remove_small_components_keeps_only_large_component():
    mask = np.zeros((12, 12), dtype=bool)
    mask[1, 1] = True
    mask[5:8, 6:9] = True

    cleaned, sizes = remove_small_components(mask, minimum_pixels=4)

    assert not cleaned[1, 1]
    assert cleaned[5:8, 6:9].all()
    assert sizes == [9]


def test_run_tiled_inference_stitches_tiles_and_writes_artifacts(tmp_path: Path):
    image = np.zeros((29, 37, 3), dtype=np.uint8)
    image[10:18, 12:25, 0] = 255
    image_path = tmp_path / "synthetic.png"
    Image.fromarray(image, mode="RGB").save(image_path)

    outputs = run_tiled_inference(
        image_path,
        RedChannelPredictor(),
        tmp_path / "out",
        TiledInferenceConfig(
            tile_size=16,
            overlap=4,
            threshold=0.5,
            minimum_component_pixels=3,
            image_level_minimum_pixels=10,
            preview_max_size=64,
        ),
    )

    probability = np.load(outputs.probability_path)
    mask = np.asarray(Image.open(outputs.mask_path)) > 0
    summary = json.loads(Path(outputs.summary_path).read_text(encoding="utf-8"))

    assert probability.shape == (29, 37)
    assert probability[10:18, 12:25].min() > 0.99
    assert probability[:5, :5].max() == 0
    assert mask.sum() == 8 * 13
    assert summary["image_size"] == {"width": 37, "height": 29}
    assert summary["crack_pixels"] == 104
    assert summary["presence"] is True
    assert summary["components"]["count"] == 1
    assert Path(outputs.overlay_path).exists()


def test_run_tiled_inference_presence_false_after_component_filter(tmp_path: Path):
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[2, 2, 0] = 255
    image_path = tmp_path / "tiny.png"
    Image.fromarray(image, mode="RGB").save(image_path)

    outputs = run_tiled_inference(
        image_path,
        RedChannelPredictor(),
        tmp_path / "out",
        TiledInferenceConfig(
            tile_size=9,
            overlap=2,
            threshold=0.5,
            minimum_component_pixels=2,
            image_level_minimum_pixels=1,
            preview_max_size=32,
        ),
    )

    mask = np.asarray(Image.open(outputs.mask_path)) > 0
    assert mask.sum() == 0
    assert outputs.summary["presence"] is False
    assert outputs.summary["components"]["count"] == 0


def test_run_tiled_inference_supports_memmap_accumulators(tmp_path: Path):
    image = np.zeros((11, 13, 3), dtype=np.uint8)
    image_path = tmp_path / "constant.png"
    Image.fromarray(image, mode="RGB").save(image_path)

    outputs = run_tiled_inference(
        image_path,
        ConstantPredictor(0.75),
        tmp_path / "out",
        TiledInferenceConfig(
            tile_size=7,
            overlap=3,
            threshold=0.5,
            minimum_component_pixels=1,
            image_level_minimum_pixels=1,
            memmap_dir=str(tmp_path / "memmap"),
            preview_max_size=32,
            maximum_positive_ratio=1.0,
        ),
    )

    probability = np.load(outputs.probability_path)
    assert np.allclose(probability, 0.75)
    assert outputs.summary["presence"] is True


def test_dense_prediction_skips_connected_components_and_withholds_presence(
    tmp_path: Path, monkeypatch
):
    image_path = tmp_path / "dense.png"
    Image.new("RGB", (64, 48), "black").save(image_path)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("connected-component traversal must be skipped")

    monkeypatch.setattr(
        "ourbrain_cv.inference.remove_small_components",
        fail_if_called,
    )
    outputs = run_tiled_inference(
        image_path,
        ConstantPredictor(0.99),
        tmp_path / "out",
        TiledInferenceConfig(
            tile_size=32,
            overlap=8,
            threshold=0.5,
            minimum_component_pixels=2,
            image_level_minimum_pixels=1,
            preview_max_size=32,
            maximum_positive_ratio=0.1,
        ),
    )

    assert outputs.summary["quality_gate"]["passed"] is False
    assert outputs.summary["postprocessing_applied"] is False
    assert outputs.summary["presence"] is None
    assert outputs.summary["components"]["count"] is None
