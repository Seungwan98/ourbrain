from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from ourbrain_cv.modeling import (
    LABEL2ID,
    MPSCompatibleAdaptiveAvgPool2d,
    SegmentationModelWrapper,
    _load_state_file,
    build_model,
    enable_mps_compatibility,
    load_model_for_inference,
    load_pretrained_segmentation_model,
    load_upernet_swin_tiny,
)


class LowResModel(nn.Module):
    def forward(self, pixel_values, **kwargs):
        return SimpleNamespace(logits=torch.randn(pixel_values.shape[0], 2, 4, 4))


def test_wrapper_upsamples_logits_to_label_size():
    wrapper = SegmentationModelWrapper(LowResModel())
    pixel_values = torch.randn(1, 3, 16, 16)
    labels = torch.zeros(1, 12, 10, dtype=torch.long)
    outputs = wrapper(pixel_values=pixel_values, labels=labels)
    assert outputs["logits"].shape == (1, 2, 12, 10)


def test_load_upernet_uses_expected_hf_arguments(monkeypatch):
    calls = {}

    class DummyHF:
        @classmethod
        def from_pretrained(cls, checkpoint, **kwargs):
            calls["checkpoint"] = checkpoint
            calls["kwargs"] = kwargs
            return LowResModel()

    import ourbrain_cv.modeling as modeling

    def fake_hf_load(checkpoint, **kwargs):
        return DummyHF.from_pretrained(checkpoint, **kwargs)

    monkeypatch.setattr(modeling, "_hf_load", fake_hf_load)
    model = load_upernet_swin_tiny("openmmlab/upernet-swin-tiny")
    assert isinstance(model, SegmentationModelWrapper)
    assert calls["checkpoint"] == "openmmlab/upernet-swin-tiny"
    assert calls["kwargs"]["architecture"] == "upernet"
    assert calls["kwargs"]["num_labels"] == 2
    assert calls["kwargs"]["label2id"] == LABEL2ID
    assert calls["kwargs"]["ignore_mismatched_sizes"] is True


def test_build_model_selects_segformer_architecture(monkeypatch):
    calls = {}

    def fake_hf_load(checkpoint, **kwargs):
        calls["checkpoint"] = checkpoint
        calls["kwargs"] = kwargs
        return LowResModel()

    import ourbrain_cv.modeling as modeling

    monkeypatch.setattr(modeling, "_hf_load", fake_hf_load)
    model = build_model(
        {
            "architecture": "segformer",
            "checkpoint": "nvidia/segformer-b1-finetuned-ade-512-512",
        }
    )

    assert isinstance(model, SegmentationModelWrapper)
    assert calls["kwargs"]["architecture"] == "segformer"


def test_build_model_rejects_unknown_architecture():
    with pytest.raises(ValueError, match="Unsupported segmentation architecture"):
        load_pretrained_segmentation_model("unused", architecture="mask2former")


def test_inference_hf_directory_uses_auto_model_detection(monkeypatch, tmp_path: Path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    calls = {}

    def fake_hf_load(checkpoint, **kwargs):
        calls["checkpoint"] = checkpoint
        calls["kwargs"] = kwargs
        return LowResModel()

    import ourbrain_cv.modeling as modeling

    monkeypatch.setattr(modeling, "_hf_load", fake_hf_load)
    model = load_model_for_inference(tmp_path)

    assert isinstance(model, SegmentationModelWrapper)
    assert calls["checkpoint"] == tmp_path
    assert calls["kwargs"]["architecture"] is None


def test_enable_mps_compatibility_replaces_adaptive_pool_without_changing_cpu_result():
    class ListBackedPool(nn.Module):
        def __init__(self):
            super().__init__()
            pool = nn.AdaptiveAvgPool2d((3, 3))
            self.layers = [pool]
            self.add_module("0", pool)

        def forward(self, inputs):
            return self.layers[0](inputs)

    original = ListBackedPool()
    inputs = torch.randn(1, 2, 7, 5)
    expected = original(inputs)

    converted = enable_mps_compatibility(original)

    assert isinstance(converted.layers[0], MPSCompatibleAdaptiveAvgPool2d)
    torch.testing.assert_close(converted(inputs), expected)


def test_strict_checkpoint_reload_handles_wrapper_prefix_and_preserves_logits(
    tmp_path: Path,
):
    source = SegmentationModelWrapper(nn.Conv2d(3, 2, kernel_size=1))
    checkpoint = tmp_path / "model.safetensors"
    save_file(source.state_dict(), str(checkpoint))
    target = nn.Conv2d(3, 2, kernel_size=1)

    _load_state_file(target, checkpoint)

    inputs = torch.randn(1, 3, 8, 8)
    torch.testing.assert_close(source.model(inputs), target(inputs))


def test_strict_checkpoint_reload_rejects_partial_state(tmp_path: Path):
    checkpoint = tmp_path / "model.safetensors"
    save_file({"weight": torch.randn(2, 3, 1, 1)}, str(checkpoint))

    with pytest.raises(RuntimeError, match="Checkpoint is incompatible"):
        _load_state_file(nn.Conv2d(3, 2, kernel_size=1), checkpoint)
