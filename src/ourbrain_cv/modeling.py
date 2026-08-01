"""Model construction helpers for tunnel crack semantic segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

DEFAULT_CHECKPOINT = "openmmlab/upernet-swin-tiny"
DEFAULT_ARCHITECTURE = "upernet"
SUPPORTED_ARCHITECTURES = frozenset({"upernet", "segformer"})
ID2LABEL = {0: "background", 1: "crack"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for the Hugging Face segmentation backbone."""

    architecture: str = DEFAULT_ARCHITECTURE
    checkpoint: str = DEFAULT_CHECKPOINT
    num_labels: int = 2
    id2label: dict[int, str] | None = None
    label2id: dict[str, int] | None = None


class SegmentationModelWrapper(nn.Module):
    """Normalize Hugging Face segmentation outputs to the target mask size."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, pixel_values: torch.Tensor, labels: torch.Tensor | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        outputs = self.model(pixel_values=pixel_values, **kwargs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
        target_size = labels.shape[-2:] if labels is not None else pixel_values.shape[-2:]
        if logits.shape[-2:] != tuple(target_size):
            logits = F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)

        if isinstance(outputs, dict):
            result = dict(outputs)
            result["logits"] = logits
            return result
        return {"logits": logits, "raw_outputs": outputs}

    def save_pretrained(self, output_dir: str | Path, **kwargs: Any) -> None:
        """Delegate Hugging Face-style saving when available."""

        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(output_dir, **kwargs)
            return
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        from safetensors.torch import save_file

        save_file(self.state_dict(), str(output / "model.safetensors"))


class MPSCompatibleAdaptiveAvgPool2d(nn.Module):
    """Preserve adaptive-pooling semantics for shapes unsupported by MPS."""

    def __init__(self, output_size: int | tuple[int, int]) -> None:
        super().__init__()
        self.output_size = output_size

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.device.type == "mps":
            pooled = F.adaptive_avg_pool2d(inputs.to("cpu"), self.output_size)
            return pooled.to(inputs.device)
        return F.adaptive_avg_pool2d(inputs, self.output_size)


def enable_mps_compatibility(model: nn.Module) -> nn.Module:
    """Replace adaptive pools with a CPU-fallback implementation in-place.

    PyTorch MPS currently requires adaptive-pooling input sizes to be divisible
    by output sizes. UPerNet intentionally uses pyramid bins 1, 2, 3 and 6, so
    only those small feature maps are routed through CPU.
    """
    for name, child in list(model.named_children()):
        if isinstance(child, nn.AdaptiveAvgPool2d):
            replacement = MPSCompatibleAdaptiveAvgPool2d(child.output_size)
            setattr(model, name, replacement)
            layer_list = getattr(model, "layers", None)
            if isinstance(layer_list, list):
                for index, layer in enumerate(layer_list):
                    if layer is child:
                        layer_list[index] = replacement
        else:
            enable_mps_compatibility(child)
    return model


def _normalize_architecture(architecture: str) -> str:
    normalized = architecture.strip().lower().replace("-", "_")
    if normalized not in SUPPORTED_ARCHITECTURES:
        supported = ", ".join(sorted(SUPPORTED_ARCHITECTURES))
        raise ValueError(
            f"Unsupported segmentation architecture {architecture!r}; expected one of: {supported}"
        )
    return normalized


def _hf_load(
    checkpoint: str | Path,
    *,
    architecture: str | None = None,
    **kwargs: Any,
) -> nn.Module:
    import transformers

    if architecture is None:
        model_cls = transformers.AutoModelForSemanticSegmentation
    else:
        class_name = {
            "upernet": "UperNetForSemanticSegmentation",
            "segformer": "SegformerForSemanticSegmentation",
        }[_normalize_architecture(architecture)]
        model_cls = getattr(transformers, class_name)
    return model_cls.from_pretrained(str(checkpoint), **kwargs)


def load_pretrained_segmentation_model(
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    *,
    architecture: str = DEFAULT_ARCHITECTURE,
    num_labels: int = 2,
    id2label: dict[int, str] | None = None,
    label2id: dict[str, int] | None = None,
    wrap: bool = True,
    ignore_mismatched_sizes: bool = True,
    **kwargs: Any,
) -> nn.Module:
    """Load a supported Hugging Face semantic-segmentation architecture."""

    model = _hf_load(
        checkpoint,
        architecture=architecture,
        num_labels=num_labels,
        id2label=id2label or ID2LABEL,
        label2id=label2id or LABEL2ID,
        ignore_mismatched_sizes=ignore_mismatched_sizes,
        **kwargs,
    )
    return SegmentationModelWrapper(model) if wrap else model


def load_upernet_swin_tiny(
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    *,
    num_labels: int = 2,
    id2label: dict[int, str] | None = None,
    label2id: dict[str, int] | None = None,
    wrap: bool = True,
    ignore_mismatched_sizes: bool = True,
    **kwargs: Any,
) -> nn.Module:
    """Load ``openmmlab/upernet-swin-tiny`` for two-class crack segmentation."""

    return load_pretrained_segmentation_model(
        checkpoint,
        architecture="upernet",
        num_labels=num_labels,
        id2label=id2label or ID2LABEL,
        label2id=label2id or LABEL2ID,
        wrap=wrap,
        ignore_mismatched_sizes=ignore_mismatched_sizes,
        **kwargs,
    )


def build_model(config: ModelConfig | dict[str, Any] | None = None, **overrides: Any) -> nn.Module:
    """Build the configured segmentation model."""

    if config is None:
        params = ModelConfig().__dict__
    elif isinstance(config, dict):
        params = {**config}
    else:
        params = {**config.__dict__}
    params.update(overrides)
    return load_pretrained_segmentation_model(**params)


def _load_state_file(model: nn.Module, checkpoint: Path) -> None:
    if checkpoint.is_dir():
        safetensors_path = checkpoint / "model.safetensors"
        torch_path = checkpoint / "pytorch_model.bin"
    else:
        safetensors_path = (
            checkpoint if checkpoint.suffix == ".safetensors" else Path("__missing__")
        )
        torch_path = checkpoint if checkpoint.suffix != ".safetensors" else Path("__missing__")

    if safetensors_path.exists():
        from safetensors.torch import load_file

        state = load_file(str(safetensors_path))
    elif torch_path.exists():
        state = torch.load(torch_path, map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(f"No supported checkpoint state found in {checkpoint}")

    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint state must be a mapping, got {type(state).__name__}")

    # A fallback checkpoint can come from SegmentationModelWrapper and therefore
    # carry a uniform ``model.`` prefix. Select the key layout that exactly matches
    # the destination model, then load strictly so partial/corrupt checkpoints fail.
    model_keys = set(model.state_dict())
    candidates = [state]
    if state and all(key.startswith("model.") for key in state):
        candidates.append({key.removeprefix("model."): value for key, value in state.items()})
    selected = max(candidates, key=lambda candidate: len(model_keys & set(candidate)))
    try:
        model.load_state_dict(selected, strict=True)
    except RuntimeError as exc:
        missing = sorted(model_keys - set(selected))
        unexpected = sorted(set(selected) - model_keys)
        raise RuntimeError(
            "Checkpoint is incompatible with the configured model; "
            f"missing_keys={missing[:8]}, unexpected_keys={unexpected[:8]}"
        ) from exc


def load_model_for_inference(
    checkpoint: str | Path,
    *,
    base_checkpoint: str | Path = DEFAULT_CHECKPOINT,
    architecture: str | None = None,
    num_labels: int = 2,
    id2label: dict[int, str] | None = None,
    label2id: dict[str, int] | None = None,
    wrap: bool = True,
) -> nn.Module:
    """Load a trained checkpoint for inference.

    Supports either a Hugging Face ``save_pretrained`` directory or the fallback
    ``model.safetensors``/``pytorch_model.bin`` state written by training.
    PyTorch state files are loaded with ``weights_only=True`` and all state keys
    must match the configured model.
    """

    checkpoint_path = Path(checkpoint)
    common = {
        "num_labels": num_labels,
        "id2label": id2label or ID2LABEL,
        "label2id": label2id or LABEL2ID,
        "ignore_mismatched_sizes": True,
    }

    if checkpoint_path.is_dir() and (checkpoint_path / "config.json").exists():
        model = _hf_load(checkpoint_path, architecture=None, **common)
    else:
        model = _hf_load(base_checkpoint, architecture=architecture, **common)
        _load_state_file(model, checkpoint_path)
    return SegmentationModelWrapper(model) if wrap else model


# Compatibility aliases used by inference.py helper discovery.
load_segmentation_model = load_model_for_inference
load_model = load_model_for_inference
