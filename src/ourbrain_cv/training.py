"""Small, dependency-light training loop for crack segmentation."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

from .losses import CrackSegmentationLoss, LossWeights
from .metrics import compute_segmentation_metrics
from .modeling import build_model, enable_mps_compatibility


class BatchDataset(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> dict[str, Any]: ...


@dataclass
class TrainingConfig:
    output_dir: str = "checkpoints/upernet-swin-tiny"
    epochs: int = 30
    initial_epoch: int = 0
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 6e-5
    weight_decay: float = 0.01
    num_workers: int = 0
    mixed_precision: bool = True
    freeze_batch_norm: bool = True
    early_stopping_patience: int = 6
    seed: int = 42
    val_fraction: float = 0.2
    focal_gamma: float = 2.0
    focal_weight: float = 1.0
    dice_weight: float = 1.0
    boundary_weight: float = 0.25
    monitor: str = "crack_dice"
    save_safetensors: bool = True
    save_last_checkpoint: bool = False
    max_train_samples: int | None = None
    max_val_samples: int | None = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def freeze_batch_norm_stats(model: nn.Module) -> None:
    """Keep pretrained BatchNorm statistics during batch-size-one training.

    UPerNet's pyramid pooling branch creates 1x1 feature maps. BatchNorm cannot
    estimate statistics from one value, so its running statistics stay in
    evaluation mode while affine parameters remain trainable.
    """
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def group_train_val_split(
    dataset: Dataset[Any],
    *,
    val_fraction: float = 0.2,
    seed: int = 42,
    group_key: str = "group",
) -> tuple[list[int], list[int]]:
    """Split indices by group to prevent same source image leaking into validation."""

    groups: dict[Any, list[int]] = {}
    for idx in range(len(dataset)):
        item = dataset[idx]
        if isinstance(item, dict):
            group = item.get(group_key, item.get("group_id", item.get("source_id", idx)))
        else:
            group = idx
        groups.setdefault(group, []).append(idx)

    rng = random.Random(seed)
    keys = list(groups)
    rng.shuffle(keys)
    val_group_count = max(1, math.ceil(len(keys) * val_fraction)) if len(keys) > 1 else 0
    val_keys = set(keys[:val_group_count])
    train_idx: list[int] = []
    val_idx: list[int] = []
    for key in keys:
        (val_idx if key in val_keys else train_idx).extend(groups[key])
    if not train_idx and val_idx:
        train_idx.append(val_idx.pop())
    return sorted(train_idx), sorted(val_idx)


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    keys = batch[0].keys()
    out: dict[str, Any] = {}
    for key in keys:
        vals = [b[key] for b in batch]
        if torch.is_tensor(vals[0]):
            out[key] = torch.stack(vals)
        else:
            out[key] = vals
    return out


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def _forward_logits(model: nn.Module, batch: dict[str, Any]) -> torch.Tensor:
    labels = batch.get("labels")
    outputs = model(pixel_values=batch["pixel_values"], labels=labels)
    return outputs.logits if hasattr(outputs, "logits") else outputs["logits"]


def _mean_metrics(items: list[dict[str, Any]]) -> dict[str, float]:
    if not items:
        return {}
    keys = items[0].keys()
    return {
        k: float(np.mean([m[k] for m in items if isinstance(m.get(k), (int, float))]))
        for k in keys
    }


def save_checkpoint(model: nn.Module, output_dir: str | Path, *, safetensors: bool = True) -> Path:
    """Save a reloadable checkpoint, preferring Hugging Face save_pretrained."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "save_pretrained"):
        try:
            model.save_pretrained(output, safe_serialization=safetensors)
            return output
        except TypeError:
            model.save_pretrained(output)
            return output
        except Exception:
            # Custom/dummy models in tests may expose incomplete save_pretrained.
            pass

    state = model.state_dict()
    if safetensors:
        try:
            from safetensors.torch import save_file

            path = output / "model.safetensors"
            save_file(state, str(path))
            return path
        except Exception:
            # Fall back to PyTorch for environments without safetensors support.
            pass
    path = output / "pytorch_model.bin"
    torch.save(state, path)
    return path


def train_model(
    train_dataset: Dataset[Any],
    val_dataset: Dataset[Any] | None = None,
    *,
    model: nn.Module | None = None,
    model_config: dict[str, Any] | None = None,
    config: TrainingConfig | dict[str, Any] | None = None,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Train and validate a segmentation model using batch dicts.

    Dataset items must include ``pixel_values`` and ``labels`` tensors. If no
    validation dataset is supplied, a group-aware split is created from
    ``train_dataset`` using ``group`` or ``source_id`` metadata when present.
    """

    cfg = TrainingConfig(**config) if isinstance(config, dict) else (config or TrainingConfig())
    if cfg.initial_epoch < 0:
        raise ValueError("initial_epoch must be non-negative")
    if cfg.initial_epoch >= cfg.epochs:
        raise ValueError("initial_epoch must be smaller than epochs")
    set_seed(cfg.seed)
    dev = torch.device(device) if device is not None else auto_device()

    if val_dataset is None:
        train_idx, val_idx = group_train_val_split(
            train_dataset, val_fraction=cfg.val_fraction, seed=cfg.seed
        )
        val_dataset = Subset(train_dataset, val_idx) if val_idx else None
        train_dataset = Subset(train_dataset, train_idx)

    if cfg.max_train_samples is not None:
        train_dataset = Subset(
            train_dataset,
            range(min(len(train_dataset), max(0, cfg.max_train_samples))),
        )
    if val_dataset is not None and cfg.max_val_samples is not None:
        val_dataset = Subset(
            val_dataset,
            range(min(len(val_dataset), max(0, cfg.max_val_samples))),
        )
    if len(train_dataset) == 0:
        raise ValueError("training dataset is empty after sample limits")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=_collate,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=_collate,
        )
        if val_dataset is not None and len(val_dataset) > 0
        else None
    )

    model = model or build_model(model_config)
    model.to(dev)
    if dev.type == "mps":
        enable_mps_compatibility(model)
    criterion = CrackSegmentationLoss(
        LossWeights(
            focal=cfg.focal_weight,
            dice=cfg.dice_weight,
            boundary=cfg.boundary_weight,
            focal_gamma=cfg.focal_gamma,
        )
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    use_amp = cfg.mixed_precision and dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    bad_epochs = 0
    best_path: Path | None = None

    for epoch in range(cfg.initial_epoch + 1, cfg.epochs + 1):
        model.train()
        if cfg.freeze_batch_norm:
            freeze_batch_norm_stats(model)
        optimizer.zero_grad(set_to_none=True)
        train_losses: list[float] = []
        for step, batch in enumerate(
            tqdm(train_loader, desc=f"epoch {epoch}/{cfg.epochs}", leave=False), start=1
        ):
            batch = _move_batch(batch, dev)
            with torch.autocast(device_type=dev.type, enabled=use_amp):
                logits = _forward_logits(model, batch)
                loss = criterion(logits, batch["labels"]) / cfg.gradient_accumulation_steps
            scaler.scale(loss).backward()
            if step % cfg.gradient_accumulation_steps == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            train_losses.append(float(loss.detach().cpu()) * cfg.gradient_accumulation_steps)

        val_metrics: dict[str, float] = {}
        val_loss = None
        if val_loader is not None:
            model.eval()
            metrics_rows: list[dict[str, Any]] = []
            val_losses: list[float] = []
            with torch.no_grad():
                for batch in val_loader:
                    batch = _move_batch(batch, dev)
                    logits = _forward_logits(model, batch)
                    val_losses.append(float(criterion(logits, batch["labels"]).detach().cpu()))
                    metrics_rows.append(
                        compute_segmentation_metrics(
                            logits.detach().cpu(), batch["labels"].detach().cpu()
                        )
                    )
            val_metrics = _mean_metrics(metrics_rows)
            val_loss = float(np.mean(val_losses)) if val_losses else None

        score = val_metrics.get(cfg.monitor, -float(np.mean(train_losses)))
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)) if train_losses else None,
            "val_loss": val_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(record)

        if score > best_score:
            best_score = score
            bad_epochs = 0
            best_path = save_checkpoint(model, cfg.output_dir, safetensors=cfg.save_safetensors)
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.early_stopping_patience:
                break

    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    last_path = (
        save_checkpoint(model, output / "last", safetensors=cfg.save_safetensors)
        if cfg.save_last_checkpoint
        else None
    )
    history_path = output / "history.json"
    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "training_config.json").write_text(
        json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "history": history,
        "history_path": str(history_path),
        "best_checkpoint": str(best_path) if best_path else None,
        "last_checkpoint": str(last_path) if last_path else None,
        "best_score": best_score,
        "device": str(dev),
    }
