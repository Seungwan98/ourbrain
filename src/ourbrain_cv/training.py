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
    freeze_backbone_epochs: int = 0
    early_stopping_patience: int = 6
    seed: int = 42
    val_fraction: float = 0.2
    focal_gamma: float = 2.0
    focal_alpha: float = 0.75
    focal_weight: float = 1.0
    dice_weight: float = 1.0
    boundary_weight: float = 0.25
    tversky_weight: float = 0.0
    tversky_alpha: float = 0.7
    tversky_beta: float = 0.3
    cldice_weight: float = 0.0
    cldice_iterations: int = 3
    lr_scheduler: str = "constant"
    warmup_ratio: float = 0.0
    minimum_learning_rate_ratio: float = 0.0
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


def find_backbone_module(model: nn.Module) -> nn.Module | None:
    """Find the feature backbone in wrapped or bare Hugging Face segmentation models."""

    roots = [model]
    wrapped = getattr(model, "model", None)
    if isinstance(wrapped, nn.Module):
        roots.append(wrapped)
    for root in roots:
        for path in ("upernet.backbone", "segformer", "backbone"):
            current: Any = root
            for name in path.split("."):
                current = getattr(current, name, None)
                if current is None:
                    break
            if isinstance(current, nn.Module):
                return current
    return None


def set_module_trainable(module: nn.Module, trainable: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(trainable)


def _build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    name: str,
    total_steps: int,
    warmup_ratio: float,
    minimum_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR | None:
    if name == "constant":
        return None
    if name != "cosine":
        raise ValueError("lr_scheduler must be 'constant' or 'cosine'")
    if total_steps < 1:
        raise ValueError("total scheduler steps must be positive")
    if not 0.0 <= warmup_ratio < 1.0:
        raise ValueError("warmup_ratio must be between 0 (inclusive) and 1 (exclusive)")
    if not 0.0 <= minimum_ratio <= 1.0:
        raise ValueError("minimum_learning_rate_ratio must be between 0 and 1")

    warmup_steps = round(total_steps * warmup_ratio)

    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max((step + 1) / warmup_steps, 1.0 / warmup_steps)
        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum_ratio + (1.0 - minimum_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


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
    run_epochs = cfg.epochs - cfg.initial_epoch
    if not 0 <= cfg.freeze_backbone_epochs <= run_epochs:
        raise ValueError("freeze_backbone_epochs must fit within the configured epoch range")
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
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    if dev.type == "mps":
        enable_mps_compatibility(model)
    criterion = CrackSegmentationLoss(
        LossWeights(
            focal=cfg.focal_weight,
            dice=cfg.dice_weight,
            boundary=cfg.boundary_weight,
            tversky=cfg.tversky_weight,
            cldice=cfg.cldice_weight,
            focal_alpha=cfg.focal_alpha,
            focal_gamma=cfg.focal_gamma,
            tversky_alpha=cfg.tversky_alpha,
            tversky_beta=cfg.tversky_beta,
            cldice_iterations=cfg.cldice_iterations,
        )
    )
    backbone = find_backbone_module(model)
    if cfg.freeze_backbone_epochs and backbone is None:
        raise ValueError("freeze_backbone_epochs requires a model with a discoverable backbone")
    if backbone is not None and cfg.freeze_backbone_epochs:
        set_module_trainable(backbone, False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    optimizer_steps_per_epoch = math.ceil(
        len(train_loader) / cfg.gradient_accumulation_steps
    )
    scheduler = _build_lr_scheduler(
        optimizer,
        name=cfg.lr_scheduler,
        total_steps=optimizer_steps_per_epoch * run_epochs,
        warmup_ratio=cfg.warmup_ratio,
        minimum_ratio=cfg.minimum_learning_rate_ratio,
    )

    use_amp = cfg.mixed_precision and dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    bad_epochs = 0
    best_path: Path | None = None

    for epoch in range(cfg.initial_epoch + 1, cfg.epochs + 1):
        run_epoch = epoch - cfg.initial_epoch
        backbone_frozen = bool(
            backbone is not None and run_epoch <= cfg.freeze_backbone_epochs
        )
        if backbone is not None:
            set_module_trainable(backbone, not backbone_frozen)
        model.train()
        if backbone_frozen and backbone is not None:
            backbone.eval()
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
                if scheduler is not None:
                    scheduler.step()
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
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "backbone_frozen": backbone_frozen,
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
    model_config_path = output / "model_config.json"
    model_config_path.write_text(
        json.dumps(model_config or {}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "history": history,
        "history_path": str(history_path),
        "best_checkpoint": str(best_path) if best_path else None,
        "last_checkpoint": str(last_path) if last_path else None,
        "best_score": best_score,
        "device": str(dev),
        "model_config_path": str(model_config_path),
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(dev)) if dev.type == "cuda" else None
        ),
    }
