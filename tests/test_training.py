from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import Dataset

from ourbrain_cv.training import freeze_batch_norm_stats, group_train_val_split, train_model


class TinyDataset(Dataset):
    def __init__(self):
        self.groups = ["a", "a", "b", "b"]

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        labels = torch.zeros(8, 8, dtype=torch.long)
        if idx % 2 == 0:
            labels[2, 2:5] = 1
        return {
            "pixel_values": torch.randn(3, 8, 8),
            "labels": labels,
            "group": self.groups[idx],
        }


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 2, kernel_size=1)

    def forward(self, pixel_values, labels=None):
        # Intentionally low-res to exercise loss/metric-compatible logits in training only
        return SimpleNamespace(logits=self.conv(pixel_values))


def test_freeze_batch_norm_stats_keeps_affine_trainable():
    model = nn.Sequential(nn.BatchNorm2d(3))
    model.train()
    freeze_batch_norm_stats(model)
    batch_norm = model[0]
    assert not batch_norm.training
    assert batch_norm.weight.requires_grad


def test_group_split_keeps_groups_disjoint():
    ds = TinyDataset()
    train_idx, val_idx = group_train_val_split(ds, val_fraction=0.5, seed=1)
    train_groups = {ds.groups[i] for i in train_idx}
    val_groups = {ds.groups[i] for i in val_idx}
    assert train_groups.isdisjoint(val_groups)
    assert train_idx and val_idx


def test_train_model_with_dummy_model_writes_history(tmp_path):
    result = train_model(
        TinyDataset(),
        model=TinyModel(),
        config={
            "output_dir": str(tmp_path),
            "epochs": 1,
            "batch_size": 2,
            "gradient_accumulation_steps": 1,
            "mixed_precision": True,
            "early_stopping_patience": 2,
            "save_safetensors": False,
        },
        device="cpu",
    )
    assert result["history"]
    assert (tmp_path / "history.json").exists()
    assert (tmp_path / "pytorch_model.bin").exists()
