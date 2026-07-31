"""Stable hashing and atomic JSON helpers for model artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_sha256(checkpoint: str | Path) -> str:
    root = Path(checkpoint).expanduser().resolve()
    if root.is_file():
        files = [root]
        relative_to = root.parent
    elif root.is_dir():
        files = sorted(
            {
                *root.glob("*.safetensors"),
                *root.glob("*.bin"),
                *root.glob("*.json"),
            }
        )
        relative_to = root
    else:
        raise ValueError(f"checkpoint does not exist: {root}")
    if not files:
        raise ValueError(f"checkpoint has no model/config files: {root}")

    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(relative_to).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(output)
    return output


__all__ = ["checkpoint_sha256", "sha256_file", "write_json_atomic"]
