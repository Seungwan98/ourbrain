"""Configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a configuration file is incomplete or invalid."""


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file and validate required sections."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise ConfigError(f"Configuration must be a mapping: {config_path}")

    required_sections = {"model", "data", "training", "inference"}
    missing = required_sections.difference(raw)
    if missing:
        raise ConfigError(f"Missing configuration sections: {', '.join(sorted(missing))}")

    return raw

