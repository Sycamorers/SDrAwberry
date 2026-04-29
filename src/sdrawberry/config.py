"""YAML configuration helpers for SDrAwberry."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Load a YAML config file into a dictionary."""
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read configuration files.") from exc

    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def deep_update(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge *updates* into *base* and return *base*."""
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def drop_none(values: Mapping[str, Any]) -> dict[str, Any]:
    """Return a dict without keys whose value is None."""
    return {key: value for key, value in values.items() if value is not None}
