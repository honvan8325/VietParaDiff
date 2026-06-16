"""YAML config loading and dotted-key overrides."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_by_dotted_key(cfg: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    node = cfg
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def parse_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def apply_overrides(cfg: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    out = deepcopy(cfg)
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Override must be KEY=VALUE, got {override}")
        key, value = override.split("=", 1)
        set_by_dotted_key(out, key, parse_value(value))
    return out
