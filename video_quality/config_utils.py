from __future__ import annotations

import os
from pathlib import Path

import yaml


def _expand_value(value, base_dir: Path):
    if isinstance(value, dict):
        return {k: _expand_value(v, base_dir) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_value(v, base_dir) for v in value]
    if not isinstance(value, str):
        return value

    expanded = os.path.expandvars(os.path.expanduser(value))
    if expanded.startswith(("http://", "https://")):
        return expanded
    if expanded.startswith("./") or expanded.startswith("../"):
        return str((base_dir / expanded).resolve())
    return expanded


def load_yaml_config(config_path: str):
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return _expand_value(config, path.parent)
