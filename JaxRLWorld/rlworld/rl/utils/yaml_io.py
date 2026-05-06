"""YAML I/O utilities for checkpoint serialization."""

import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _sanitize_for_yaml(obj: Any) -> Any:
    """Recursively convert non-YAML-native types to serializable equivalents."""
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_yaml(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_yaml(item) for item in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def dump_yaml(path: str | Path, data: dict) -> None:
    """Serialize *data* to a YAML file at *path*."""
    sanitized = _sanitize_for_yaml(data)
    with open(path, "w") as f:
        yaml.dump(sanitized, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def load_yaml(path: str | Path) -> dict:
    """Load a YAML file and return its contents as a dict."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"YAML file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)
