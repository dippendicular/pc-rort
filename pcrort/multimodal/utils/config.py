"""Configuration loading and merging utilities."""

from __future__ import annotations

import copy
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge `override` into `base`.

    - Dict values are merged recursively.
    - All other values in `override` replace those in `base`.
    - Keys only in `base` are preserved.

    Returns a new dict (neither input is mutated).
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a single YAML file and return its contents as a dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}


def load_config(
    config_paths: Union[str, Path, List[Union[str, Path]]],
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Load and merge multiple YAML config files in order.

    The first file is treated as the base; subsequent files override it.
    An optional `overrides` dict is applied last.

    Args:
        config_paths: Single path or list of paths to YAML config files.
        overrides: Optional dict of programmatic overrides.

    Returns:
        Merged configuration dictionary.
    """
    if isinstance(config_paths, (str, Path)):
        config_paths = [config_paths]

    merged: Dict[str, Any] = {}
    for path in config_paths:
        data = load_yaml(path)
        merged = deep_merge(merged, data)

    if overrides:
        merged = deep_merge(merged, overrides)

    return merged


def save_config(config: Dict[str, Any], path: Union[str, Path]) -> None:
    """Save a config dict as a YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def get_nested(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get a value from a nested dict using dot-separated key path.

    Example:
        get_nested(cfg, "training.learning_rate", 0.001)
    """
    keys = key_path.split(".")
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value


def set_nested(config: Dict[str, Any], key_path: str, value: Any) -> None:
    """
    Set a value in a nested dict using dot-separated key path.

    Creates intermediate dicts as needed.
    """
    keys = key_path.split(".")
    d = config
    for key in keys[:-1]:
        if key not in d or not isinstance(d[key], dict):
            d[key] = {}
        d = d[key]
    d[keys[-1]] = value


def resolve_paths(config: Dict[str, Any], base_dir: Union[str, Path]) -> Dict[str, Any]:
    """
    Resolve relative paths in the config against a base directory.

    Resolves keys ending in '_dir', '_path', or '_csv' that contain
    relative path strings.
    """
    base_dir = Path(base_dir).resolve()
    resolved = copy.deepcopy(config)

    def _resolve(d: Dict[str, Any]) -> None:
        for key, value in d.items():
            if isinstance(value, dict):
                _resolve(value)
            elif isinstance(value, str) and any(
                key.endswith(suffix) for suffix in ("_dir", "_path", "_csv")
            ):
                if value and not os.path.isabs(value):
                    d[key] = str(base_dir / value)

    _resolve(resolved)
    return resolved


def print_config(config: Dict[str, Any], title: str = "Configuration") -> None:
    """Pretty-print a configuration dict."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")
    print(yaml.dump(config, default_flow_style=False, sort_keys=False))
    print(f"{'='*60}\n")


def validate_required_keys(
    config: Dict[str, Any], required_keys: List[str]
) -> List[str]:
    """
    Validate that required keys exist and are non-empty in the config.

    Args:
        config: Configuration dictionary.
        required_keys: List of dot-separated key paths.

    Returns:
        List of missing or empty keys. Empty list means all valid.
    """
    missing = []
    for key_path in required_keys:
        value = get_nested(config, key_path)
        if value is None or value == "":
            missing.append(key_path)
    return missing
