"""YAML configuration inheritance shared by training and analysis."""

import copy
from pathlib import Path

import yaml


def deep_merge_config(base: dict, override: dict) -> dict:
    """Merge nested mappings; replace other values without mutating inputs."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge_config(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(config_path='config.yaml', *, _seen=None) -> dict:
    """Resolve the first path from cwd and each parent from its child file."""
    path = Path(config_path).expanduser().resolve()
    seen = set() if _seen is None else set(_seen)
    if path in seen:
        raise ValueError(f"Circular config extends detected at {path}")
    seen.add(path)
    with path.open() as stream:
        config = yaml.safe_load(stream)
    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise TypeError(f"Configuration must be a mapping: {path}")
    parent = config.pop('extends', None)
    if parent is None:
        return config
    if not isinstance(parent, str) or not parent.strip():
        raise ValueError(f"extends must be a nonempty path string: {path}")
    parent_path = Path(parent).expanduser()
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    return deep_merge_config(load_config(parent_path, _seen=seen), config)
