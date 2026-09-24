"""YAML configuration inheritance shared by training and analysis."""

import copy
from datetime import UTC, datetime
from pathlib import Path

import yaml


def resolve_run_name(config: dict, *, generate: bool = False, now=None) -> str:
    """Resolve the canonical local/tracker run name, accepting legacy run_num."""
    trainer = config.setdefault("trainer", {})
    run_name = trainer.get("run_name")
    legacy_run_num = trainer.get("run_num")

    if run_name is None and legacy_run_num is not None:
        run_name = str(legacy_run_num)
    elif (
        run_name is not None
        and legacy_run_num is not None
        and str(run_name) != str(legacy_run_num)
    ):
        raise ValueError("trainer.run_name and legacy trainer.run_num disagree")

    if run_name is None:
        if not generate:
            raise ValueError(
                "trainer.run_name is required when resuming or reading an existing run"
            )
        timestamp = now or datetime.now(tz=UTC).astimezone()
        run_name = timestamp.strftime("%Y%m%d-%H%M%S")

    run_name = str(run_name).strip()
    if not run_name or run_name in {".", ".."} or "/" in run_name or "\\" in run_name:
        raise ValueError(
            "trainer.run_name must be a non-empty directory-safe name without slashes"
        )
    trainer["run_name"] = run_name
    return run_name


def run_directory(config: dict) -> Path:
    """Return the model directory for an already resolved run name."""
    run_name = resolve_run_name(config, generate=False)
    return Path(config["paths"]["model_save_path"]).expanduser() / f"run{run_name}"


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
