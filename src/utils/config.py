"""YAML config loading. Configs live in ``configs/`` at the repository root."""
from __future__ import annotations

import yaml

from . import paths


def load_yaml(path) -> dict:
    """Load a YAML file, resolving relative paths against the project root."""
    p = paths.resolve(path)
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(name: str) -> dict:
    """Load ``configs/<name>.yaml`` (e.g. ``load_config('datasets')``)."""
    return load_yaml(paths.CONFIGS_DIR / f"{name}.yaml")


def load_all_configs() -> dict:
    """Load every config file into a single dict keyed by stem."""
    return {
        "datasets": load_config("datasets"),
        "models": load_config("models"),
        "experiments": load_config("experiments"),
    }
