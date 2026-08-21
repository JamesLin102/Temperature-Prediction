"""Canonical path helpers.

Every path resolves relative to the repository root, so the project can be cloned
anywhere and run without configuration. Experiment code never reads files from
outside the repository.
"""
from __future__ import annotations

from pathlib import Path

# paths.py lives at <repo>/src/utils/paths.py -> parents[2] is the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
SEATTLE_DIR = DATA_DIR / "seattle"
SEOUL_DIR = DATA_DIR / "seoul"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_TABLES_DIR = OUTPUTS_DIR / "tables"
OUTPUTS_PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
OUTPUTS_FIGURES_DIR = OUTPUTS_DIR / "figures"  # publication figure set
OUTPUTS_ARRAYS_DIR = OUTPUTS_DIR / "arrays"
OUTPUTS_LOGS_DIR = OUTPUTS_DIR / "logs"


def ensure_dir(path) -> Path:
    """Create ``path`` (and parents) if missing; return it as a ``Path``."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def resolve(relpath) -> Path:
    """Resolve ``relpath`` against ``PROJECT_ROOT``. Absolute paths pass through."""
    p = Path(relpath)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def assert_internal(path) -> Path:
    """Raise if ``path`` is not inside ``PROJECT_ROOT`` (runtime isolation guard)."""
    p = Path(path).resolve()
    root = PROJECT_ROOT.resolve()
    if p != root and root not in p.parents:
        raise ValueError(f"Path escapes the project root: {p}")
    return p
