"""Data access layer. Reads ONLY from inside the repository.

Workflow / baseline experiments consume the pre-split ``.npy`` fold files;
the multivariate experiment and ``scripts/prepare_data.py`` read the raw CSVs.
Neither the CSVs nor the ``.npy`` assets ship with the repository — see
``data/README.md`` for how to obtain and build them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..utils import paths

FOLD_KINDS = ("ori_training", "synthetic", "testing")


# --- raw CSVs ---------------------------------------------------------------
def load_seattle_csv() -> pd.DataFrame:
    """Load ``data/seattle/seattle-weather.csv`` (downloaded by the user)."""
    path = paths.assert_internal(paths.SEATTLE_DIR / "seattle-weather.csv")
    if not path.exists():
        raise FileNotFoundError(
            f"Seattle CSV not found: {path}\n"
            "The datasets are not redistributed with this repository; see data/README.md."
        )
    return pd.read_csv(path)


def load_seoul_csv() -> pd.DataFrame:
    """Load ``data/seoul/Bias_correction_ucl.csv`` (downloaded by the user)."""
    path = paths.assert_internal(paths.SEOUL_DIR / "Bias_correction_ucl.csv")
    if not path.exists():
        raise FileNotFoundError(
            f"Seoul CSV not found: {path}\n"
            "The datasets are not redistributed with this repository; see data/README.md."
        )
    return pd.read_csv(path)


# --- legacy KFold fold assets ----------------------------------------------
def _fold_dir(dataset: str, target: str) -> Path:
    if dataset == "seattle":
        return paths.SEATTLE_DIR / target
    if dataset == "seoul":
        return paths.SEOUL_DIR / target
    raise ValueError(f"Unknown dataset: {dataset}")


def legacy_fold_path(dataset: str, target: str, fold_id: int, kind: str) -> Path:
    """Path of a single ``<kind>_data_<fold_id>.npy`` fold file."""
    if kind not in FOLD_KINDS:
        raise ValueError(f"kind must be one of {FOLD_KINDS}, got {kind!r}")
    return _fold_dir(dataset, target) / f"{kind}_data_{fold_id}.npy"


def load_legacy_fold(dataset: str, target: str, fold_id: int, kind: str) -> np.ndarray:
    """Load one ``(N, 6)`` fold array (cols 0-4 lookback values, col 5 target)."""
    path = paths.assert_internal(legacy_fold_path(dataset, target, fold_id, kind))
    if not path.exists():
        raise FileNotFoundError(
            f"Fold asset not found: {path}\n"
            "Download the raw CSVs and build the folds first:\n"
            "    python scripts/prepare_data.py --dataset all\n"
            "See data/README.md."
        )
    return np.load(path)


def fold_assets_available(
    dataset: str,
    target: str,
    n_splits: int = 10,
    kinds: tuple[str, ...] = ("ori_training", "testing"),
) -> bool:
    """True only if every requested fold ``.npy`` exists for ``(dataset, target)``.

    This is the graceful-skip gate: ``seoul`` may have no assets yet, so
    runners and the smoke test call this first and skip it instead of crashing.
    """
    for fold_id in range(1, n_splits + 1):
        for kind in kinds:
            if not legacy_fold_path(dataset, target, fold_id, kind).exists():
                return False
    return True

