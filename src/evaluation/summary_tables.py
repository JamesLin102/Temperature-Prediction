"""Summary table construction and export (spec section 8.3).

Each summary row is one (dataset, target, workflow/model[, setting]) combination
with per-fold MAE/RMSE/R2 plus their mean and std across folds. Tables go to
``<outputs>/tables/``; ``outputs_dir`` defaults to the project ``outputs/``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils import paths

_METRICS = ("MAE", "RMSE", "R2")


def summary_row(base_fields: dict, per_fold_metrics: list) -> dict:
    """Build one summary row: base fields + per-metric mean/std across folds."""
    row = dict(base_fields)
    row["n_folds"] = len(per_fold_metrics)
    for metric in _METRICS:
        values = np.array(
            [m[metric] for m in per_fold_metrics if metric in m], dtype=np.float64
        )
        row[f"{metric}_mean"] = float(values.mean()) if values.size else float("nan")
        row[f"{metric}_std"] = float(values.std()) if values.size else float("nan")
    return row


def write_summary(rows: list, filename: str, outputs_dir=None) -> str:
    """Write a list of summary-row dicts to ``<outputs>/tables/<filename>``."""
    base = paths.OUTPUTS_DIR if outputs_dir is None else paths.resolve(outputs_dir)
    out_dir = paths.ensure_dir(base / "tables")
    out_path = out_dir / filename
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return str(out_path)
