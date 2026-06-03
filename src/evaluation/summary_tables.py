"""Summary table construction and export (spec section 8.3).

Each summary row is one (dataset, target, workflow/model[, setting]) combination
with **pooled** MAE/RMSE/R2 — metrics computed once over every fold's predictions
concatenated together, not the mean of per-fold metrics. This matches the values
annotated on the prediction figures. Tables go to ``<outputs>/tables/``;
``outputs_dir`` defaults to the project ``outputs/``.
"""
from __future__ import annotations

import pandas as pd

from ..utils import paths

_METRICS = ("MAE", "RMSE", "R2")


def summary_row(base_fields: dict, aggregated) -> dict:
    """Build one summary row: base fields + pooled MAE/RMSE/R2 across all folds.

    ``aggregated`` is an ``aggregator.AggregatedResult`` whose ``overall_metrics``
    are computed on the concatenated ``y_true_all`` / ``y_pred_all`` (the 10-fold
    pooled aggregation), so each metric is a single value, not a mean ± std.
    """
    row = dict(base_fields)
    row["n_folds"] = len(aggregated.per_fold_metrics)
    row["n_samples"] = int(aggregated.y_true_all.shape[0])
    for metric in _METRICS:
        row[metric] = float(aggregated.overall_metrics[metric])
    return row


def write_summary(rows: list, filename: str, outputs_dir=None) -> str:
    """Write a list of summary-row dicts to ``<outputs>/tables/<filename>``."""
    base = paths.OUTPUTS_DIR if outputs_dir is None else paths.resolve(outputs_dir)
    out_dir = paths.ensure_dir(base / "tables")
    out_path = out_dir / filename
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return str(out_path)
