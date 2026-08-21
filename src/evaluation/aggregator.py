"""Fold aggregation: stack per-fold predictions into legacy ``(N, 1)`` arrays."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import metrics

# Workflow name -> aggregated-array filename prefix.
_WORKFLOW_PREFIX = {
    "TCN": "T",
    "TCN + CTGAN": "TC",
    "TCN + CTGAN (processed)": "TCP",
}


def legacy_array_prefix(workflow: str) -> str:
    """Return ``'T'`` | ``'TC'`` | ``'TCP'`` for a workflow name."""
    if workflow not in _WORKFLOW_PREFIX:
        raise KeyError(f"No legacy prefix for workflow {workflow!r}")
    return _WORKFLOW_PREFIX[workflow]


@dataclass
class FoldResult:
    """One fold's predictions and metrics (returned by fold_runner.run_fold)."""

    fold_id: int
    y_true: np.ndarray  # (N, 1)
    y_pred: np.ndarray  # (N, 1)
    metrics: dict = field(default_factory=dict)


@dataclass
class AggregatedResult:
    """Concatenated predictions across folds plus per-fold and overall metrics."""

    y_true_all: np.ndarray  # (sum N, 1)
    y_pred_all: np.ndarray  # (sum N, 1)
    per_fold_metrics: list  # list[dict], one per fold, each with fold_id + MAE/RMSE/R2
    overall_metrics: dict


def aggregate_folds(fold_results) -> AggregatedResult:
    """Stack fold predictions into ``(sum N, 1)`` arrays.

    Uses a clean ``np.vstack`` of the real folds — value-identical to the
    original's ``(1, 1)``-zero-placeholder-then-``[1:]`` trick, without the
    placeholder row that would otherwise corrupt metrics.
    """
    ordered = sorted(fold_results, key=lambda r: r.fold_id)
    if not ordered:
        raise ValueError("aggregate_folds received no fold results")

    y_true_all = np.vstack(
        [np.asarray(r.y_true, dtype=np.float64).reshape(-1, 1) for r in ordered]
    )
    y_pred_all = np.vstack(
        [np.asarray(r.y_pred, dtype=np.float64).reshape(-1, 1) for r in ordered]
    )
    per_fold = [
        {"fold_id": r.fold_id, **metrics.compute_all(r.y_true, r.y_pred)} for r in ordered
    ]
    overall = metrics.compute_all(y_true_all, y_pred_all)
    return AggregatedResult(y_true_all, y_pred_all, per_fold, overall)
