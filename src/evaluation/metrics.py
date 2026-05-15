"""Regression metrics: MAE, RMSE, R2 (spec section 8.3)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _flat(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float64).reshape(-1)


def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(_flat(y_true), _flat(y_pred)))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(_flat(y_true), _flat(y_pred))))


def r2(y_true, y_pred) -> float:
    return float(r2_score(_flat(y_true), _flat(y_pred)))


def compute_all(y_true, y_pred) -> dict:
    """Return ``{'MAE', 'RMSE', 'R2'}`` as floats."""
    return {"MAE": mae(y_true, y_pred), "RMSE": rmse(y_true, y_pred), "R2": r2(y_true, y_pred)}


def compute_all_rounded(y_true, y_pred, ndigits: int = 3) -> dict:
    """Rounded metrics matching predict.py.score_calculation (for figure text)."""
    return {k: round(v, ndigits) for k, v in compute_all(y_true, y_pred).items()}
