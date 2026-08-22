"""Sliding-window sequence construction.

The off-by-one convention ``range(len - lookback - 1)`` matches the original 2024
implementation (``data_gan.py``, git tag ``legacy-2024``) — it is why Seattle
yields 1455 = 1461 - 5 - 1 sequences. Every builder here uses the same
convention, so generated Seoul assets and runtime-built multivariate sequences
share the same ``(N, 6)`` format.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_univariate_sequences(series, lookback: int = 5):
    """Return ``(X, y)`` with ``X`` shape ``(N, lookback, 1)`` and ``y`` shape ``(N,)``."""
    values = np.asarray(series, dtype=np.float64).reshape(-1)
    x_list, y_list = [], []
    for i in range(len(values) - lookback - 1):
        x_list.append(values[i : i + lookback])
        y_list.append(values[i + lookback])
    if not x_list:
        return np.empty((0, lookback, 1), dtype=np.float64), np.empty((0,), dtype=np.float64)
    X = np.array(x_list, dtype=np.float64).reshape(-1, lookback, 1)
    y = np.array(y_list, dtype=np.float64)
    return X, y


def build_multivariate_sequences(df: pd.DataFrame, feature_cols, target_col: str, lookback: int = 5):
    """Return ``(X, y)`` with ``X`` shape ``(N, lookback, n_features)`` and ``y`` shape ``(N,)``."""
    features = df[list(feature_cols)].to_numpy(dtype=np.float64)
    target = df[target_col].to_numpy(dtype=np.float64)
    n_features = features.shape[1]
    x_list, y_list = [], []
    for i in range(len(df) - lookback - 1):
        x_list.append(features[i : i + lookback])
        y_list.append(target[i + lookback])
    if not x_list:
        return np.empty((0, lookback, n_features), dtype=np.float64), np.empty((0,), dtype=np.float64)
    X = np.array(x_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.float64)
    return X, y


def build_station_safe_sequences(
    df: pd.DataFrame,
    target_col: str,
    station_col: str,
    date_col: str,
    lookback: int = 5,
):
    """Build univariate sequences per station (sorted by date), then pool them.

    Sequences never cross stations. Date gaps within a station are tolerated:
    windows are taken over row order after sorting by date, which is how the
    original implementation treated the Seattle series.
    """
    x_parts, y_parts = [], []
    for _, station_df in df.groupby(station_col, sort=False):
        ordered = station_df.sort_values(date_col)
        series = ordered[target_col].to_numpy(dtype=np.float64)
        if len(series) <= lookback + 1:
            continue
        X, y = build_univariate_sequences(series, lookback)
        if len(X) > 0:
            x_parts.append(X)
            y_parts.append(y)
    if not x_parts:
        return np.empty((0, lookback, 1), dtype=np.float64), np.empty((0,), dtype=np.float64)
    return np.concatenate(x_parts, axis=0), np.concatenate(y_parts, axis=0)


def flatten_sequences(X) -> np.ndarray:
    """``(N, lookback, F)`` -> ``(N, lookback * F)`` for sklearn-style models."""
    X = np.asarray(X)
    return X.reshape(X.shape[0], -1)


def to_six_col(X, y) -> np.ndarray:
    """``(N, lookback, 1)`` + ``(N,)`` -> legacy ``(N, lookback + 1)`` array."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    flat = X.reshape(X.shape[0], -1)
    return np.concatenate([flat, y], axis=1)


def from_six_col(arr, lookback: int = 5):
    """Legacy ``(N, lookback + 1)`` array -> ``(X (N, lookback, 1), y (N,))``."""
    arr = np.asarray(arr, dtype=np.float64)
    X = arr[:, :lookback].reshape(-1, lookback, 1)
    y = arr[:, lookback]
    return X, y
