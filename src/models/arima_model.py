"""ARIMA baseline (statsmodels), fixed order, rolling one-step-ahead.

The original program has no ARIMA reference. The fold ``.npy`` rows are
overlapping lookback windows of a single contiguous series (Seattle) or pooled
per-station windows (new_dataset). The training fold's underlying 1-D series is
reconstructed as ``concatenate(X_train[0].flatten(), y_train)`` — exact for a
contiguous KFold train fold, approximate at station boundaries for the pooled
new_dataset (acceptable: new_dataset gets static checks only this pass).

A fixed ``order`` (default ``(5, 1, 0)``) is used rather than an ``auto_arima``
search to keep runs fast and deterministic. If statsmodels fails to fit or
forecast a fold, the model falls back to a persistence forecast (predict the
last observed lookback value) and logs a warning, so a run never crashes.
"""
from __future__ import annotations

import warnings

import numpy as np

from ..utils.logger import get_logger
from .base_model import BaseModel

log = get_logger(__name__)


def _persistence(X_test: np.ndarray) -> np.ndarray:
    """Predict the last lookback value of each window (naive persistence)."""
    arr = np.asarray(X_test, dtype=np.float64)
    last = arr[:, -1, 0] if arr.ndim == 3 else arr[:, -1]
    return last.astype(np.float32).reshape(-1)


class ARIMAModel(BaseModel):
    """Rolling one-step-ahead ARIMA forecaster with a persistence fallback."""

    def __init__(self, order=(5, 1, 0), trend: str = "t"):
        self.order = tuple(order)
        self.trend = trend
        self._results = None
        self._fallback = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "ARIMAModel":
        X_train = np.asarray(X_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)
        full_series = np.concatenate([X_train[0].flatten(), y_train])
        self._fallback = False
        self._results = None
        try:
            from statsmodels.tsa.arima.model import ARIMA

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._results = ARIMA(full_series, order=self.order, trend=self.trend).fit()
        except Exception as exc:  # noqa: BLE001 - fall back rather than crash a run
            log.warning("ARIMA fit failed (%s); using persistence forecast.", exc)
            self._fallback = True
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        X_test = np.asarray(X_test, dtype=np.float64)
        n_steps = len(X_test)
        if self._fallback or self._results is None:
            return _persistence(X_test)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Burn-in: append the first test window's lookback context so the
                # series is positioned just before the first prediction target.
                results = self._results.append(X_test[0].flatten(), refit=False)
                preds = []
                for i in range(n_steps):
                    preds.append(float(np.asarray(results.forecast(1)).ravel()[-1]))
                    if i < n_steps - 1:
                        # X_test[i+1][-1] == y_test[i] in the sliding-window scheme.
                        true_val = float(X_test[i + 1].flatten()[-1])
                        results = results.append([true_val], refit=False)
            return np.array(preds, dtype=np.float32)
        except Exception as exc:  # noqa: BLE001
            log.warning("ARIMA predict failed (%s); using persistence forecast.", exc)
            return _persistence(X_test)
