"""Abstract base class for every forecasting model in revised_framework_v2."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseModel(ABC):
    """Common model interface.

    Input contract: ``X`` is ``(N, lookback, n_features)`` float, ``y`` is
    ``(N,)`` or ``(N, 1)``. Models that need 2-D input (sklearn) or a 1-D series
    (ARIMA) flatten or reshape internally, so callers never branch on model type.
    ``predict`` always returns a 1-D ``(N,)`` array.
    """

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "BaseModel":
        """Train the model and return ``self``."""

    @abstractmethod
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Return 1-D predictions of shape ``(N,)``."""

    def fit_predict(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
        self.fit(X_train, y_train)
        return self.predict(X_test)

    def save_weights(self, path) -> None:  # noqa: D401 - default no-op
        """Persist trained parameters to ``path``. Default implementation: no-op."""
