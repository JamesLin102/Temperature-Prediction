"""SVR baseline (scikit-learn). Operates on flattened lag features."""
from __future__ import annotations

import numpy as np
from sklearn.svm import SVR

from ..data.sequence_builder import flatten_sequences
from .base_model import BaseModel


class SVRModel(BaseModel):
    def __init__(self, kernel: str = "rbf", C: float = 1.0, epsilon: float = 0.1, gamma: str = "scale"):
        self._svr = SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "SVRModel":
        self._svr.fit(flatten_sequences(X_train), np.asarray(y_train).reshape(-1))
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self._svr.predict(flatten_sequences(X_test)).astype(np.float32).reshape(-1)
