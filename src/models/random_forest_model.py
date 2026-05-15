"""Random Forest baseline (scikit-learn). Operates on flattened lag features."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from ..data.sequence_builder import flatten_sequences
from .base_model import BaseModel


class RandomForestModel(BaseModel):
    def __init__(self, n_estimators: int = 100, max_depth: int | None = None, random_state: int = 7):
        self._rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "RandomForestModel":
        self._rf.fit(flatten_sequences(X_train), np.asarray(y_train).reshape(-1))
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self._rf.predict(flatten_sequences(X_test)).astype(np.float32).reshape(-1)
