"""LightGBM baseline. Operates on flattened lag features."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..data.sequence_builder import flatten_sequences
from .base_model import BaseModel


def _as_frame(X_flat: np.ndarray) -> pd.DataFrame:
    """Wrap flattened features in a consistently-named DataFrame.

    LightGBM 4.x auto-assigns feature names ('Column_0', ...) at fit time even
    for plain arrays; passing a matching DataFrame to both fit and predict
    avoids the sklearn 'X does not have valid feature names' warning.
    """
    return pd.DataFrame(X_flat, columns=[f"Column_{i}" for i in range(X_flat.shape[1])])


class LightGBMModel(BaseModel):
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = -1,
        num_leaves: int = 31,
        random_state: int = 7,
        verbose: int = -1,
    ):
        self._model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            random_state=random_state,
            verbose=verbose,
            n_jobs=-1,
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "LightGBMModel":
        self._model.fit(_as_frame(flatten_sequences(X_train)), np.asarray(y_train).reshape(-1))
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self._model.predict(_as_frame(flatten_sequences(X_test))).astype(np.float32).reshape(-1)
