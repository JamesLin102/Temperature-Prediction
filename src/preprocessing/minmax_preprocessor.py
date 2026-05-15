"""MinMax preprocessing for the TCN and TCN + CTGAN workflows.

Mirrors Original Programe/model.py exactly, including its per-component scalers:
each of {ori_x, ori_y, [gan_x, gan_y,] test_x, test_y} gets its OWN MinMaxScaler
fit independently (the original even fits a scaler on the test set itself). This
behaviour is preserved on purpose for faithfulness (spec sections 2 and 11).

Operates on legacy ``(N, 6)`` fold arrays: columns 0-4 are the 5 lookback
values, column 5 is the target.
"""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import MinMaxScaler

_LOOKBACK = 5


class MinMaxPreprocessor:
    """Four-scaler (TCN) or six-scaler (TCN + CTGAN) MinMax preprocessing."""

    def __init__(self, mode: str = "four_scaler"):
        if mode not in ("four_scaler", "six_scaler"):
            raise ValueError(f"mode must be 'four_scaler' or 'six_scaler', got {mode!r}")
        self.mode = mode
        self._y_test_scaler: MinMaxScaler | None = None

    def fit_transform(self, ori_training_data, testing_data, synthetic_data=None):
        """Return ``(x_train (N,5,1), y_train (N,1), x_test (N,5,1), y_true (N,1))``.

        ``y_true`` is the un-scaled test target captured before any scaling.
        """
        ori = np.asarray(ori_training_data, dtype=np.float64)
        test = np.asarray(testing_data, dtype=np.float64)

        ori_x, ori_y = ori[:, :_LOOKBACK], ori[:, _LOOKBACK].reshape(-1, 1)
        test_x, test_y = test[:, :_LOOKBACK], test[:, _LOOKBACK].reshape(-1, 1)
        y_true = test_y.copy()

        if self.mode == "six_scaler":
            if synthetic_data is None:
                raise ValueError("six_scaler mode requires synthetic_data")
            gan = np.asarray(synthetic_data, dtype=np.float64)
            gan_x, gan_y = gan[:, :_LOOKBACK], gan[:, _LOOKBACK].reshape(-1, 1)

            ori_x = MinMaxScaler().fit_transform(ori_x)
            ori_y = MinMaxScaler().fit_transform(ori_y)
            gan_x = MinMaxScaler().fit_transform(gan_x)
            gan_y = MinMaxScaler().fit_transform(gan_y)
            test_x = MinMaxScaler().fit_transform(test_x)
            self._y_test_scaler = MinMaxScaler().fit(test_y)

            x_train = np.concatenate([ori_x, gan_x], axis=0)
            y_train = np.concatenate([ori_y, gan_y], axis=0)
        else:  # four_scaler
            x_train = MinMaxScaler().fit_transform(ori_x)
            y_train = MinMaxScaler().fit_transform(ori_y)
            test_x = MinMaxScaler().fit_transform(test_x)
            self._y_test_scaler = MinMaxScaler().fit(test_y)

        x_train = x_train.reshape(x_train.shape[0], _LOOKBACK, 1)
        x_test = test_x.reshape(test_x.shape[0], _LOOKBACK, 1)
        return x_train, y_train, x_test, y_true

    def inverse_y(self, y_pred) -> np.ndarray:
        """Inverse-transform predictions with the test-y scaler (as in model.py)."""
        if self._y_test_scaler is None:
            raise RuntimeError("inverse_y called before fit_transform")
        y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1, 1)
        return self._y_test_scaler.inverse_transform(y_pred)
