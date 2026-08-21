"""StandardScaler + Yeo-Johnson preprocessing for TCN + CTGAN (processed).

Mirrors the original 2024 implementation: each of the six components
{ori_x, ori_y, gan_x, gan_y, test_x, test_y} is ``StandardScaler.fit_transform``-ed
and then ``PowerTransformer(method='yeo-johnson').fit_transform``-ed, with its own
scaler/transformer pair. Inverse for predictions is Yeo-Johnson inverse then
StandardScaler inverse, using the test-y pair (exact reverse of model.py).

Operates on legacy ``(N, 6)`` fold arrays. The synthetic array passed in should
already have had outliers removed (see ``outlier_filter.remove_outliers``).
"""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import PowerTransformer, StandardScaler

_LOOKBACK = 5


def _scale_then_power(arr: np.ndarray):
    """StandardScaler then Yeo-Johnson; return ``(transformed, scaler, transformer)``."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(arr)
    transformer = PowerTransformer(method="yeo-johnson")
    transformed = transformer.fit_transform(scaled)
    return transformed, scaler, transformer


class ProcessedPreprocessor:
    """6x StandardScaler + 6x Yeo-Johnson, with the exact model.py inverse chain."""

    def __init__(self):
        self._y_test_scaler: StandardScaler | None = None
        self._y_test_transformer: PowerTransformer | None = None

    def fit_transform(self, ori_training_data, testing_data, synthetic_data):
        """Return ``(x_train (N,5,1), y_train (N,1), x_test (N,5,1), y_true (N,1))``."""
        ori = np.asarray(ori_training_data, dtype=np.float64)
        gan = np.asarray(synthetic_data, dtype=np.float64)
        test = np.asarray(testing_data, dtype=np.float64)

        ori_x, ori_y = ori[:, :_LOOKBACK], ori[:, _LOOKBACK].reshape(-1, 1)
        gan_x, gan_y = gan[:, :_LOOKBACK], gan[:, _LOOKBACK].reshape(-1, 1)
        test_x, test_y = test[:, :_LOOKBACK], test[:, _LOOKBACK].reshape(-1, 1)
        y_true = test_y.copy()

        ori_x, _, _ = _scale_then_power(ori_x)
        ori_y, _, _ = _scale_then_power(ori_y)
        gan_x, _, _ = _scale_then_power(gan_x)
        gan_y, _, _ = _scale_then_power(gan_y)
        test_x, _, _ = _scale_then_power(test_x)
        _, self._y_test_scaler, self._y_test_transformer = _scale_then_power(test_y)

        x_train = np.concatenate([ori_x, gan_x], axis=0).reshape(-1, _LOOKBACK, 1)
        y_train = np.concatenate([ori_y, gan_y], axis=0)
        x_test = test_x.reshape(-1, _LOOKBACK, 1)
        return x_train, y_train, x_test, y_true

    def inverse_y(self, y_pred) -> np.ndarray:
        """Inverse: Yeo-Johnson inverse then StandardScaler inverse (test-y pair)."""
        if self._y_test_scaler is None or self._y_test_transformer is None:
            raise RuntimeError("inverse_y called before fit_transform")
        y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1, 1)
        y_pred = self._y_test_transformer.inverse_transform(y_pred)
        y_pred = self._y_test_scaler.inverse_transform(y_pred)
        return y_pred
