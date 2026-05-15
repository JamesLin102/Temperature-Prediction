"""Outlier removal for synthetic data (TCN + CTGAN (processed) workflow).

Mirrors Original Programe/model.py: ``OneClassSVM().fit_predict`` on the full
``(N, 6)`` synthetic array, then drop every row labelled ``-1``.
"""
from __future__ import annotations

import numpy as np
from sklearn.svm import OneClassSVM


def remove_outliers(arr_2d: np.ndarray, nu: float = 0.05) -> np.ndarray:
    """Drop rows of ``arr_2d`` flagged as outliers by a One-Class SVM.

    nu controls the upper bound on the fraction of outliers (default 0.05 keeps
    ~95 % of synthetic rows, vs the sklearn default of 0.5 which drops ~50 %).
    """
    arr_2d = np.asarray(arr_2d, dtype=np.float64)
    labels = OneClassSVM(nu=nu).fit_predict(arr_2d)
    outlier_idx = np.where(labels == -1)[0]
    return np.delete(arr_2d, outlier_idx, axis=0)
