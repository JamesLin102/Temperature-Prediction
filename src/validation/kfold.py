"""KFold validation — the validation method used throughout this project.

Mirrors the original ``KFold(n_splits=10)`` with ``shuffle=False``. Used by the
multivariate experiment and the Seoul asset generator; the workflow and baseline
experiments instead read the pre-split ``.npy`` fold files directly by id.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold


def generate_kfold_splits(X, y=None, n_splits: int = 10, shuffle: bool = False, random_state=None):
    """Return a list of ``(train_idx, test_idx)`` arrays.

    ``y`` is accepted for interface symmetry but unused (KFold splits on length).
    """
    X = np.asarray(X)
    kf = KFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state if shuffle else None,
    )
    return [(train_idx, test_idx) for train_idx, test_idx in kf.split(X)]
