"""Shared per-fold engine backing the workflow, baseline and multivariate experiments.

* ``run_fold`` — for the legacy ``.npy``-based experiments (workflow + baseline):
  loads one fold's ``.npy`` arrays, routes through the right preprocessor, fits a
  model, predicts, inverse-transforms.
* ``run_in_memory_fold`` — for the multivariate experiment: data is already split
  in memory; applies 4-scaler-style MinMax generalized to multi-feature input.

Both return an ``aggregator.FoldResult``.

Plain TCN results (four_scaler MinMax, legacy .npy assets) are cached at the
module level so workflow comparison, baseline comparison, and multivariate
univariate all share one training run per fold.
"""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from ..data.dataset_loader import load_legacy_fold
from ..data.sequence_builder import from_six_col
from ..evaluation import metrics
from ..evaluation.aggregator import FoldResult
from ..models.model_factory import build_model, model_kind
from ..preprocessing.minmax_preprocessor import MinMaxPreprocessor
from ..preprocessing.outlier_filter import remove_outliers
from ..preprocessing.processed_preprocessor import ProcessedPreprocessor
from ..utils.seed import derive_fold_seed, set_seed

# workflow name -> (use_ctgan, use_processed)
_WORKFLOW_FLAGS = {
    "TCN": (False, False),
    "TCN + CTGAN": (True, False),
    "TCN + CTGAN (processed)": (True, True),
}

# Cache for plain TCN (four_scaler) results — keyed by (dataset, target, fold_id, cfg_key)
_TCN_FOUR_SCALER_CACHE: dict[tuple, FoldResult] = {}
_TCN_SEED_LABEL = "tcn"  # fixed label used for seeding all plain-TCN folds


def _tcn_cfg_key(model_cfg: dict | None) -> tuple:
    if not model_cfg:
        return ()
    return tuple(sorted(model_cfg.items()))


def _is_plain_tcn(workflow: str | None, model_name: str) -> bool:
    """True when this call is a plain TCN with four_scaler (no CTGAN, no processed)."""
    return (workflow == "TCN") or (workflow is None and model_name == "tcn")


def workflow_flags(workflow: str) -> tuple[bool, bool]:
    """Return ``(use_ctgan, use_processed)`` for a workflow name."""
    if workflow not in _WORKFLOW_FLAGS:
        raise KeyError(f"Unknown workflow {workflow!r}")
    return _WORKFLOW_FLAGS[workflow]


def run_fold(
    dataset: str,
    target: str,
    fold_id: int,
    *,
    workflow: str | None = None,
    model_name: str = "tcn",
    model_cfg: dict | None = None,
    seed: int = 7,
) -> FoldResult:
    """Run one fold from pre-split legacy ``.npy`` assets.

    If ``workflow`` is given (workflow comparison), the model is always TCN and
    preprocessing follows the workflow. Otherwise (baseline comparison) the model
    is ``model_name`` and preprocessing follows its kind (nn / sklearn / arima).

    Plain TCN (four_scaler MinMax) results are cached so workflow, baseline, and
    multivariate-univariate share one training run per fold.
    """
    # --- plain TCN cache check ---
    if _is_plain_tcn(workflow, model_name):
        cache_key = (dataset, target, fold_id, _tcn_cfg_key(model_cfg))
        if cache_key in _TCN_FOUR_SCALER_CACHE:
            return _TCN_FOUR_SCALER_CACHE[cache_key]
        # Use fixed seed label so all callers reproduce the same weights
        set_seed(derive_fold_seed(seed, dataset, target, _TCN_SEED_LABEL, fold_id))
        ori = load_legacy_fold(dataset, target, fold_id, "ori_training")
        test = load_legacy_fold(dataset, target, fold_id, "testing")
        pre = MinMaxPreprocessor("four_scaler")
        x_train, y_train, x_test, y_true = pre.fit_transform(ori, test)
        model = build_model("tcn", model_cfg)
        model.fit(x_train, y_train)
        y_pred = pre.inverse_y(model.predict(x_test))
        y_true = np.asarray(y_true, dtype=np.float64).reshape(-1, 1)
        y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1, 1)
        result = FoldResult(fold_id=fold_id, y_true=y_true, y_pred=y_pred,
                            metrics=metrics.compute_all(y_true, y_pred))
        _TCN_FOUR_SCALER_CACHE[cache_key] = result
        return result

    # --- all other models / workflows ---
    label = workflow if workflow is not None else model_name
    set_seed(derive_fold_seed(seed, dataset, target, label, fold_id))

    ori = load_legacy_fold(dataset, target, fold_id, "ori_training")
    test = load_legacy_fold(dataset, target, fold_id, "testing")

    if workflow is not None:
        use_ctgan, use_processed = workflow_flags(workflow)
        if use_processed:
            synthetic = remove_outliers(load_legacy_fold(dataset, target, fold_id, "synthetic"))
            pre = ProcessedPreprocessor()
            x_train, y_train, x_test, y_true = pre.fit_transform(ori, test, synthetic)
        elif use_ctgan:
            synthetic = load_legacy_fold(dataset, target, fold_id, "synthetic")
            pre = MinMaxPreprocessor("six_scaler")
            x_train, y_train, x_test, y_true = pre.fit_transform(ori, test, synthetic)
        else:
            pre = MinMaxPreprocessor("four_scaler")
            x_train, y_train, x_test, y_true = pre.fit_transform(ori, test)

        model = build_model("tcn", model_cfg)
        model.fit(x_train, y_train)
        y_pred = pre.inverse_y(model.predict(x_test))

    else:
        kind = model_kind(model_name)
        if kind == "nn":
            pre = MinMaxPreprocessor("four_scaler")
            x_train, y_train, x_test, y_true = pre.fit_transform(ori, test)
            model = build_model(model_name, model_cfg)
            model.fit(x_train, y_train)
            y_pred = pre.inverse_y(model.predict(x_test))
        else:  # sklearn / arima: raw fold values, no scaling, no inverse
            x_train, y_train = from_six_col(ori)
            x_test, y_test = from_six_col(test)
            y_true = y_test.reshape(-1, 1)
            model = build_model(model_name, model_cfg)
            model.fit(x_train, y_train)
            y_pred = np.asarray(model.predict(x_test), dtype=np.float64).reshape(-1, 1)

    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1, 1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1, 1)
    return FoldResult(
        fold_id=fold_id,
        y_true=y_true,
        y_pred=y_pred,
        metrics=metrics.compute_all(y_true, y_pred),
    )


def run_in_memory_fold(
    fold_id: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    model_name: str = "tcn",
    model_cfg: dict | None = None,
    seed: int = 7,
    seed_parts: tuple = (),
) -> FoldResult:
    """Run one fold from already-split in-memory sequences (multivariate experiment).

    ``x_*`` are ``(N, lookback, n_features)``; ``y_*`` are ``(N,)``. Applies
    4-scaler-style MinMax generalized to multi-feature input — one scaler each on
    train X, train y, test X, test y, every scaler fit independently (matching
    the original's per-split independent fit).
    """
    set_seed(derive_fold_seed(seed, *seed_parts, fold_id))

    x_train = np.asarray(x_train, dtype=np.float64)
    x_test = np.asarray(x_test, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64).reshape(-1, 1)
    y_true = np.asarray(y_test, dtype=np.float64).reshape(-1, 1)

    n_tr, lookback, n_feat = x_train.shape
    n_te = x_test.shape[0]

    x_train_s = MinMaxScaler().fit_transform(x_train.reshape(n_tr, -1)).reshape(n_tr, lookback, n_feat)
    y_train_s = MinMaxScaler().fit_transform(y_train)
    x_test_s = MinMaxScaler().fit_transform(x_test.reshape(n_te, -1)).reshape(n_te, lookback, n_feat)
    y_test_scaler = MinMaxScaler().fit(y_true)

    model = build_model(model_name, model_cfg)
    model.fit(x_train_s, y_train_s)
    y_pred = y_test_scaler.inverse_transform(
        np.asarray(model.predict(x_test_s), dtype=np.float64).reshape(-1, 1)
    )

    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1, 1)
    return FoldResult(
        fold_id=fold_id,
        y_true=y_true,
        y_pred=y_pred,
        metrics=metrics.compute_all(y_true, y_pred),
    )
