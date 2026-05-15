"""Per-fold prediction CSV + aggregated array export (spec sections 8.1, 8.2).

Per-fold CSVs go to ``<outputs>/predictions/<dataset>/<target>/<slug>/`` and
aggregated legacy-named arrays to ``<outputs>/arrays/<dataset>/<target>/<slug>/``.
``outputs_dir`` defaults to the project ``outputs/`` directory; the smoke test
passes a temporary directory so it never clobbers real results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils import paths


def _outputs_base(outputs_dir):
    return paths.OUTPUTS_DIR if outputs_dir is None else paths.resolve(outputs_dir)


def write_fold_predictions(
    fold_id: int,
    dataset: str,
    target: str,
    label_name: str,
    label_value: str,
    y_true,
    y_pred,
    slug: str,
    outputs_dir=None,
) -> str:
    """Write one ``fold_<id>_predictions.csv``.

    ``label_name`` is ``'workflow'`` or ``'model'``; ``label_value`` is its value.
    """
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    frame = pd.DataFrame(
        {
            "fold_id": fold_id,
            "dataset": dataset,
            "target": target,
            label_name: label_value,
            "y_true": y_true,
            "y_pred": y_pred,
            "residual": y_pred - y_true,
        }
    )
    out_dir = paths.ensure_dir(_outputs_base(outputs_dir) / "predictions" / dataset / target / slug)
    out_path = out_dir / f"fold_{fold_id}_predictions.csv"
    frame.to_csv(out_path, index=False)
    return str(out_path)


def save_aggregated_arrays(
    y_true_all,
    y_pred_all,
    dataset: str,
    target: str,
    slug: str,
    prefix: str,
    outputs_dir=None,
) -> dict:
    """Save legacy-named aggregated arrays, e.g. ``T_y_true_all.npy`` / ``T_y_pred_all.npy``.

    ``prefix`` is ``'T'`` | ``'TC'`` | ``'TCP'`` from ``aggregator.legacy_array_prefix``.
    """
    out_dir = paths.ensure_dir(_outputs_base(outputs_dir) / "arrays" / dataset / target / slug)
    true_path = out_dir / f"{prefix}_y_true_all.npy"
    pred_path = out_dir / f"{prefix}_y_pred_all.npy"
    np.save(true_path, np.asarray(y_true_all, dtype=np.float64).reshape(-1, 1))
    np.save(pred_path, np.asarray(y_pred_all, dtype=np.float64).reshape(-1, 1))
    return {"y_true": str(true_path), "y_pred": str(pred_path)}
