"""Synthetic-vs-original distribution metrics across the TCP preprocessing stages.

Ported from the former ``scripts/compute_distribution_metrics.py``. For both
datasets x both targets x 4 preprocessing stages, computes KS statistic / p-value,
Jensen–Shannon divergence, Wasserstein distance, and mean/std of the Y column on a
representative fold. Writes ``<outputs>/tables/distribution_metrics.csv``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.svm import OneClassSVM

from ..utils import paths

_DATASETS = ["seattle", "seoul"]
_TARGETS = ["temp_max", "temp_min"]
_NU = 0.05
_STAGE_ORDER = [
    "Before outlier removal",
    "After outlier removal",
    "After standardisation",
    "After Yeo-Johnson transformation",
]


def _compute_stages(data_dir: Path, dataset: str, target: str, fold: int) -> dict:
    base = data_dir / dataset / target
    ori = np.load(base / f"ori_training_data_{fold}.npy")
    syn = np.load(base / f"synthetic_data_{fold}.npy")
    ori_y = ori[:, 5]
    syn_y = syn[:, 5]

    labels = OneClassSVM(nu=_NU).fit_predict(syn)
    filt_y = syn[labels == 1, 5]

    ss = StandardScaler()
    ori_std = ss.fit_transform(ori_y.reshape(-1, 1)).ravel()
    filt_std = ss.transform(filt_y.reshape(-1, 1)).ravel()

    pt = PowerTransformer(method="yeo-johnson")
    ori_yj = pt.fit_transform(ori_std.reshape(-1, 1)).ravel()
    filt_yj = pt.transform(filt_std.reshape(-1, 1)).ravel()

    return {
        "Before outlier removal":           (ori_y, syn_y),
        "After outlier removal":            (ori_y, filt_y),
        "After standardisation":            (ori_std, filt_std),
        "After Yeo-Johnson transformation": (ori_yj, filt_yj),
    }


def _jsd(a, b, n_bins: int = 100) -> float:
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    bins = np.linspace(lo, hi, n_bins + 1)
    p, _ = np.histogram(a, bins=bins)
    q, _ = np.histogram(b, bins=bins)
    p = p.astype(float) + 1e-10
    q = q.astype(float) + 1e-10
    p /= p.sum()
    q /= q.sum()
    return float(jensenshannon(p, q))


def _metrics(ori, syn) -> dict:
    ks_stat, ks_p = ks_2samp(ori, syn)
    return {
        "KS Statistic":         round(float(ks_stat), 4),
        "KS p-value":           round(float(ks_p), 4),
        "JSD":                  round(_jsd(ori, syn), 4),
        "Wasserstein Distance": round(float(wasserstein_distance(ori, syn)), 4),
        "Mean (Original)":      round(float(np.mean(ori)), 4),
        "Mean (Synthetic)":     round(float(np.mean(syn)), 4),
        "Std (Original)":       round(float(np.std(ori)), 4),
        "Std (Synthetic)":      round(float(np.std(syn)), 4),
    }


def compute_distribution_metrics(data_dir=None, *, fold: int = 1) -> pd.DataFrame:
    """Build the distribution-metrics table (one row per dataset/target/stage)."""
    data_dir = paths.DATA_DIR if data_dir is None else paths.resolve(data_dir)
    rows = []
    for dataset in _DATASETS:
        for target in _TARGETS:
            stages = _compute_stages(data_dir, dataset, target, fold)
            for stage in _STAGE_ORDER:
                ori, syn = stages[stage]
                rows.append({
                    "Dataset": dataset.capitalize(),
                    "Target": "Temp Max" if target == "temp_max" else "Temp Min",
                    "Stage": stage,
                    **_metrics(ori, syn),
                })
    return pd.DataFrame(rows)


def write_distribution_metrics(outputs_dir=None, data_dir=None, *, fold: int = 1) -> str:
    """Compute and write ``<outputs>/tables/distribution_metrics.csv``."""
    base = paths.OUTPUTS_DIR if outputs_dir is None else paths.resolve(outputs_dir)
    out_dir = paths.ensure_dir(base / "tables")
    out_path = out_dir / "distribution_metrics.csv"
    df = compute_distribution_metrics(data_dir, fold=fold)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return str(out_path)
