"""Figure-generation orchestrator.

Reproduces the full publication figure set under ``<outputs>/figures/`` plus the
``distribution_metrics.csv`` table, so a single ``run_experiments.py`` run yields
everything. Prediction/residual figures read the aggregated TCP arrays written by
the workflow comparison; temperature and distribution figures read only ``data/``.
"""
from __future__ import annotations

import numpy as np

from ..evaluation.distribution_metrics import write_distribution_metrics
from ..utils import paths
from ..utils.logger import get_logger
from . import distribution_plots, prediction_plots, temperature_plots

# Publication prediction/residual figures cover the TCP (processed) workflow only.
_TCP_SLUG = "TCN_CTGAN_processed"
_TCP_PREFIX = "TCP"


def _load_tcp_arrays(outputs_base, dataset: str, target: str):
    d = outputs_base / "arrays" / dataset / target / _TCP_SLUG
    true_path = d / f"{_TCP_PREFIX}_y_true_all.npy"
    pred_path = d / f"{_TCP_PREFIX}_y_pred_all.npy"
    if not (true_path.exists() and pred_path.exists()):
        return None
    return np.load(true_path), np.load(pred_path)


def generate_all_figures(
    configs: dict,
    *,
    datasets=None,
    targets=None,
    outputs_dir=None,
    logger=None,
) -> list[str]:
    """Generate every publication figure + the distribution metrics table.

    Everything is written under ``outputs_dir`` (defaults to the project
    ``outputs/``): the aggregated TCP arrays are read from ``<outputs>/arrays``,
    figures go to ``<outputs>/figures``, and ``distribution_metrics.csv`` to
    ``<outputs>/tables``. Passing a temp ``outputs_dir`` lets a verification run
    avoid touching real results.
    """
    log = logger or get_logger("figures")
    outputs_base = paths.OUTPUTS_DIR if outputs_dir is None else paths.resolve(outputs_dir)
    figures_base = outputs_base / "figures"

    exp = configs["experiments"]["workflow_comparison"]
    ds_list = datasets or exp["datasets"]
    tg_list = targets or exp["targets"]

    written: list[str] = []

    # 1. Prediction + residual figures (TCP workflow), per dataset/target.
    for dataset in ds_list:
        for target in tg_list:
            arrays = _load_tcp_arrays(outputs_base, dataset, target)
            if arrays is None:
                log.warning(
                    "Skipping prediction figures for %s/%s: TCP arrays not found under %s.",
                    dataset, target, outputs_base / "arrays",
                )
                continue
            y_true, y_pred = arrays
            paths_written = prediction_plots.export_prediction_figures(
                dataset, target, y_true, y_pred, figures_base,
            )
            written += paths_written
            log.info("prediction figures: %s/%s -> %d files", dataset, target, len(paths_written))

    # 2. Temperature-over-time figures (both datasets; read raw CSVs).
    temp_written = temperature_plots.export_temperature_figures(figures_base, paths.DATA_DIR)
    written += temp_written
    log.info("temperature figures -> %d files", len(temp_written))

    # 3. Preprocessing-stage distribution figures (both targets; read fold arrays).
    dist_written = distribution_plots.export_distribution_figures(figures_base, paths.DATA_DIR)
    written += dist_written
    log.info("distribution figures -> %d files", len(dist_written))

    # 4. Distribution-metrics table.
    table_path = write_distribution_metrics(outputs_dir=outputs_base, data_dir=paths.DATA_DIR)
    written.append(table_path)
    log.info("distribution metrics table -> %s", table_path)

    return written
