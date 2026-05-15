"""Legacy prediction & residual plots, ported from Original Programe/predict.py.

matplotlib uses the non-interactive 'Agg' backend (set before pyplot is imported)
and every function SAVES the figure (the original called ``plt.show()``). Style
is preserved: 3 residual colour bins, thresholds 2.5/5, GridSpec(4,4) residual
layout, 60-bin histograms, metrics textbox.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..evaluation.metrics import compute_all_rounded  # noqa: E402
from ..utils import paths  # noqa: E402

_DEFAULT_THRESHOLDS = (2.5, 5.0)
_DEFAULT_COLORS = {
    "bin1": "royalblue",
    "bin2": "yellowgreen",
    "bin3": "orange",
    "diagonal": "r",
    "hist": "g",
}


def _residual_bins(y: np.ndarray, y_pred: np.ndarray, th_1: float, th_2: float):
    residuals = y_pred - y
    res_abs = np.abs(residuals)
    r1 = np.where(res_abs <= th_1)[0]
    r2 = np.where((res_abs > th_1) & (res_abs <= th_2))[0]
    r3 = np.where(res_abs > th_2)[0]
    return residuals, r1, r2, r3


def _metrics_text(y, y_pred) -> str:
    m = compute_all_rounded(y, y_pred)
    return f"MAE: {m['MAE']}\nRMSE: {m['RMSE']}\nR2 Score: {m['R2']}\n"


def _save(fig, output_path, dpi) -> str:
    out = paths.resolve(output_path)
    paths.ensure_dir(out.parent)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return str(out)


def plot_pred(
    y_true,
    y_pred,
    model_name: str,
    output_path,
    thresholds=_DEFAULT_THRESHOLDS,
    colors=None,
    figsize=(7, 5),
    dpi: int = 150,
) -> str:
    """Scatter of actual vs predicted with 3 residual colour bins (predict.py.plot_pred)."""
    colors = {**_DEFAULT_COLORS, **(colors or {})}
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    th_1, th_2 = thresholds
    _, r1, r2, r3 = _residual_bins(y, yp, th_1, th_2)

    fig = plt.figure(figsize=tuple(figsize))
    ax = fig.add_subplot(111)
    ax.scatter(y[r1], yp[r1], c=colors["bin1"], alpha=0.15, s=40, label=rf"|R|$\leq${th_1}")
    ax.scatter(y[r2], yp[r2], c=colors["bin2"], alpha=0.15, s=40, label=rf"{th_1}<|R|$\leq${th_2}")
    ax.scatter(y[r3], yp[r3], c=colors["bin3"], alpha=0.15, s=40, label=rf"|R|>{th_2}")
    if y.size:
        lo, hi = float(y.min()), float(y.max())
        ax.plot([lo, hi], [lo, hi], linestyle="--", color=colors["diagonal"], lw=1.5)
    ax.set_title(f"{model_name} Prediction Results")
    ax.set_xlabel("Actual Value")
    ax.set_ylabel("Predicted Value")
    ax.text(0.8, 0.05, _metrics_text(y, yp), ha="left", va="center", transform=ax.transAxes)
    ax.legend(loc="upper left")
    ax.grid(True)
    return _save(fig, output_path, dpi)


def plot_residuals(
    y_true,
    y_pred,
    model_name: str,
    output_path,
    thresholds=_DEFAULT_THRESHOLDS,
    colors=None,
    figsize=(7, 5),
    hist_bins: int = 60,
    dpi: int = 150,
) -> str:
    """Residual scatter + marginal histograms (predict.py.plot_residuals)."""
    colors = {**_DEFAULT_COLORS, **(colors or {})}
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    th_1, th_2 = thresholds
    residuals, r1, r2, r3 = _residual_bins(y, yp, th_1, th_2)

    fig = plt.figure(figsize=tuple(figsize))
    text_ax = fig.add_subplot(111)  # holds the metrics textbox (as in the original)
    grid = fig.add_gridspec(4, 4, wspace=0.5, hspace=0.5)

    main_ax = fig.add_subplot(grid[0:3, 1:4])
    main_ax.scatter(yp[r1], residuals[r1], c=colors["bin1"], alpha=0.15, s=40, label=rf"|R|$\leq${th_1}")
    main_ax.scatter(yp[r2], residuals[r2], c=colors["bin2"], alpha=0.15, s=40, label=rf"{th_1}<|R|$\leq${th_2}")
    main_ax.scatter(yp[r3], residuals[r3], c=colors["bin3"], alpha=0.15, s=40, label=rf"|R|>{th_2}")
    if yp.size:
        main_ax.plot([float(yp.min()), float(yp.max())], [0, 0],
                     linestyle="--", color=colors["diagonal"], lw=1.5)
    main_ax.legend(loc="upper left")
    main_ax.grid(True)
    main_ax.set_title(f"Residuals for {model_name} Model")

    y_hist = fig.add_subplot(grid[0:3, 0], sharey=main_ax)
    y_hist.hist(residuals, hist_bins, orientation="horizontal", color=colors["hist"])
    y_hist.invert_xaxis()
    y_hist.set_ylabel("Residuals")
    y_hist.grid(True)

    x_hist = fig.add_subplot(grid[3, 1:4], sharex=main_ax)
    x_hist.hist(yp, hist_bins, orientation="vertical", color=colors["hist"])
    x_hist.invert_yaxis()
    x_hist.set_xlabel("Predicted Value")
    x_hist.grid(True)

    text_ax.text(0.0, 0.05, _metrics_text(y, yp), ha="left", va="center", transform=text_ax.transAxes)
    return _save(fig, output_path, dpi)
