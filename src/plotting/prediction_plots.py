"""Publication prediction & residual figures for the TCN + CTGAN (processed) workflow.

Ported verbatim (style-wise) from the former ``scripts/plot_tcn_ctgan_processed.py``.
For each dataset x target it draws

  * a prediction scatter (actual vs predicted, 3 residual colour bins), and
  * a residual scatter with marginal histograms,

both annotated with **pooled** MAE / RMSE / R2 — metrics computed once over the
full 10-fold aggregated array (``outputs/arrays/.../TCP_y_*_all.npy``), matching
the pooled values written to ``workflow_comparison_summary.csv``.

Style: Times New Roman, STIX math. Saves PNG + PDF to
``<figures>/predictions/{dataset}/t{max,min}_{pred,resi}.{png,pdf}``.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.gridspec as gridspec  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..evaluation.metrics import compute_all  # noqa: E402

# ── Typography (mirrors the temperature / distribution figures) ───────────── #
_RC = {
    "font.family":         "Times New Roman",
    "mathtext.fontset":    "stix",
    "axes.linewidth":      0.75,
    "xtick.direction":     "out",
    "ytick.direction":     "out",
    "xtick.major.size":    4.0,
    "ytick.major.size":    4.0,
    "xtick.minor.size":    2.0,
    "ytick.minor.size":    2.0,
    "xtick.major.width":   0.75,
    "ytick.major.width":   0.75,
    "xtick.labelsize":     9,
    "ytick.labelsize":     9,
    "axes.labelsize":      10,
    "legend.fontsize":     8.5,
    "legend.frameon":      True,
    "legend.framealpha":   0.92,
    "legend.edgecolor":    "#CCCCCC",
    "legend.handlelength": 1.4,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
}

# Residual colour-bin thresholds and palette.
TH1, TH2 = 2.5, 5.0
C_BIN1 = "royalblue"
C_BIN2 = "yellowgreen"
C_BIN3 = "orange"
C_DIAG = "#B03A2E"
C_HIST = "#1A5276"
ALPHA = 0.18
S_DOT = 28
DPI = 200

DATASET_LABEL = {"seattle": "Seattle", "seoul": "Seoul"}
TARGET_LABEL = {"temp_max": "Max Temperature (°C)", "temp_min": "Min Temperature (°C)"}
TARGET_SUFFIX = {"temp_max": "tmax", "temp_min": "tmin"}


def pooled_metrics(y_true, y_pred) -> tuple[float, float, float]:
    """Pooled MAE, RMSE, R2 over the full aggregated array, rounded to 3 d.p."""
    m = compute_all(y_true, y_pred)
    return round(m["MAE"], 3), round(m["RMSE"], 3), round(m["R2"], 3)


def _residual_bins(y, yp):
    r = yp - y
    ra = np.abs(r)
    return r, (ra <= TH1), ((ra > TH1) & (ra <= TH2)), (ra > TH2)


def _metrics_box(mae, rmse, r2) -> str:
    return f"MAE:  {mae:.3f}\nRMSE: {rmse:.3f}\n$R^2$:    {r2:.3f}"


def _grid_style(ax, x=True, y=True):
    if y:
        ax.yaxis.grid(True, linestyle="--", linewidth=0.40, alpha=0.55, color="#AAAAAA")
    if x:
        ax.xaxis.grid(True, linestyle="--", linewidth=0.40, alpha=0.55, color="#AAAAAA")
    ax.set_axisbelow(True)


def _save(fig, stem: Path, dpi: int) -> list[str]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in ("png", "pdf"):
        p = stem.with_suffix(f".{ext}")
        # Drop the PDF creation timestamp so re-runs are byte-reproducible.
        meta = {"CreationDate": None} if ext == "pdf" else None
        fig.savefig(p, dpi=dpi, bbox_inches="tight", metadata=meta)
        written.append(str(p))
    plt.close(fig)
    return written


def _plot_pred(y, yp, dataset, target, mae, rmse, r2, stem: Path, dpi: int) -> list[str]:
    _, m1, m2, m3 = _residual_bins(y, yp)

    fig, ax = plt.subplots(
        figsize=(6.2, 5.0),
        gridspec_kw=dict(left=0.13, right=0.96, top=0.91, bottom=0.12),
    )

    kw = dict(alpha=ALPHA, s=S_DOT, linewidths=0)
    ax.scatter(y[m1], yp[m1], c=C_BIN1, label=rf"$|R| \leq {TH1}$", **kw)
    ax.scatter(y[m2], yp[m2], c=C_BIN2, label=rf"${TH1} < |R| \leq {TH2}$", **kw)
    ax.scatter(y[m3], yp[m3], c=C_BIN3, label=rf"$|R| > {TH2}$", **kw)

    lo, hi = float(y.min()), float(y.max())
    mg = (hi - lo) * 0.06
    ax.plot([lo - mg, hi + mg], [lo - mg, hi + mg], "--", color=C_DIAG, lw=1.2, zorder=3)

    ax.set_xlabel(f"Actual {TARGET_LABEL[target]}", labelpad=3)
    ax.set_ylabel(f"Predicted {TARGET_LABEL[target]}", labelpad=3)
    ax.set_title(
        f"TCN + CTGAN (processed)  –  "
        f"{DATASET_LABEL[dataset]} / {target.replace('_', ' ').title()}",
        fontsize=10, pad=5,
    )
    ax.legend(loc="upper left", handletextpad=0.4, borderpad=0.6)
    _grid_style(ax)

    ax.text(
        0.97, 0.05, _metrics_box(mae, rmse, r2),
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.40", fc="white", ec="#CCCCCC", alpha=0.88),
    )
    return _save(fig, stem, dpi)


def _plot_resi(y, yp, dataset, target, mae, rmse, r2, stem: Path, dpi: int) -> list[str]:
    r, m1, m2, m3 = _residual_bins(y, yp)

    fig = plt.figure(figsize=(7.0, 5.2))
    gs = gridspec.GridSpec(
        4, 4, figure=fig,
        wspace=0.40, hspace=0.42,
        left=0.12, right=0.96, top=0.91, bottom=0.12,
    )
    main_ax = fig.add_subplot(gs[0:3, 1:4])
    y_hist = fig.add_subplot(gs[0:3, 0], sharey=main_ax)
    x_hist = fig.add_subplot(gs[3, 1:4], sharex=main_ax)

    kw = dict(alpha=ALPHA, s=S_DOT, linewidths=0)
    main_ax.scatter(yp[m1], r[m1], c=C_BIN1, label=rf"$|R| \leq {TH1}$", **kw)
    main_ax.scatter(yp[m2], r[m2], c=C_BIN2, label=rf"${TH1} < |R| \leq {TH2}$", **kw)
    main_ax.scatter(yp[m3], r[m3], c=C_BIN3, label=rf"$|R| > {TH2}$", **kw)
    main_ax.axhline(0, linestyle="--", color=C_DIAG, lw=1.2, zorder=3)

    main_ax.spines["top"].set_visible(False)
    main_ax.spines["right"].set_visible(False)
    _grid_style(main_ax, x=False)
    main_ax.tick_params(labelbottom=False)

    main_ax.set_title(
        f"TCN + CTGAN (processed)  –  "
        f"{DATASET_LABEL[dataset]} / {target.replace('_', ' ').title()}",
        fontsize=10, pad=5,
    )
    main_ax.legend(loc="upper right", handletextpad=0.4, borderpad=0.6)
    main_ax.text(
        0.02, 0.04, _metrics_box(mae, rmse, r2),
        transform=main_ax.transAxes, ha="left", va="bottom", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#CCCCCC", alpha=0.88),
    )

    # Left histogram (residual distribution).
    y_hist.hist(r, 60, orientation="horizontal", color=C_HIST, alpha=0.80)
    y_hist.invert_xaxis()
    y_hist.set_ylabel("Residual", labelpad=3)
    y_hist.spines["top"].set_visible(False)
    y_hist.spines["right"].set_visible(False)
    y_hist.tick_params(labelbottom=False)
    _grid_style(y_hist, x=False)

    # Bottom histogram (predicted value distribution).
    x_hist.hist(yp, 60, orientation="vertical", color=C_HIST, alpha=0.80)
    x_hist.invert_yaxis()
    x_hist.set_xlabel(f"Predicted {TARGET_LABEL[target]}", labelpad=3)
    x_hist.spines["top"].set_visible(False)
    x_hist.spines["right"].set_visible(False)
    x_hist.tick_params(labelleft=False)
    _grid_style(x_hist, y=False)

    return _save(fig, stem, dpi)


def export_prediction_figures(
    dataset: str,
    target: str,
    y_true,
    y_pred,
    figures_dir,
    *,
    dpi: int = DPI,
) -> list[str]:
    """Draw + save the pred and residual figures for one dataset/target (TCP workflow).

    ``y_true`` / ``y_pred`` are the full 10-fold aggregated arrays. Returns the
    list of written file paths (png + pdf for each of pred and resi).
    """
    matplotlib.rcParams.update(_RC)
    y = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    mae, rmse, r2 = pooled_metrics(y, yp)

    out = Path(figures_dir) / "predictions" / dataset
    sfx = TARGET_SUFFIX[target]
    written = []
    written += _plot_pred(y, yp, dataset, target, mae, rmse, r2, out / f"{sfx}_pred", dpi)
    written += _plot_resi(y, yp, dataset, target, mae, rmse, r2, out / f"{sfx}_resi", dpi)
    return written
