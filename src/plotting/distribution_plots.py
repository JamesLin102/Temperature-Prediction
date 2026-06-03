"""Preprocessing-stage distribution figures (synthetic vs original).

Ported verbatim (style-wise) from the former ``scripts/plot_preprocessing_distributions.py``.
For each target, draws a 2x2 grid of KDE overlays showing how the synthetic
distribution evolves through the TCP preprocessing pipeline, for both datasets
on one figure:

  (a) Before outlier removal   (b) After outlier removal
  (c) After standardisation    (d) After Yeo-Johnson transformation

Reads a representative fold from ``data/`` and writes
``<figures>/fig_dist_t{max,min}.{png,pdf}``.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402
from sklearn.preprocessing import PowerTransformer, StandardScaler  # noqa: E402
from sklearn.svm import OneClassSVM  # noqa: E402

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
    "legend.fontsize":     9,
    "legend.frameon":      True,
    "legend.framealpha":   0.92,
    "legend.edgecolor":    "#CCCCCC",
    "legend.handlelength": 1.6,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
}

# key, line colour, fill colour, linestyle, linewidth, label
CURVES = [
    ("sea_ori", "#1A5276", "#2E86C1", "-",  1.4, "Seattle – Original"),
    ("sea_syn", "#2980B9", "#AED6F1", "--", 1.1, "Seattle – Synthetic"),
    ("seo_ori", "#922B21", "#E74C3C", "-",  1.4, "Seoul – Original"),
    ("seo_syn", "#D35400", "#F0B27A", "--", 1.1, "Seoul – Synthetic"),
]
A_FILL = 0.18
DPI = 300
_NU = 0.05

TARGETS = ["temp_max", "temp_min"]
TARGET_LABEL = {"temp_max": "Maximum Temperature", "temp_min": "Minimum Temperature"}

# (row, col, panel, title, xlabel, ori_key, syn_key)
SUBPLOT_CFG = [
    (0, 0, "(a)", "Before outlier removal",          "Temperature (°C)",   "raw_ori", "raw_syn"),
    (0, 1, "(b)", "After outlier removal",            "Temperature (°C)",   "raw_ori", "filt_syn"),
    (1, 0, "(c)", "After standardisation",            "Standardised value", "std_ori", "std_syn"),
    (1, 1, "(d)", "After Yeo–Johnson transformation", "Transformed value",  "yj_ori",  "yj_syn"),
]


def _compute_stages(data_dir: Path, dataset: str, target: str, fold: int) -> dict:
    """Y-column distributions at each preprocessing stage for one dataset/target.

    Stages 3–4 fit the scaler on original data and apply it to both, so relative
    positions are preserved for a meaningful visual comparison.
    """
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
        "raw_ori": ori_y, "raw_syn": syn_y, "filt_syn": filt_y,
        "std_ori": ori_std, "std_syn": filt_std,
        "yj_ori": ori_yj, "yj_syn": filt_yj,
    }


def _plot_4_kde(ax, arrays: dict, xlabel: str):
    """Overlay 4 KDE curves with fills. ``arrays``: {curve_key: 1-D ndarray}."""
    all_vals = np.concatenate(list(arrays.values()))
    lo, hi = all_vals.min(), all_vals.max()
    margin = (hi - lo) * 0.10
    xs = np.linspace(lo - margin, hi + margin, 600)

    ys = {key: gaussian_kde(arrays[key])(xs) for key, *_ in CURVES}

    for key, lc, fc, ls, lw, label in CURVES:
        ax.fill_between(xs, ys[key], alpha=A_FILL, color=fc, linewidth=0, zorder=1)
    for key, lc, fc, ls, lw, label in CURVES:
        ax.plot(xs, ys[key], color=lc, lw=lw, linestyle=ls, label=label, zorder=2)

    ax.set_xlabel(xlabel, labelpad=3)
    ax.set_ylabel("Density")
    ax.yaxis.grid(True, linestyle="--", linewidth=0.40, alpha=0.55, color="#AAAAAA")
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))


def _build_figure(data_dir: Path, target: str, fold: int):
    sea = _compute_stages(data_dir, "seattle", target, fold)
    seo = _compute_stages(data_dir, "seoul", target, fold)

    fig, axes = plt.subplots(
        2, 2, figsize=(7.4, 5.4),
        gridspec_kw=dict(left=0.11, right=0.97, top=0.93, bottom=0.17,
                         hspace=0.50, wspace=0.38),
    )

    for row, col, panel, title, xlabel, ori_key, syn_key in SUBPLOT_CFG:
        ax = axes[row, col]
        arrays = {
            "sea_ori": sea[ori_key], "sea_syn": sea[syn_key],
            "seo_ori": seo[ori_key], "seo_syn": seo[syn_key],
        }
        _plot_4_kde(ax, arrays, xlabel)
        ax.set_title(title, fontsize=9.5, pad=5)
        ax.text(0.5, -0.26, panel, transform=ax.transAxes,
                fontsize=9, fontweight="bold", va="top", ha="center")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.54, 0.01),
        fontsize=8.5, frameon=True, framealpha=0.95, edgecolor="#CCCCCC",
        handlelength=1.8, handletextpad=0.5, columnspacing=1.2,
    )
    return fig


def export_distribution_figures(figures_dir, data_dir, *, fold: int = 1, dpi: int = DPI) -> list[str]:
    """Draw fig_dist_tmax and fig_dist_tmin from a representative fold's arrays."""
    matplotlib.rcParams.update(_RC)
    data_dir = Path(data_dir)
    out_dir = Path(figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for target in TARGETS:
        suffix = "tmax" if target == "temp_max" else "tmin"
        fig = _build_figure(data_dir, target, fold)
        for ext in ("png", "pdf"):
            path = out_dir / f"fig_dist_{suffix}.{ext}"
            # Drop the PDF creation timestamp so re-runs are byte-reproducible.
            meta = {"CreationDate": None} if ext == "pdf" else None
            fig.savefig(path, dpi=dpi, bbox_inches="tight", metadata=meta)
            written.append(str(path))
        plt.close(fig)
    return written
