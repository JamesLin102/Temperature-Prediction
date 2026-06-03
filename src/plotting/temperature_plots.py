"""Temperature-over-time figures for Seattle and Seoul.

Ported verbatim (style-wise) from the former ``scripts/plot_temperature_over_time.py``.
Produces, reading only from ``data/``:

  * ``<figures>/temperature_over_time_seattle.{png,pdf}`` — full daily series.
  * ``<figures>/temperature_over_time_seoul.{png,pdf}`` — five summers (Jun–Aug,
    2013–2017) with wave-break separators between years.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.gridspec as gridspec  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

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
    "xtick.labelsize":     10,
    "ytick.labelsize":     10,
    "axes.labelsize":      11,
    "legend.fontsize":     9,
    "legend.frameon":      True,
    "legend.framealpha":   0.92,
    "legend.edgecolor":    "#CCCCCC",
    "legend.handlelength": 1.6,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
}

C_MAX = "#B03A2E"
C_MIN = "#1A5276"
C_FILL_MAX = "#E74C3C"
C_FILL_MIN = "#2E86C1"
LW = 1.15
A_LINE = 0.88
A_FILL = 0.17
DPI = 300

_LEGEND_KW = dict(loc="upper right", ncol=2, columnspacing=1.0, handletextpad=0.5,
                  borderpad=0.6, labelspacing=0.3)

_YEARS = [2013, 2014, 2015, 2016, 2017]


def _fill_band(ax, x, hi, lo):
    mid = (hi + lo) / 2
    ax.fill_between(x, hi, mid, alpha=A_FILL, color=C_FILL_MAX, linewidth=0)
    ax.fill_between(x, mid, lo, alpha=A_FILL, color=C_FILL_MIN, linewidth=0)


def _style_grid(ax):
    ax.yaxis.grid(True, linestyle="--", linewidth=0.40, alpha=0.55, color="#AAAAAA")
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)


def _draw_waves(ax, n_cycles=3, color="#777777", lw=1.0):
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1)
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.75)
    ax.spines["bottom"].set_color("#555555")
    ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
    t = np.linspace(0.05, 0.95, 400)
    wav = 0.28 * np.sin(t * n_cycles * 2 * np.pi)
    for cx in (-0.22, 0.22):
        ax.plot(cx + wav, t, color=color, lw=lw, clip_on=True, zorder=5)


def _save(fig, out_dir: Path, stem: str, dpi: int) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in ("png", "pdf"):
        path = out_dir / f"{stem}.{ext}"
        # Drop the PDF creation timestamp so re-runs are byte-reproducible.
        meta = {"CreationDate": None} if ext == "pdf" else None
        fig.savefig(path, dpi=dpi, bbox_inches="tight", metadata=meta)
        written.append(str(path))
    plt.close(fig)
    return written


def _plot_seattle(sea: pd.DataFrame, out_dir: Path, dpi: int) -> list[str]:
    fig, ax = plt.subplots(
        figsize=(8.0, 2.8),
        gridspec_kw=dict(left=0.10, right=0.97, top=0.93, bottom=0.17),
    )
    d, hi, lo = sea["date"], sea["temp_max"], sea["temp_min"]
    _fill_band(ax, d, hi, lo)
    ax.plot(d, hi, color=C_MAX, lw=LW, alpha=A_LINE, label="Max Temperature")
    ax.plot(d, lo, color=C_MIN, lw=LW, alpha=A_LINE, label="Min Temperature")

    ax.set_ylabel("Temperature (°C)")
    ax.set_xlim(d.iloc[0], d.iloc[-1])
    ax.set_ylim(-12, 42)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=7))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center")

    ax.set_xlabel("Date")
    ax.legend(**_LEGEND_KW)
    _style_grid(ax)
    return _save(fig, out_dir, "temperature_over_time_seattle", dpi)


def _plot_seoul(summers: dict, out_dir: Path, dpi: int) -> list[str]:
    fig = plt.figure(figsize=(8.0, 2.8))
    n = len(_YEARS)
    gap_w = 0.13  # gap column width relative to data column

    col_w = []
    for i in range(n):
        col_w.append(1.0)
        if i < n - 1:
            col_w.append(gap_w)

    gs = gridspec.GridSpec(
        1, 2 * n - 1, figure=fig, width_ratios=col_w, wspace=0.0,
        left=0.10, right=0.97, top=0.93, bottom=0.17,
    )

    axes_s = []
    for col_idx in range(2 * n - 1):
        if col_idx % 2 == 1:  # gap column -> wave break
            ax_g = fig.add_subplot(gs[0, col_idx])
            _draw_waves(ax_g, n_cycles=3)
            continue

        i = col_idx // 2
        yr = _YEARS[i]
        sharey = axes_s[0] if axes_s else None
        ax = fig.add_subplot(gs[0, col_idx], sharey=sharey)
        axes_s.append(ax)

        df = summers[yr]
        hi_s = df["Present_Tmax"]
        lo_s = df["Present_Tmin"]
        ds = df["Date"]

        _fill_band(ax, ds, hi_s, lo_s)
        ax.plot(ds, hi_s, color=C_MAX, lw=LW, alpha=A_LINE,
                label="Max Temperature" if i == 0 else "_nolegend_")
        ax.plot(ds, lo_s, color=C_MIN, lw=LW, alpha=A_LINE,
                label="Min Temperature" if i == 0 else "_nolegend_")

        ax.set_xlim(pd.Timestamp(f"{yr}-06-01"), pd.Timestamp(f"{yr}-09-01"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[6, 7, 8]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center", fontsize=9)
        ax.set_xlabel(str(yr), fontsize=10, labelpad=3)
        _style_grid(ax)

        if i == 0:
            ax.set_ylabel("Temperature (°C)")
            ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
        else:
            ax.tick_params(labelleft=False)
            ax.spines["left"].set_visible(False)

    axes_s[0].set_ylim(14, 42)
    handles, labels = axes_s[0].get_legend_handles_labels()
    axes_s[-1].legend(handles, labels, **_LEGEND_KW)
    return _save(fig, out_dir, "temperature_over_time_seoul", dpi)


def export_temperature_figures(figures_dir, data_dir, *, dpi: int = DPI) -> list[str]:
    """Draw both temperature-over-time figures from the raw dataset CSVs."""
    matplotlib.rcParams.update(_RC)
    data_dir = Path(data_dir)
    out_dir = Path(figures_dir)

    sea = (pd.read_csv(data_dir / "seattle" / "seattle-weather.csv", parse_dates=["date"])
           .sort_values("date").reset_index(drop=True))

    seoul_raw = (pd.read_csv(data_dir / "seoul" / "Bias_correction_ucl.csv", parse_dates=["Date"])
                 .sort_values("Date"))
    daily = seoul_raw.groupby("Date")[["Present_Tmax", "Present_Tmin"]].mean()
    summers = {
        yr: daily[(daily.index.year == yr) & (daily.index.month.isin([6, 7, 8]))].reset_index()
        for yr in _YEARS
    }

    written = []
    written += _plot_seattle(sea, out_dir, dpi)
    written += _plot_seoul(summers, out_dir, dpi)
    return written
