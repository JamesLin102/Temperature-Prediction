"""Algorithm comparison line plot, ported from Original Programe/plot.py.

Plots real temperature against the three workflows' predictions over a fixed
index slice. Saves the figure (the original called ``plt.show()``).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..utils import paths  # noqa: E402


def plot_comparison(
    tcp_true,
    tcp_pred,
    tc_pred,
    t_pred,
    *,
    start: int,
    end: int,
    title: str,
    output_path,
    figsize=(15, 8),
    xlim=(0, 100),
    dpi: int = 150,
) -> str:
    """Comparison of real temperature vs TCN+CTGAN(processed)/TCN+CTGAN/TCN predictions."""
    tcp_true = np.asarray(tcp_true, dtype=np.float64).reshape(-1)
    tcp_pred = np.asarray(tcp_pred, dtype=np.float64).reshape(-1)
    tc_pred = np.asarray(tc_pred, dtype=np.float64).reshape(-1)
    t_pred = np.asarray(t_pred, dtype=np.float64).reshape(-1)

    fig = plt.figure(figsize=tuple(figsize))
    ax = fig.add_subplot(111)
    ax.plot(tcp_true[start:end], label="Real Temperature")
    ax.plot(tcp_pred[start:end], label="Prediction of TCN+CTGAN(processed)")
    ax.plot(tc_pred[start:end], label="Prediction of TCN+CTGAN")
    ax.plot(t_pred[start:end], label="Prediction of TCN")
    ax.set_xlabel("Time (day)", fontsize=18)
    ax.set_ylabel("Temperature (°C)", fontsize=18)
    ax.set_xlim(list(xlim))
    ax.set_title(title, fontsize=18)
    ax.grid(True)
    ax.legend(fontsize=15)

    out = paths.resolve(output_path)
    paths.ensure_dir(out.parent)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return str(out)
