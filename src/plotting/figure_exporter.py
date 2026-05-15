"""Figure export orchestration with legacy filenames (spec section 8.4)."""
from __future__ import annotations

from ..utils import paths
from . import comparison_plot, legacy_plot

# workflow name -> legacy aggregated-array prefix
_WORKFLOW_PREFIX = {
    "TCN": "T",
    "TCN + CTGAN": "TC",
    "TCN + CTGAN (processed)": "TCP",
}


def _figures_dir(dataset: str, target: str, outputs_dir=None):
    base = paths.OUTPUTS_DIR if outputs_dir is None else paths.resolve(outputs_dir)
    return paths.ensure_dir(base / "figures" / dataset / target)


def _style(plotting_cfg: dict | None) -> dict:
    cfg = plotting_cfg or {}
    return {
        "thresholds": tuple(cfg.get("residual_thresholds", (2.5, 5.0))),
        "colors": cfg.get("colors"),
        "pred_figsize": tuple(cfg.get("pred_figsize", (7, 5))),
        "residual_figsize": tuple(cfg.get("residual_figsize", (7, 5))),
        "comparison_figsize": tuple(cfg.get("comparison_figsize", (15, 8))),
        "hist_bins": cfg.get("hist_bins", 60),
        "dpi": cfg.get("figure_dpi", 150),
        "comparison_xlim": tuple(cfg.get("comparison_xlim", (0, 100))),
        "workflow_figure_names": cfg.get("workflow_figure_names", {}),
        "comparison_slices": cfg.get("comparison_slices", []),
    }


def export_workflow_figures(dataset, target, arrays_by_workflow, plotting_cfg, outputs_dir=None):
    """Write pred + residual figures. ``arrays_by_workflow``: {workflow: (y_true, y_pred)}."""
    style = _style(plotting_cfg)
    fig_dir = _figures_dir(dataset, target, outputs_dir)
    written = []
    for workflow, (y_true, y_pred) in arrays_by_workflow.items():
        names = style["workflow_figure_names"].get(workflow)
        if names is None:
            continue
        written.append(
            legacy_plot.plot_pred(
                y_true, y_pred, workflow, fig_dir / names["pred"],
                thresholds=style["thresholds"], colors=style["colors"],
                figsize=style["pred_figsize"], dpi=style["dpi"],
            )
        )
        written.append(
            legacy_plot.plot_residuals(
                y_true, y_pred, workflow, fig_dir / names["resi"],
                thresholds=style["thresholds"], colors=style["colors"],
                figsize=style["residual_figsize"], hist_bins=style["hist_bins"], dpi=style["dpi"],
            )
        )
    return written


def export_comparison_figures(dataset, target, arrays_by_prefix, plotting_cfg, outputs_dir=None):
    """Write the algorithm comparison figures. ``arrays_by_prefix``: {'T'|'TC'|'TCP': (true, pred)}."""
    style = _style(plotting_cfg)
    fig_dir = _figures_dir(dataset, target, outputs_dir)
    tcp_true, tcp_pred = arrays_by_prefix["TCP"]
    _, tc_pred = arrays_by_prefix["TC"]
    _, t_pred = arrays_by_prefix["T"]
    written = []
    for sl in style["comparison_slices"]:
        written.append(
            comparison_plot.plot_comparison(
                tcp_true, tcp_pred, tc_pred, t_pred,
                start=sl["start"], end=sl["end"], title=sl["title"],
                output_path=fig_dir / sl["filename"],
                figsize=style["comparison_figsize"], xlim=style["comparison_xlim"], dpi=style["dpi"],
            )
        )
    return written
