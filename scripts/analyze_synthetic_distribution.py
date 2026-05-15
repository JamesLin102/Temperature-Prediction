"""Compare distributions of original, raw CTGAN, and filtered CTGAN data.

For each target × fold, computes summary statistics and overlaid histograms
showing that OneClassSVM filtering (nu=0.05) brings synthetic data back toward
the original distribution — the core argument for the TCP (processed) workflow.

Usage:
    python scripts/analyze_synthetic_distribution.py [--fold FOLD] [--target TARGET]

Outputs:
    outputs/figures/distribution/<target>/fold_<N>_distribution.png
    outputs/figures/distribution/<target>/summary_stats.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.utils.config import load_all_configs
from src.utils.logger import get_logger
import src.utils.paths as paths

_TARGETS = ["temp_max", "temp_min"]
_COL_LABELS = ["T-4", "T-3", "T-2", "T-1", "T", "T+1 (target)"]
_NU = 0.05


def _filter_synthetic(synthetic: np.ndarray) -> np.ndarray:
    labels = OneClassSVM(nu=_NU).fit_predict(synthetic)
    return np.delete(synthetic, np.where(labels == -1)[0], axis=0)


def _stats(arr: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame(arr, columns=_COL_LABELS)
    return df.describe().loc[["mean", "std", "min", "max"]].round(4)


def plot_fold_distribution(
    ori: np.ndarray,
    synthetic_raw: np.ndarray,
    synthetic_filt: np.ndarray,
    target: str,
    fold_id: int,
    out_path: Path,
) -> None:
    n_cols = len(_COL_LABELS)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, (ax, col) in enumerate(zip(axes, _COL_LABELS)):
        ax.hist(ori[:, i], bins=30, alpha=0.5, color="steelblue",
                label=f"Original (n={len(ori)})", density=True)
        ax.hist(synthetic_raw[:, i], bins=30, alpha=0.4, color="orange",
                label=f"CTGAN raw (n={len(synthetic_raw)})", density=True)
        ax.hist(synthetic_filt[:, i], bins=30, alpha=0.5, color="green",
                label=f"CTGAN filtered (n={len(synthetic_filt)})", density=True)
        ax.set_title(col)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        if i == 0:
            ax.legend(fontsize=7)

    fig.suptitle(
        f"{target}  |  Fold {fold_id}  |  Distribution comparison\n"
        f"OneClassSVM nu={_NU}: {len(synthetic_raw)} raw → {len(synthetic_filt)} filtered "
        f"({100*len(synthetic_filt)/len(synthetic_raw):.1f}% kept)",
        fontsize=11,
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=1,
                        help="Which fold to plot (1-10, default=1)")
    parser.add_argument("--target", choices=_TARGETS + ["all"], default="all")
    args = parser.parse_args()

    log = get_logger("analyze_synthetic_distribution",
                     log_file="outputs/logs/analyze_synthetic_distribution.log")
    configs = load_all_configs()
    targets = _TARGETS if args.target == "all" else [args.target]

    all_rows = []

    for target in targets:
        kfold_dir = _ROOT / "data" / "seattle" / target
        out_dir = _ROOT / "outputs" / "figures" / "distribution" / target
        out_dir.mkdir(parents=True, exist_ok=True)

        # --- per-fold plot ---
        fold_id = args.fold
        ori_path = kfold_dir / f"ori_training_data_{fold_id}.npy"
        syn_path = kfold_dir / f"synthetic_data_{fold_id}.npy"

        ori = np.load(ori_path)
        syn_raw = np.load(syn_path)
        syn_filt = _filter_synthetic(syn_raw)

        plot_path = out_dir / f"fold_{fold_id}_distribution.png"
        plot_fold_distribution(ori, syn_raw, syn_filt, target, fold_id, plot_path)
        log.info("Saved %s", plot_path)

        # --- summary stats across all folds ---
        for fid in range(1, 11):
            ori_f = np.load(kfold_dir / f"ori_training_data_{fid}.npy")
            syn_f = np.load(kfold_dir / f"synthetic_data_{fid}.npy")
            filt_f = _filter_synthetic(syn_f)

            for col_idx, col_name in enumerate(_COL_LABELS):
                all_rows.append({
                    "target": target,
                    "fold": fid,
                    "feature": col_name,
                    "ori_mean": round(ori_f[:, col_idx].mean(), 4),
                    "ori_std":  round(ori_f[:, col_idx].std(),  4),
                    "raw_mean": round(syn_f[:, col_idx].mean(), 4),
                    "raw_std":  round(syn_f[:, col_idx].std(),  4),
                    "filt_mean": round(filt_f[:, col_idx].mean(), 4),
                    "filt_std":  round(filt_f[:, col_idx].std(),  4),
                    "n_ori":     len(ori_f),
                    "n_raw":     len(syn_f),
                    "n_filt":    len(filt_f),
                    "pct_kept":  round(100 * len(filt_f) / len(syn_f), 1),
                })

        # Print aggregate summary (mean across folds)
        subset = [r for r in all_rows if r["target"] == target]
        df = pd.DataFrame(subset)
        agg = df.groupby("feature")[["ori_mean", "filt_mean", "ori_std", "filt_std"]].mean().round(4)
        print(f"\n=== {target}: mean statistics across 10 folds ===")
        print(agg.to_string())

        avg_kept = df["pct_kept"].mean()
        print(f"\n  Avg rows kept after filtering: {avg_kept:.1f}%  "
              f"(~{df['n_filt'].mean():.0f} / {df['n_raw'].mean():.0f})")

    # Save full table
    df_all = pd.DataFrame(all_rows)
    csv_path = _ROOT / "outputs" / "figures" / "distribution" / "summary_stats.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(csv_path, index=False)
    log.info("Summary CSV saved to %s", csv_path)
    print(f"\nFull stats → {csv_path}")


if __name__ == "__main__":
    main()
