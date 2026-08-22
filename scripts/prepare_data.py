"""Turn the downloaded raw CSVs into the 10-fold .npy assets the experiments read.

The datasets are not redistributed in this repository. Download them yourself
(see data/README.md), drop the CSVs in:

    data/seattle/seattle-weather.csv
    data/seoul/Bias_correction_ucl.csv

and then run:

    python scripts/prepare_data.py --dataset all

For every dataset and target this writes, under data/<dataset>/<target>/:

    ori_training_data_<fold>.npy   real training windows  (N, 6)
    testing_data_<fold>.npy        real test windows      (N, 6)
    synthetic_data_<fold>.npy      CTGAN samples          (num_samples, 6)

Each row is five lookback values plus the next-day target. Sequences are built
with the off-by-one convention of the original 2024 implementation and split
with KFold(n_splits=10, shuffle=False), so the real folds come out identical to
the ones used for the published results. The CTGAN step is stochastic: the
synthetic folds will differ from run to run, which moves the final metrics
slightly.

Useful flags:

    --dataset seattle|seoul|all   which dataset to prepare (default: all)
    --skip-synthetic              real folds only; fast, no CTGAN training
    --synthetic-only              keep the real folds, retrain CTGAN only
    --smoke                       tiny CTGAN settings, for a fast functional check
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.augmentation.ctgan_augmenter import CTGANAugmenter  # noqa: E402
from src.data import dataset_loader  # noqa: E402
from src.data.sequence_builder import (  # noqa: E402
    build_station_safe_sequences,
    build_univariate_sequences,
    to_six_col,
)
from src.utils import config, paths  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402
from src.validation.kfold import generate_kfold_splits  # noqa: E402

DOWNLOAD_HINTS = {
    "seattle": (
        "data/seattle/seattle-weather.csv\n"
        "    https://www.kaggle.com/datasets/ananthr1/weather-prediction/data"
    ),
    "seoul": (
        "data/seoul/Bias_correction_ucl.csv\n"
        "    https://archive.ics.uci.edu/dataset/514/"
        "bias+correction+of+numerical+prediction+model+temperature+forecast"
    ),
}


def _require_csv(dataset: str, loader) -> pd.DataFrame:
    """Load a raw CSV, or explain exactly what to download and where to put it."""
    try:
        return loader()
    except FileNotFoundError as exc:
        raise SystemExit(
            f"\n{exc}\n\n"
            f"The {dataset} dataset is not redistributed in this repository.\n"
            f"Download it and save it as:\n\n    {DOWNLOAD_HINTS[dataset]}\n\n"
            "See data/README.md for details.\n"
        ) from exc


def _ctgan_settings(configs: dict, smoke: bool) -> dict:
    cfg = dict(configs["models"]["ctgan"])
    if smoke:
        smoke_cfg = configs["experiments"]["smoke"]
        cfg["epochs"] = smoke_cfg.get("ctgan_epochs", 2)
        cfg["num_samples"] = smoke_cfg.get("ctgan_num_samples", 50)
    return cfg


def _write_folds(out_dir: Path, X, y, splits, log) -> None:
    """Save the real training/testing fold arrays."""
    for fold_id, (train_idx, test_idx) in enumerate(splits, start=1):
        ori_six = to_six_col(X[train_idx], y[train_idx])
        test_six = to_six_col(X[test_idx], y[test_idx])
        np.save(out_dir / f"ori_training_data_{fold_id}.npy", ori_six)
        np.save(out_dir / f"testing_data_{fold_id}.npy", test_six)
        log.info("fold %2d: ori=%s testing=%s", fold_id, ori_six.shape, test_six.shape)


def _write_synthetic(out_dir: Path, n_splits: int, ctgan_cfg: dict, log) -> None:
    """Train CTGAN on each real training fold and save the synthetic samples."""
    augmenter = CTGANAugmenter(**ctgan_cfg)
    for fold_id in range(1, n_splits + 1):
        ori_path = out_dir / f"ori_training_data_{fold_id}.npy"
        if not ori_path.exists():
            raise SystemExit(
                f"\nMissing {ori_path}.\n"
                "Build the real folds first (drop --synthetic-only).\n"
            )
        ori_six = np.load(ori_path)
        synthetic = augmenter.fit_sample(ori_six)
        np.save(out_dir / f"synthetic_data_{fold_id}.npy", synthetic)
        log.info("fold %2d: synthetic=%s", fold_id, synthetic.shape)


def prepare_seattle(configs: dict, *, skip_synthetic: bool, synthetic_only: bool,
                    smoke: bool, log) -> None:
    """Build the Seattle fold assets from data/seattle/seattle-weather.csv."""
    lookback = configs["datasets"]["lookback"]
    n_splits = configs["datasets"]["n_splits"]
    shuffle = configs["datasets"].get("shuffle", False)
    targets = configs["datasets"]["datasets"]["seattle"]["targets"]
    ctgan_cfg = _ctgan_settings(configs, smoke)

    df = None if synthetic_only else _require_csv("seattle", dataset_loader.load_seattle_csv)

    for target in targets:
        log.info("=== seattle / %s ===", target)
        out_dir = paths.ensure_dir(paths.SEATTLE_DIR / target)

        if not synthetic_only:
            X, y = build_univariate_sequences(df[target], lookback)
            log.info("sequences: X=%s y=%s", X.shape, y.shape)
            splits = generate_kfold_splits(X, n_splits=n_splits, shuffle=shuffle)
            _write_folds(out_dir, X, y, splits, log)

        if not skip_synthetic:
            log.info("training CTGAN (%s)", ctgan_cfg)
            _write_synthetic(out_dir, n_splits, ctgan_cfg, log)

    log.info("Seattle assets written under %s", paths.SEATTLE_DIR)


def prepare_seoul(configs: dict, *, skip_synthetic: bool, synthetic_only: bool,
                  smoke: bool, log) -> None:
    """Build the Seoul fold assets from data/seoul/Bias_correction_ucl.csv.

    Sequences are built per station and never cross station boundaries.
    """
    lookback = configs["datasets"]["lookback"]
    n_splits = configs["datasets"]["n_splits"]
    shuffle = configs["datasets"].get("shuffle", False)
    ds_cfg = configs["datasets"]["datasets"]["seoul"]
    date_col = ds_cfg["date_col"]
    station_col = ds_cfg["station_col"]
    target_map = ds_cfg["target_map"]  # {Next_Tmax: temp_max, Next_Tmin: temp_min}
    ctgan_cfg = _ctgan_settings(configs, smoke)

    df = None
    if not synthetic_only:
        df = _require_csv("seoul", dataset_loader.load_seoul_csv)
        df[date_col] = pd.to_datetime(df[date_col])

        # Drop rows with missing station ID (cannot assign to any station).
        n_before = len(df)
        df = df.dropna(subset=[station_col]).copy()
        if len(df) < n_before:
            log.warning("Dropped %d rows with NaN station — %d remain",
                        n_before - len(df), len(df))

    for csv_col, target in target_map.items():
        log.info("=== seoul / %s (%s) ===", target, csv_col)
        out_dir = paths.ensure_dir(paths.SEOUL_DIR / target)

        if not synthetic_only:
            # Fill NaN in the target column within each station (ffill then bfill).
            # This avoids phantom sequences across time gaps while keeping station continuity.
            n_nan = df[csv_col].isna().sum()
            if n_nan:
                df[csv_col] = df.groupby(station_col)[csv_col].transform(
                    lambda x: x.ffill().bfill()
                )
                log.warning("Filled %d NaN in %s via ffill/bfill per station", n_nan, csv_col)

            X, y = build_station_safe_sequences(df, csv_col, station_col, date_col, lookback)
            log.info("pooled station-safe sequences: X=%s y=%s", X.shape, y.shape)
            splits = generate_kfold_splits(X, n_splits=n_splits, shuffle=shuffle)
            _write_folds(out_dir, X, y, splits, log)

        if not skip_synthetic:
            log.info("training CTGAN (%s)", ctgan_cfg)
            _write_synthetic(out_dir, n_splits, ctgan_cfg, log)

    log.info("Seoul assets written under %s", paths.SEOUL_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the 10-fold .npy assets from the downloaded raw CSVs."
    )
    parser.add_argument(
        "--dataset",
        choices=["seattle", "seoul", "all"],
        default="all",
        help="Which dataset to prepare (default: all).",
    )
    parser.add_argument(
        "--skip-synthetic",
        action="store_true",
        help="Build the real folds only — fast, no CTGAN training.",
    )
    parser.add_argument(
        "--synthetic-only",
        action="store_true",
        help="Keep the existing real folds and retrain CTGAN only.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny CTGAN settings for a fast functional check, not real training.",
    )
    args = parser.parse_args()

    if args.skip_synthetic and args.synthetic_only:
        parser.error("--skip-synthetic and --synthetic-only are mutually exclusive.")

    log = get_logger("prepare_data", log_file="outputs/logs/prepare_data.log")
    configs = config.load_all_configs()
    set_seed(configs["models"].get("seed", 7))

    datasets = ["seattle", "seoul"] if args.dataset == "all" else [args.dataset]
    kwargs = dict(
        skip_synthetic=args.skip_synthetic,
        synthetic_only=args.synthetic_only,
        smoke=args.smoke,
    )

    if "seattle" in datasets:
        prepare_seattle(configs, log=log, **kwargs)
    if "seoul" in datasets:
        prepare_seoul(configs, log=log, **kwargs)

    log.info("Done. Run scripts/run_experiments.py next.")


if __name__ == "__main__":
    main()
