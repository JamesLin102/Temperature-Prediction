"""Generate new-dataset (Bias_correction_ucl) KFold .npy assets.

Mirrors Original Programe/data_gan.py, adapted for the multi-station CSV:
  * build station-safe lookback sequences (sequences never cross stations),
  * pool all stations, split with KFold(n_splits, shuffle=False),
  * per fold save legacy (N, 6) arrays:
      ori_training_data_<fold>.npy   (real training windows)
      synthetic_data_<fold>.npy      (CTGAN samples, mirrors data_gan.py)
      testing_data_<fold>.npy        (real test windows)
    into data/new_dataset/<target>/.

This script is WRITTEN but not run as part of the build pass. CTGAN training is
heavy (~hours for 10 folds x 2 targets); run it deliberately when ready:

    python scripts/generate_new_dataset_kfold_assets.py            # full
    python scripts/generate_new_dataset_kfold_assets.py --smoke    # tiny/fast
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
from src.data.dataset_loader import load_seoul_csv  # noqa: E402
from src.data.sequence_builder import build_station_safe_sequences, to_six_col  # noqa: E402
from src.utils import config, paths  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402
from src.validation.kfold import generate_kfold_splits  # noqa: E402


def generate(smoke: bool = False) -> None:
    """Build and save the new-dataset KFold assets for both targets."""
    log = get_logger("generate_new_dataset")
    configs = config.load_all_configs()

    seed = configs["models"].get("seed", 7)
    set_seed(seed)
    lookback = configs["datasets"]["lookback"]
    n_splits = configs["datasets"]["n_splits"]
    shuffle = configs["datasets"].get("shuffle", False)

    ds_cfg = configs["datasets"]["datasets"]["seoul"]
    date_col = ds_cfg["date_col"]
    station_col = ds_cfg["station_col"]
    target_map = ds_cfg["target_map"]  # {Next_Tmax: temp_max, Next_Tmin: temp_min}

    ctgan_cfg = dict(configs["models"]["ctgan"])
    if smoke:
        smoke_cfg = configs["experiments"]["smoke"]
        ctgan_cfg["epochs"] = smoke_cfg.get("ctgan_epochs", 2)
        ctgan_cfg["num_samples"] = smoke_cfg.get("ctgan_num_samples", 50)
    log.info("CTGAN settings: %s", ctgan_cfg)

    df = load_seoul_csv()
    df[date_col] = pd.to_datetime(df[date_col])

    # Drop rows with missing station ID (cannot assign to any station).
    n_before = len(df)
    df = df.dropna(subset=[station_col]).copy()
    if len(df) < n_before:
        log.warning("Dropped %d rows with NaN station — %d remain", n_before - len(df), len(df))

    for csv_col, target_name in target_map.items():
        log.info("=== %s -> %s ===", csv_col, target_name)

        # Fill NaN in the target column within each station (ffill then bfill).
        # This avoids phantom sequences across time gaps while keeping station continuity.
        n_nan = df[csv_col].isna().sum()
        if n_nan:
            df[csv_col] = df.groupby(station_col)[csv_col].transform(
                lambda x: x.ffill().bfill()
            )
            log.warning("Filled %d NaN in %s via ffill/bfill per station", n_nan, csv_col)

        X, y = build_station_safe_sequences(df, csv_col, station_col, date_col, lookback)
        log.info("Pooled station-safe sequences: X=%s y=%s", X.shape, y.shape)

        splits = generate_kfold_splits(X, y, n_splits=n_splits, shuffle=shuffle)
        out_dir = paths.ensure_dir(paths.SEOUL_DIR / target_name)
        augmenter = CTGANAugmenter(**ctgan_cfg)

        for fold_id, (train_idx, test_idx) in enumerate(splits, start=1):
            ori_six = to_six_col(X[train_idx], y[train_idx])
            test_six = to_six_col(X[test_idx], y[test_idx])
            np.save(out_dir / f"ori_training_data_{fold_id}.npy", ori_six)
            np.save(out_dir / f"testing_data_{fold_id}.npy", test_six)

            synthetic = augmenter.fit_sample(ori_six)
            np.save(out_dir / f"synthetic_data_{fold_id}.npy", synthetic)

            log.info(
                "fold %2d: ori=%s synthetic=%s testing=%s",
                fold_id, ori_six.shape, synthetic.shape, test_six.shape,
            )

    log.info("Done. Seoul KFold assets written under %s", paths.SEOUL_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate new_dataset KFold .npy assets.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use tiny CTGAN settings (fast, for verification only).",
    )
    args = parser.parse_args()
    generate(smoke=args.smoke)


if __name__ == "__main__":
    main()
