"""Regenerate Seattle synthetic fold .npy files using CTGANSynthesizer.

Reads existing ori_training_data_*.npy (already KFold-split), runs CTGAN on each,
saves new synthetic_data_*.npy in place.  ori_training / testing folds are untouched.

Usage:
    python scripts/regenerate_seattle_synthetic.py [--seed SEED]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import numpy as np

from src.augmentation.ctgan_augmenter import CTGANAugmenter
from src.utils.config import load_all_configs
from src.utils.logger import get_logger
from src.utils.seed import set_seed

_TARGETS = ["temp_max", "temp_min"]
_N_FOLDS = 10


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7, help="Base random seed")
    args = parser.parse_args()

    log = get_logger("regenerate_seattle_synthetic",
                     log_file="outputs/logs/regenerate_seattle_synthetic.log")
    configs = load_all_configs()
    ctgan_cfg = configs["models"]["ctgan"]

    augmenter_kwargs = dict(
        epochs=ctgan_cfg["epochs"],
        num_samples=ctgan_cfg["num_samples"],
        enforce_min_max_values=ctgan_cfg["enforce_min_max_values"],
    )

    for target in _TARGETS:
        kfold_dir = _ROOT / "data" / "seattle" / target
        log.info("=== %s ===", target)

        for fold_id in range(1, _N_FOLDS + 1):
            ori_path = kfold_dir / f"ori_training_data_{fold_id}.npy"
            out_path = kfold_dir / f"synthetic_data_{fold_id}.npy"

            ori = np.load(ori_path)
            log.info("fold %2d | ori=%d rows -> fitting CTGAN ...", fold_id, len(ori))

            # Derive a per-fold seed so each fold has different but reproducible CTGAN weights
            fold_seed = args.seed * 100 + fold_id
            set_seed(fold_seed)

            aug = CTGANAugmenter(**augmenter_kwargs)
            synthetic = aug.fit_sample(ori)

            np.save(out_path, synthetic)
            log.info("fold %2d | saved %s  (%d rows)", fold_id, out_path.name, len(synthetic))

    log.info("Done. All synthetic data regenerated.")


if __name__ == "__main__":
    main()
