"""Multivariate input experiment (Seattle only).

The univariate setting uses pre-split legacy ``.npy`` fold assets (same as the
workflow and baseline experiments) so its TCN results are identical to those
experiments.  The multivariate settings build sequences from the raw CSV and
split with ``KFold(n_splits, shuffle=False)``.
"""
from __future__ import annotations

import pandas as pd

from ..data.dataset_loader import load_seattle_csv
from ..data.sequence_builder import build_multivariate_sequences
from ..evaluation.aggregator import aggregate_folds
from ..evaluation.export_predictions import write_fold_predictions
from ..evaluation.summary_tables import summary_row, write_summary
from ..utils.logger import get_logger
from ..validation.kfold import generate_kfold_splits
from .fold_runner import run_fold, run_in_memory_fold


def run(
    configs: dict,
    *,
    smoke: bool = False,
    targets=None,
    settings=None,
    fold_limit: int | None = None,
    outputs_dir=None,
    logger=None,
) -> list[dict]:
    """Run the multivariate experiment (Seattle only); return the summary rows."""
    log = logger or get_logger("multivariate_experiment")
    exp = configs["experiments"]["multivariate_comparison"]
    dataset = exp["dataset"]
    tg_list = targets or exp["targets"]
    setting_list = settings or exp["settings"]
    seed = configs["models"].get("seed", 7)
    lookback = configs["datasets"]["lookback"]
    n_splits = configs["datasets"]["n_splits"]
    shuffle = configs["datasets"].get("shuffle", False)

    ds_cfg = configs["datasets"]["datasets"][dataset]
    mv_features = list(ds_cfg["multivariate_features"])
    cat_col = ds_cfg.get("optional_categorical")

    model_cfg = dict(configs["models"]["models"]["tcn"])
    series_limit = None
    if smoke:
        model_cfg["epochs"] = configs["experiments"]["smoke"]["epochs"]
        series_limit = configs["experiments"]["smoke"].get("multivariate_series_limit")
        if fold_limit is None:
            fold_limit = configs["experiments"]["smoke"]["n_splits"]

    # Load CSV once — only needed for multivariate settings; univariate uses legacy assets.
    needs_csv = any(s["name"] != "univariate" for s in setting_list)
    df: pd.DataFrame | None = None
    if needs_csv:
        df = load_seattle_csv()
        if series_limit is not None:
            df = df.iloc[:series_limit].reset_index(drop=True)

    summary_rows: list[dict] = []
    for target in tg_list:
        for setting in setting_list:
            name = setting["name"]
            use_weather = bool(setting.get("use_weather", False))
            slug = f"multivariate_{name}"
            fold_results = []

            if name == "univariate":
                # Use legacy fold assets — same data/splits as workflow & baseline.
                n_folds = fold_limit if fold_limit is not None else n_splits
                for fold_id in range(1, n_folds + 1):
                    result = run_fold(
                        dataset, target, fold_id,
                        model_name="tcn", model_cfg=model_cfg, seed=seed,
                    )
                    fold_results.append(result)
                    write_fold_predictions(
                        fold_id, dataset, target, "model", "tcn_univariate",
                        result.y_true, result.y_pred, slug, outputs_dir=outputs_dir,
                    )
            else:
                feature_cols = list(mv_features)
                work_df = df
                if use_weather and cat_col:
                    work_df = df.copy()
                    if not pd.api.types.is_numeric_dtype(work_df[cat_col]):
                        work_df[cat_col] = work_df[cat_col].astype("category").cat.codes
                    feature_cols = feature_cols + [cat_col]
                X, y = build_multivariate_sequences(work_df, feature_cols, target, lookback)

                splits = generate_kfold_splits(X, y, n_splits=n_splits, shuffle=shuffle)
                if fold_limit is not None:
                    splits = splits[:fold_limit]

                for fold_id, (train_idx, test_idx) in enumerate(splits, start=1):
                    result = run_in_memory_fold(
                        fold_id,
                        X[train_idx], y[train_idx], X[test_idx], y[test_idx],
                        model_name="tcn", model_cfg=model_cfg, seed=seed,
                        seed_parts=(dataset, target, name),
                    )
                    fold_results.append(result)
                    write_fold_predictions(
                        fold_id, dataset, target, "model", f"tcn_{name}",
                        result.y_true, result.y_pred, slug, outputs_dir=outputs_dir,
                    )

            agg = aggregate_folds(fold_results)
            summary_rows.append(
                summary_row(
                    {
                        "dataset": dataset,
                        "target": target,
                        "setting": name,
                        "use_weather": use_weather,
                    },
                    agg,
                )
            )
            log.info(
                "setting=%-26s target=%-9s folds=%d MAE=%.4f",
                name, target, len(fold_results), agg.overall_metrics["MAE"],
            )

    write_summary(summary_rows, "multivariate_comparison_summary.csv", outputs_dir=outputs_dir)
    return summary_rows
