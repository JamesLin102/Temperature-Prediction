"""Experiment 7.2 — baseline comparison.

{ARIMA, SVR, RandomForest, LightGBM, LSTM, GRU, TCN} x datasets x targets. No
CTGAN, no processed workflow — uses ``ori_training`` + ``testing`` folds only.
A dataset with no assets (new_dataset this pass) is skipped gracefully.
"""
from __future__ import annotations

from ..data.dataset_loader import fold_assets_available
from ..evaluation.aggregator import aggregate_folds
from ..evaluation.export_predictions import write_fold_predictions
from ..evaluation.summary_tables import summary_row, write_summary
from ..utils.logger import get_logger
from .fold_runner import run_fold

_NN_MODELS = {"tcn", "lstm", "gru"}


def run(
    configs: dict,
    *,
    smoke: bool = False,
    datasets=None,
    targets=None,
    models=None,
    fold_limit: int | None = None,
    outputs_dir=None,
    logger=None,
) -> list[dict]:
    """Run the baseline comparison; return the list of summary rows."""
    log = logger or get_logger("baseline_comparison")
    exp = configs["experiments"]["baseline_comparison"]
    ds_list = datasets or exp["datasets"]
    tg_list = targets or exp["targets"]
    model_list = models or exp["models"]
    seed = configs["models"].get("seed", 7)
    n_splits = configs["datasets"]["n_splits"]

    smoke_epochs = configs["experiments"]["smoke"]["epochs"] if smoke else None
    if smoke and fold_limit is None:
        fold_limit = configs["experiments"]["smoke"]["n_splits"]

    fold_ids = list(range(1, n_splits + 1))
    if fold_limit is not None:
        fold_ids = fold_ids[:fold_limit]

    summary_rows: list[dict] = []
    for dataset in ds_list:
        for target in tg_list:
            if not fold_assets_available(dataset, target, n_splits, kinds=("ori_training", "testing")):
                log.warning("Skipping %s/%s: KFold .npy assets not generated.", dataset, target)
                continue

            for model_name in model_list:
                model_cfg = dict(configs["models"]["models"].get(model_name, {}))
                if smoke and model_name in _NN_MODELS:
                    model_cfg["epochs"] = smoke_epochs

                fold_results = []
                for fold_id in fold_ids:
                    result = run_fold(
                        dataset, target, fold_id,
                        model_name=model_name, model_cfg=model_cfg, seed=seed,
                    )
                    fold_results.append(result)
                    write_fold_predictions(
                        fold_id, dataset, target, "model", model_name,
                        result.y_true, result.y_pred, f"baseline_{model_name}",
                        outputs_dir=outputs_dir,
                    )
                agg = aggregate_folds(fold_results)
                summary_rows.append(
                    summary_row(
                        {"dataset": dataset, "target": target, "model": model_name},
                        agg.per_fold_metrics,
                    )
                )
                log.info(
                    "model=%-14s dataset=%-11s target=%-9s folds=%d MAE=%.4f",
                    model_name, dataset, target, len(fold_results), agg.overall_metrics["MAE"],
                )

    write_summary(summary_rows, "baseline_comparison_summary.csv", outputs_dir=outputs_dir)
    return summary_rows
