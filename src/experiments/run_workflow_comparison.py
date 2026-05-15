"""Experiment 7.1 — workflow comparison.

datasets x targets x {TCN, TCN + CTGAN, TCN + CTGAN (processed)}. Reads pre-split
legacy ``.npy`` folds; a dataset with no assets (new_dataset this pass) is skipped
gracefully. Seattle additionally gets the legacy figure set.
"""
from __future__ import annotations

from ..data.dataset_loader import fold_assets_available
from ..evaluation.aggregator import aggregate_folds, legacy_array_prefix
from ..evaluation.export_predictions import save_aggregated_arrays, write_fold_predictions
from ..evaluation.summary_tables import summary_row, write_summary
from ..plotting.figure_exporter import export_comparison_figures, export_workflow_figures
from ..utils.logger import get_logger
from .fold_runner import run_fold

_WORKFLOW_SLUG = {
    "TCN": "TCN",
    "TCN + CTGAN": "TCN_CTGAN",
    "TCN + CTGAN (processed)": "TCN_CTGAN_processed",
}


def run(
    configs: dict,
    *,
    smoke: bool = False,
    datasets=None,
    targets=None,
    fold_limit: int | None = None,
    outputs_dir=None,
    logger=None,
) -> list[dict]:
    """Run the workflow comparison; return the list of summary rows."""
    log = logger or get_logger("workflow_comparison")
    exp = configs["experiments"]["workflow_comparison"]
    ds_list = datasets or exp["datasets"]
    tg_list = targets or exp["targets"]
    workflows = exp["workflows"]
    seed = configs["models"].get("seed", 7)
    n_splits = configs["datasets"]["n_splits"]

    model_cfg = dict(configs["models"]["models"]["tcn"])
    if smoke:
        model_cfg["epochs"] = configs["experiments"]["smoke"]["epochs"]
        if fold_limit is None:
            fold_limit = configs["experiments"]["smoke"]["n_splits"]

    fold_ids = list(range(1, n_splits + 1))
    if fold_limit is not None:
        fold_ids = fold_ids[:fold_limit]

    summary_rows: list[dict] = []
    for dataset in ds_list:
        for target in tg_list:
            if not fold_assets_available(
                dataset, target, n_splits, kinds=("ori_training", "synthetic", "testing")
            ):
                log.warning("Skipping %s/%s: KFold .npy assets not generated.", dataset, target)
                continue

            arrays_by_workflow: dict = {}
            arrays_by_prefix: dict = {}
            for workflow in workflows:
                slug = _WORKFLOW_SLUG[workflow]
                prefix = legacy_array_prefix(workflow)
                fold_results = []
                for fold_id in fold_ids:
                    result = run_fold(
                        dataset, target, fold_id,
                        workflow=workflow, model_cfg=model_cfg, seed=seed,
                    )
                    fold_results.append(result)
                    write_fold_predictions(
                        fold_id, dataset, target, "workflow", workflow,
                        result.y_true, result.y_pred, slug, outputs_dir=outputs_dir,
                    )
                agg = aggregate_folds(fold_results)
                save_aggregated_arrays(
                    agg.y_true_all, agg.y_pred_all, dataset, target, slug, prefix,
                    outputs_dir=outputs_dir,
                )
                arrays_by_workflow[workflow] = (agg.y_true_all, agg.y_pred_all)
                arrays_by_prefix[prefix] = (agg.y_true_all, agg.y_pred_all)
                summary_rows.append(
                    summary_row(
                        {"dataset": dataset, "target": target, "workflow": workflow},
                        agg.per_fold_metrics,
                    )
                )
                log.info(
                    "workflow=%-26s dataset=%-11s target=%-9s folds=%d MAE=%.4f",
                    workflow, dataset, target, len(fold_results), agg.overall_metrics["MAE"],
                )

            if dataset == "seattle" and {"T", "TC", "TCP"}.issubset(arrays_by_prefix):
                export_workflow_figures(
                    dataset, target, arrays_by_workflow, configs["plotting"], outputs_dir=outputs_dir
                )
                export_comparison_figures(
                    dataset, target, arrays_by_prefix, configs["plotting"], outputs_dir=outputs_dir
                )

    write_summary(summary_rows, "workflow_comparison_summary.csv", outputs_dir=outputs_dir)
    return summary_rows
