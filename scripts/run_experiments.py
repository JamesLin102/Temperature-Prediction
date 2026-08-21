"""Entry point: run one or all experiments, then draw the publication figures.

    python scripts/run_experiments.py --experiment all
    python scripts/run_experiments.py --experiment workflow
    python scripts/run_experiments.py --experiment baseline --smoke
    python scripts/run_experiments.py --experiment all --dataset seoul

A full run writes the summary tables (pooled 10-fold metrics), per-fold prediction
CSVs, aggregated arrays, and — unless ``--smoke``/``--no-figures`` — the publication
figure set under ``outputs/figures/`` plus ``outputs/tables/distribution_metrics.csv``.

``--smoke`` applies the tiny overrides from configs/experiments.yaml (1 fold,
1 epoch) for fast error-checking rather than real training (figures are skipped).

``--dataset`` restricts execution to a single dataset (e.g. ``seoul``).
Existing rows for OTHER datasets are preserved in the summary tables — Seattle
results are never overwritten when ``--dataset seoul`` is passed.

``--outputs-dir`` redirects every write (tables, predictions, arrays, figures)
under one directory (e.g. a temp dir) so a run can be verified without touching
the committed results.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.experiments import (  # noqa: E402
    run_baseline_comparison,
    run_multivariate_experiment,
    run_workflow_comparison,
)
from src.plotting.figures import generate_all_figures  # noqa: E402
from src.utils import config, paths  # noqa: E402
from src.utils.env_log import write_environment_log  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def _tables_base(outputs_dir) -> Path:
    return paths.OUTPUTS_DIR if outputs_dir is None else paths.resolve(outputs_dir)


def _load_other_rows(filename: str, exclude_dataset: str, outputs_dir=None) -> list[dict]:
    """Return rows from an existing summary CSV that belong to OTHER datasets."""
    p = _tables_base(outputs_dir) / "tables" / filename
    if not p.exists():
        return []
    df = pd.read_csv(p)
    if "dataset" not in df.columns:
        return []
    return df[df["dataset"] != exclude_dataset].to_dict("records")


def _write_merged(saved_rows: list, new_rows: list, filename: str, outputs_dir=None) -> None:
    """Write saved (other-dataset) rows + new rows into the summary CSV."""
    out = paths.ensure_dir(_tables_base(outputs_dir) / "tables") / filename
    pd.DataFrame(saved_rows + new_rows).to_csv(out, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the temperature-forecasting experiments."
    )
    parser.add_argument(
        "--experiment",
        choices=["workflow", "baseline", "multivariate", "all"],
        default="all",
        help="Which experiment to run.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny fast run (1 fold / 1 epoch) for error-checking, not real training.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Restrict to one dataset (e.g. 'seoul'). "
             "Existing rows for other datasets are preserved in the summary tables.",
    )
    parser.add_argument(
        "--outputs-dir",
        default=None,
        help="Redirect tables/predictions/arrays/figures here instead of the project outputs/.",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip figure + distribution-metrics generation (run experiments only).",
    )
    args = parser.parse_args()

    log = get_logger("run_experiments", log_file="outputs/logs/run_experiments.log")
    configs = config.load_all_configs()
    set_seed(configs["models"].get("seed", 7))

    log.info("Running experiment=%s smoke=%s dataset=%s outputs_dir=%s",
             args.experiment, args.smoke, args.dataset or "all",
             args.outputs_dir or "default")

    dataset_filter = [args.dataset] if args.dataset else None
    out_dir = args.outputs_dir

    if args.experiment in ("workflow", "all"):
        saved = _load_other_rows("workflow_comparison_summary.csv", args.dataset or "", out_dir) \
            if args.dataset else []
        new_rows = run_workflow_comparison.run(
            configs, smoke=args.smoke, datasets=dataset_filter, outputs_dir=out_dir, logger=log
        )
        if args.dataset and saved:
            _write_merged(saved, new_rows, "workflow_comparison_summary.csv", out_dir)

    if args.experiment in ("baseline", "all"):
        saved = _load_other_rows("baseline_comparison_summary.csv", args.dataset or "", out_dir) \
            if args.dataset else []
        new_rows = run_baseline_comparison.run(
            configs, smoke=args.smoke, datasets=dataset_filter, outputs_dir=out_dir, logger=log
        )
        if args.dataset and saved:
            _write_merged(saved, new_rows, "baseline_comparison_summary.csv", out_dir)

    if args.experiment in ("multivariate", "all"):
        if not args.dataset or args.dataset == "seattle":
            run_multivariate_experiment.run(
                configs, smoke=args.smoke, outputs_dir=out_dir, logger=log
            )
        else:
            log.info("Skipping multivariate (Seattle-only experiment).")

    if args.smoke:
        log.info("Smoke run: skipping figure generation.")
    elif args.no_figures:
        log.info("--no-figures: skipping figure generation.")
    else:
        generate_all_figures(
            configs, datasets=dataset_filter, outputs_dir=out_dir, logger=log,
        )

    env_log_path = None
    if out_dir is not None:
        env_log_path = paths.ensure_dir(paths.resolve(out_dir) / "logs") / "environment.txt"
    log_path = write_environment_log(
        output_path=env_log_path,
        extra={"experiment": args.experiment, "smoke": args.smoke, "dataset": args.dataset},
    )
    log.info("Environment log written to %s", log_path)
    log.info("Done.")


if __name__ == "__main__":
    main()
