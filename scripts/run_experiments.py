"""Entry point: run one or all revised_framework_v2 experiments.

    python scripts/run_experiments.py --experiment all
    python scripts/run_experiments.py --experiment workflow
    python scripts/run_experiments.py --experiment baseline --smoke
    python scripts/run_experiments.py --experiment all --dataset seoul

``--smoke`` applies the tiny overrides from configs/experiments.yaml (1 fold,
1 epoch) for fast error-checking rather than real training.

``--dataset`` restricts execution to a single dataset (e.g. ``seoul``).
Existing rows for OTHER datasets are preserved in the summary tables — Seattle
results are never overwritten when ``--dataset seoul`` is passed.
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
from src.utils import config, paths  # noqa: E402
from src.utils.env_log import write_environment_log  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def _load_other_rows(filename: str, exclude_dataset: str) -> list[dict]:
    """Return rows from an existing summary CSV that belong to OTHER datasets."""
    p = paths.OUTPUTS_DIR / "tables" / filename
    if not p.exists():
        return []
    df = pd.read_csv(p)
    if "dataset" not in df.columns:
        return []
    return df[df["dataset"] != exclude_dataset].to_dict("records")


def _write_merged(saved_rows: list, new_rows: list, filename: str) -> None:
    """Write saved (other-dataset) rows + new rows into the summary CSV."""
    out = paths.ensure_dir(paths.OUTPUTS_DIR / "tables") / filename
    pd.DataFrame(saved_rows + new_rows).to_csv(out, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run revised_framework_v2 experiments.")
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
    args = parser.parse_args()

    log = get_logger("run_experiments", log_file="outputs/logs/run_experiments.log")
    configs = config.load_all_configs()
    set_seed(configs["models"].get("seed", 7))

    log.info("Running experiment=%s smoke=%s dataset=%s",
             args.experiment, args.smoke, args.dataset or "all")

    dataset_filter = [args.dataset] if args.dataset else None

    if args.experiment in ("workflow", "all"):
        saved = _load_other_rows("workflow_comparison_summary.csv", args.dataset or "") \
            if args.dataset else []
        new_rows = run_workflow_comparison.run(
            configs, smoke=args.smoke, datasets=dataset_filter, logger=log
        )
        if args.dataset and saved:
            _write_merged(saved, new_rows, "workflow_comparison_summary.csv")

    if args.experiment in ("baseline", "all"):
        saved = _load_other_rows("baseline_comparison_summary.csv", args.dataset or "") \
            if args.dataset else []
        new_rows = run_baseline_comparison.run(
            configs, smoke=args.smoke, datasets=dataset_filter, logger=log
        )
        if args.dataset and saved:
            _write_merged(saved, new_rows, "baseline_comparison_summary.csv")

    if args.experiment in ("multivariate", "all"):
        if not args.dataset or args.dataset == "seattle":
            run_multivariate_experiment.run(configs, smoke=args.smoke, logger=log)
        else:
            log.info("Skipping multivariate (Seattle-only experiment).")

    log_path = write_environment_log(
        extra={"experiment": args.experiment, "smoke": args.smoke, "dataset": args.dataset}
    )
    log.info("Environment log written to %s", log_path)
    log.info("Done.")


if __name__ == "__main__":
    main()
