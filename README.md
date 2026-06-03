# revised_framework_v2

Temperature prediction research framework with TCN + CTGAN data augmentation.
Reproduces the original IEEE paper's 10-fold KFold validation and extends it with
a second dataset (Seoul) and a 7-model baseline comparison.

A single command — `python scripts/run_experiments.py` — runs the experiments and
then draws the full publication figure set, so the tables, arrays, and figures are
all reproduced from one entry point.

---

## Environment

Use the `tempPrediction` conda environment:

```bash
conda activate tempPrediction
# Python 3.x | torch 2.5.1 (CPU) | sklearn | sdv | ctgan | lightgbm | statsmodels | pmdarima
```

All scripts below assume this environment is active.

---

## Project Layout

```
revised_framework_v2/
├── configs/
│   ├── datasets.yaml       # dataset paths, lookback, n_splits
│   ├── models.yaml         # hyperparameters for all 7 models + CTGAN
│   └── experiments.yaml    # experiment matrices + smoke overrides
├── data/
│   ├── seattle/            # Seattle KFold .npy folds + seattle-weather.csv
│   └── seoul/              # Seoul KFold .npy folds + Bias_correction_ucl.csv
├── outputs/
│   ├── arrays/             # aggregated y_true/y_pred numpy arrays per workflow
│   ├── figures/            # publication figures (prediction, residual, distribution, temperature)
│   ├── predictions/        # per-fold prediction CSVs
│   ├── tables/             # experiment summary CSVs (pooled metrics) + distribution_metrics.csv
│   └── logs/               # run logs + environment snapshot
├── scripts/
│   ├── run_experiments.py                    # main entry point (experiments + figures)
│   ├── generate_new_dataset_kfold_assets.py  # Seoul .npy asset generator (heavy)
│   ├── regenerate_seattle_synthetic.py       # re-run CTGAN on Seattle folds
│   └── copy_required_assets.py               # one-time asset copy from source folders
└── src/                    # library code (utils, data, models, experiments, evaluation, plotting)
```

---

## Running Experiments

### Run all experiments + figures (workflow + baseline + multivariate)

```bash
python scripts/run_experiments.py --experiment all
```

This writes the summary tables, per-fold prediction CSVs, aggregated arrays, and —
unless `--smoke` / `--no-figures` is passed — the full figure set under
`outputs/figures/` plus `outputs/tables/distribution_metrics.csv`.

### Run a specific experiment

```bash
python scripts/run_experiments.py --experiment workflow
python scripts/run_experiments.py --experiment baseline
python scripts/run_experiments.py --experiment multivariate   # Seattle only
```

### Restrict to one dataset

```bash
python scripts/run_experiments.py --experiment all --dataset seattle
python scripts/run_experiments.py --experiment all --dataset seoul
```

When `--dataset` is passed, existing rows for the other dataset are preserved in
the summary tables, and only that dataset's prediction figures are redrawn.

### Other flags

```bash
python scripts/run_experiments.py --experiment all --smoke        # 1 fold / 1 epoch, no figures (code check)
python scripts/run_experiments.py --experiment all --no-figures   # experiments only, skip figures
python scripts/run_experiments.py --experiment all --outputs-dir /tmp/run1   # redirect every write
```

`--smoke` applies the tiny overrides from `configs/experiments.yaml` for fast
error-checking rather than real training. `--outputs-dir` sends tables, predictions,
arrays, **and** figures under one directory — useful for verifying a run without
touching the committed results.

Training is deterministic (per-fold seeding + a shared TCN cache), so repeated runs
reproduce identical arrays, tables, and figures.

---

## Experiments

### 1. Workflow Comparison

Compares three preprocessing + augmentation pipelines using TCN:

| Workflow | Description |
|----------|-------------|
| TCN | MinMax scaling (4 scalers), no augmentation |
| TCN + CTGAN | MinMax scaling (6 scalers), CTGAN synthetic data appended |
| TCN + CTGAN (processed) | StandardScaler + Yeo-Johnson (6 pairs), OneClassSVM-filtered synthetic data |

Runs on: Seattle and Seoul · 10-fold KFold. The aggregated arrays it saves feed the
prediction/residual figures for the best workflow (TCN + CTGAN processed).

### 2. Baseline Comparison

Compares 7 models on the same KFold splits (no CTGAN augmentation):

| Model | Type |
|-------|------|
| TCN | Temporal Convolutional Network (PyTorch) |
| LSTM | Long Short-Term Memory (PyTorch) |
| GRU | Gated Recurrent Unit (PyTorch) |
| SVR | Support Vector Regression |
| Random Forest | sklearn RandomForestRegressor |
| LightGBM | Gradient boosting |
| ARIMA | ARIMA(4,2,0) via statsmodels |

Runs on: Seattle and Seoul · 10-fold KFold.

### 3. Multivariate Comparison

Seattle-only experiment comparing feature configurations:

| Setting | Features |
|---------|----------|
| univariate | target column only |
| multivariate | temp_max, temp_min, precipitation, wind |
| multivariate_with_weather | above + label-encoded weather category |

---

## Metrics: 10-fold pooled (aggregated)

The summary tables report **pooled** metrics: every fold's test predictions are
concatenated into one array and MAE / RMSE / R² are computed **once** over the whole
set — not the mean of per-fold metrics. This matches the values annotated on the
prediction figures. Each row carries `n_folds` and `n_samples` alongside the single
`MAE`, `RMSE`, `R2` columns.

> Pooled MAE is close to the old fold-mean (folds are near-equal size), but pooled
> R²/RMSE differ noticeably because they are evaluated over the full temperature
> range rather than averaged across narrow per-fold ranges.

---

## Outputs

After a full run, results are written to `outputs/`:

| Path | Contents |
|------|----------|
| `outputs/tables/workflow_comparison_summary.csv` | Pooled MAE/RMSE/R² per workflow (10-fold aggregated) |
| `outputs/tables/baseline_comparison_summary.csv` | Same, per baseline model |
| `outputs/tables/multivariate_comparison_summary.csv` | Same, per multivariate setting |
| `outputs/tables/distribution_metrics.csv` | KS / JSD / Wasserstein / mean / std of synthetic vs original across the 4 TCP preprocessing stages |
| `outputs/predictions/<dataset>/<target>/<slug>/fold_<N>_predictions.csv` | Per-fold y_true, y_pred, residual |
| `outputs/arrays/<dataset>/<target>/<workflow>/` | Aggregated numpy arrays (all folds concatenated) |
| `outputs/figures/predictions/<dataset>/t{max,min}_{pred,resi}.{png,pdf}` | Prediction scatter + residual plots (TCN + CTGAN processed) |
| `outputs/figures/fig_dist_t{max,min}.{png,pdf}` | Synthetic-vs-original distribution across preprocessing stages |
| `outputs/figures/temperature_over_time_{seattle,seoul}.{png,pdf}` | Daily temperature series |
| `outputs/logs/environment.txt` | Package versions snapshot |

PDF figures are written without an embedded timestamp, so re-runs are byte-identical.

---

## Configuration

Edit `configs/models.yaml` to change model hyperparameters:

```yaml
models:
  tcn:
    hidden_size: 25
    kernel_size: 2
    num_levels: 4
    dropout: 0.0
    epochs: 5
    batch_size: 100
    learning_rate: 0.001

  arima:
    order: [4, 2, 0]
    trend: 'n'

  svr:
    kernel: rbf
    C: 0.04
    epsilon: 2.0
    gamma: scale

ctgan:
  epochs: 500
  num_samples: 1400
  enforce_min_max_values: true
```

Edit `configs/datasets.yaml` to change lookback window or number of folds:

```yaml
lookback: 5
n_splits: 10
shuffle: false
```

---

## Generating Seoul KFold Assets

The Seoul `.npy` fold files (already present in `data/seoul/`) were generated by:

```bash
python scripts/generate_new_dataset_kfold_assets.py
```

This is a heavy operation (~hours, runs CTGAN for each of 10 folds × 2 targets).
Only re-run if you need to regenerate the assets from scratch.

Use `--smoke` for a fast functional check (tiny CTGAN, 1 fold):

```bash
python scripts/generate_new_dataset_kfold_assets.py --smoke
```

---

## Utility Scripts

**Regenerate Seattle synthetic folds:**
```bash
python scripts/regenerate_seattle_synthetic.py
```
Re-runs CTGAN on the existing Seattle `ori_training` folds and overwrites
`synthetic_data_*.npy` in `data/seattle/`.

**Copy legacy assets (one-time setup):**
```bash
python scripts/copy_required_assets.py
```
Copies the source CSVs and pre-split `.npy` folds into `data/`. This is the only
script permitted to read from the sibling source folders; all runtime code reads
only from `revised_framework_v2/`.

> The distribution diagnostics and all publication figures are produced as part of
> `run_experiments.py` (see `src/plotting/`), so there are no separate plotting
> scripts to run.
