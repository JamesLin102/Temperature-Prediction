# revised_framework_v2

Temperature prediction research framework with TCN + CTGAN data augmentation.
Reproduces the original IEEE paper's 10-fold KFold validation and extends it with
a second dataset (Seoul) and a 7-model baseline comparison.

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
│   ├── experiments.yaml    # experiment matrices + smoke overrides
│   └── plotting.yaml       # figure sizes, colors, filenames
├── data/
│   ├── seattle/            # Seattle KFold .npy folds + seattle-weather.csv
│   ├── seoul/              # Seoul KFold .npy folds + Bias_correction_ucl.csv
│   └── raw/                # (empty after assets moved)
├── outputs/
│   ├── arrays/             # aggregated y_true/y_pred numpy arrays per workflow
│   ├── figures/            # PNG figures (pred, residuals, comparison)
│   ├── predictions/        # per-fold prediction CSVs
│   ├── tables/             # experiment summary CSVs
│   └── logs/               # run logs + environment snapshot
├── scripts/
│   ├── run_experiments.py                  # main entry point
│   ├── generate_new_dataset_kfold_assets.py  # Seoul .npy asset generator (heavy)
│   ├── analyze_synthetic_distribution.py   # CTGAN distribution diagnostics
│   ├── regenerate_seattle_synthetic.py     # re-run CTGAN on Seattle folds
│   └── copy_required_assets.py            # one-time asset copy from source folders
└── src/                    # library code (utils, data, models, experiments, evaluation, plotting)
```

---

## Running Experiments

### Run all experiments (workflow + baseline + multivariate)

```bash
python scripts/run_experiments.py --experiment all
```

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
the summary tables.

### Fast smoke run (1 fold, 1 epoch — for code verification only)

```bash
python scripts/run_experiments.py --experiment all --smoke
```

---

## Experiments

### 1. Workflow Comparison

Compares three preprocessing + augmentation pipelines using TCN:

| Workflow | Description |
|----------|-------------|
| TCN | MinMax scaling (4 scalers), no augmentation |
| TCN + CTGAN | MinMax scaling (6 scalers), CTGAN synthetic data appended |
| TCN + CTGAN (processed) | StandardScaler + Yeo-Johnson (6 pairs), OneClassSVM-filtered synthetic data |

Runs on: Seattle and Seoul · 10-fold KFold · outputs figures for Seattle.

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

## Outputs

After a full run, results are written to `outputs/`:

| Path | Contents |
|------|----------|
| `outputs/tables/workflow_comparison_summary.csv` | Mean ± std MAE/RMSE/R² per workflow across 10 folds |
| `outputs/tables/baseline_comparison_summary.csv` | Same, per baseline model |
| `outputs/tables/multivariate_comparison_summary.csv` | Same, per multivariate setting |
| `outputs/predictions/<dataset>/<target>/<workflow>/fold_<N>_predictions.csv` | Per-fold y_true, y_pred, residual |
| `outputs/arrays/<dataset>/<target>/<workflow>/` | Aggregated numpy arrays (all folds concatenated) |
| `outputs/figures/seattle/<target>/` | PNG figures: pred scatter, residual plots, comparison lines |
| `outputs/logs/environment.txt` | Package versions snapshot |

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

**Analyze CTGAN synthetic data distribution:**
```bash
python scripts/analyze_synthetic_distribution.py --fold 1 --target temp_max
python scripts/analyze_synthetic_distribution.py --fold 1 --target all
```
Outputs histograms and summary stats comparing original vs raw CTGAN vs filtered CTGAN.

**Regenerate Seattle synthetic folds:**
```bash
python scripts/regenerate_seattle_synthetic.py
```
Re-runs CTGAN on the existing Seattle `ori_training` folds and overwrites
`synthetic_data_*.npy` in `data/seattle/`.
