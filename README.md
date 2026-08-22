# Temperature Forecasting with CTGAN Data Augmentation and a Temporal Convolutional Network

Reference implementation for the paper **"Enhancing Temperature Forecasting for Sustainable Energy
Systems Using CTGAN-Based Data Augmentation and Temporal Convolutional Networks"**
(Ping-Huan Kuo, Yu-Sian Lin, Yu-Chih Chiu — National Cheng Kung University), accepted at
*Energy Reports* (Elsevier), 2026.

**Project page:** https://jameslin102.github.io/Temperature-Prediction/

Next-day maximum and minimum temperature forecasting from a 5-day window, on two geographically
distinct datasets (Seattle and Seoul). The contribution is not the TCN or the CTGAN, but the fixed
preprocessing stage placed between them — one-class SVM outlier removal, standardisation and the
Yeo–Johnson transformation — which aligns the synthetic distribution with the real one and turns
CTGAN augmentation from harmful into helpful, without any CTGAN hyperparameter search.

The two datasets belong to their original publishers and are **not** redistributed here. You
download the CSVs yourself and one script rebuilds the exact fold assets used in the paper — see
[Data setup](#data-setup). Everything else needed to check the results is in the repository: the
result tables, the per-fold predictions, the aggregated arrays and the publication figures.

---

## Quickstart

```bash
git clone https://github.com/JamesLin102/Temperature-Prediction.git
cd Temperature-Prediction
pip install -r requirements.txt
```

Download the two datasets into `data/` and build the fold assets — details in
[Data setup](#data-setup):

```bash
python scripts/prepare_data.py --dataset all
```

Then check that everything is wired up (1 fold, 1 epoch, about a minute, writes nothing into the
committed results):

```bash
python scripts/run_experiments.py --experiment all --smoke --outputs-dir _smoke_check
```

Reproduce the full result set (CPU only; the ARIMA baseline on the Seoul folds dominates the
runtime):

```bash
python scripts/run_experiments.py --experiment all
```

That single command runs the three experiments and then draws the full figure set, overwriting
`outputs/`. Training is deterministic — per-fold seeding plus a shared TCN cache — so repeated runs
reproduce identical arrays, tables and figures.

Python 3.12 is recommended. Every experiment runs on CPU; no GPU is required.

---

## Repository layout

```
.
├── configs/
│   ├── datasets.yaml       # dataset paths, lookback, n_splits
│   ├── models.yaml         # hyperparameters for all 7 models + CTGAN
│   └── experiments.yaml    # experiment matrices + smoke overrides
├── data/                   # empty: you download the CSVs here, see data/README.md
│   ├── seattle/            # -> seattle-weather.csv + 10-fold .npy assets per target
│   └── seoul/              # -> Bias_correction_ucl.csv + 10-fold .npy assets per target
├── outputs/                # committed results (see "Outputs" below)
├── scripts/
│   ├── prepare_data.py     # build the fold assets from the downloaded CSVs
│   └── run_experiments.py  # main entry point (experiments + figures)
├── src/                    # library code (data, models, preprocessing, evaluation, plotting)
└── docs/                   # the project page published via GitHub Pages
```

Every path resolves relative to the repository root, so the project can be cloned anywhere and run
without configuration.

---

## Data setup

The datasets belong to their original publishers, so they are not redistributed here. Download
them and save them under the exact paths below:

| Dataset | Download from | Save as | Records | Period |
|---------|---------------|---------|---------|--------|
| Seattle | [Kaggle — Weather Prediction](https://www.kaggle.com/datasets/ananthr1/weather-prediction/data) | `data/seattle/seattle-weather.csv` | 1,461 | 2012-01-01 – 2015-12-31, daily |
| Seoul | [UCI — Bias correction of numerical prediction model temperature forecast](https://archive.ics.uci.edu/dataset/514/bias+correction+of+numerical+prediction+model+temperature+forecast) | `data/seoul/Bias_correction_ucl.csv` | 7,750 | Summers (Jun–Aug) 2013 – 2017, 26 stations |

Then build the fold assets:

```bash
python scripts/prepare_data.py --dataset all          # both datasets, with CTGAN synthetic folds
python scripts/prepare_data.py --dataset all --skip-synthetic   # real folds only, seconds
python scripts/prepare_data.py --dataset seoul        # one dataset at a time
```

This writes `ori_training_data_<fold>.npy`, `testing_data_<fold>.npy` and
`synthetic_data_<fold>.npy` for folds 1–10 into `data/<dataset>/<target>/`. Each row of an
`(N, 6)` array is five lookback values plus the next-day target. Sequences follow the off-by-one
convention of the original implementation and are split with `KFold(n_splits=10, shuffle=False)`.

**The real folds regenerate bit-for-bit identically** to the ones behind the published results —
that path is fully deterministic and was verified against the original arrays. **The synthetic
folds do not:** CTGAN training is stochastic and version-dependent, so the workflow-comparison
metrics land within CTGAN sampling noise of the published values rather than on them exactly. The
baseline and multivariate experiments never touch the synthetic data, so `--skip-synthetic` is
enough for those.

Building the synthetic folds is the slow part — CTGAN trains 500 epochs per fold, 10 folds × 2
targets × 2 datasets. `--smoke` swaps in tiny CTGAN settings for a fast functional check.

Only the target temperature variable feeds the model — adding the other weather variables lowered
accuracy (see the multivariate experiment). Please cite the original data sources; the UCI dataset
is released under CC BY 4.0 (Cho, D., Yoo, C., Im, J., Cha, D., 2020). Full details, including the
expected CSV contents, are in [`data/README.md`](data/README.md).

---

## Experiments

### 1. Workflow comparison

| Workflow | Description |
|----------|-------------|
| TCN | MinMax scaling, no augmentation |
| TCN + CTGAN | MinMax scaling, raw CTGAN synthetic data appended |
| TCN + CTGAN (processed) | One-class SVM filtering + StandardScaler + Yeo–Johnson on the synthetic data |

Runs on Seattle and Seoul, 10-fold KFold. The aggregated arrays it saves feed the prediction and
residual figures for the best workflow.

### 2. Baseline comparison

Seven models on identical splits with no augmentation: TCN, LSTM, GRU, SVR, Random Forest,
LightGBM, ARIMA(4,2,0).

### 3. Multivariate input comparison (Seattle only)

`univariate` (target column only) vs `multivariate` (temp_max, temp_min, precipitation, wind) vs
`multivariate_with_weather` (plus the label-encoded weather category).

```bash
python scripts/run_experiments.py --experiment workflow
python scripts/run_experiments.py --experiment baseline
python scripts/run_experiments.py --experiment multivariate   # Seattle only
python scripts/run_experiments.py --experiment all --dataset seattle
```

When `--dataset` is passed, existing rows for the other dataset are preserved in the summary
tables, and only that dataset's prediction figures are redrawn.

Other flags:

| Flag | Effect |
|------|--------|
| `--smoke` | 1 fold / 1 epoch, figures skipped — a fast error check, not real training |
| `--no-figures` | Run the experiments only |
| `--outputs-dir DIR` | Redirect every write (tables, predictions, arrays, figures, environment log) to `DIR` |

---

## Results

Pooled 10-fold metrics of the proposed workflow, as published:

| Dataset | Target | MAE | RMSE | R² |
|---------|--------|-----|------|-----|
| Seattle | maximum | 2.397 | 3.039 | 0.829 |
| Seattle | minimum | 1.664 | 2.136 | 0.819 |
| Seoul | maximum | 1.863 | 2.346 | 0.441 |
| Seoul | minimum | 1.077 | 1.395 | 0.681 |

The proposed workflow beats both the TCN baseline and unprocessed CTGAN augmentation in all four
dataset–target combinations. Full tables, including the seven-model baseline comparison, are in
`outputs/tables/` and on the [project page](https://jameslin102.github.io/Temperature-Prediction/).

### Metrics are pooled, not averaged

Every fold's test predictions are concatenated into one array and MAE / RMSE / R² are computed
**once** over the whole set, rather than averaging per-fold metrics. Pooled MAE is close to the
fold-mean because the folds are near-equal in size, but pooled R² and RMSE differ noticeably: they
are evaluated over the full temperature range instead of narrow per-fold ranges. The values on the
prediction figures use the same pooled definition.

### Outputs

| Path | Contents |
|------|----------|
| `outputs/tables/workflow_comparison_summary.csv` | Pooled MAE/RMSE/R² per workflow |
| `outputs/tables/baseline_comparison_summary.csv` | Same, per baseline model |
| `outputs/tables/multivariate_comparison_summary.csv` | Same, per input configuration |
| `outputs/tables/distribution_metrics.csv` | KS / JSD / Wasserstein / mean / std of synthetic vs original across the preprocessing stages |
| `outputs/predictions/<dataset>/<target>/<slug>/fold_<N>_predictions.csv` | Per-fold y_true, y_pred, residual |
| `outputs/arrays/<dataset>/<target>/<workflow>/` | All folds concatenated, as numpy arrays |
| `outputs/figures/` | Prediction, residual, distribution and temperature figures (PNG + PDF) |
| `outputs/logs/environment.txt` | Platform, Python and package versions of the published run |

`outputs/` is committed, so the published results can be inspected without downloading the data or
running anything.

PDF figures are written without an embedded timestamp, so re-runs are byte-identical.

---

## Configuration

Hyperparameters live in `configs/models.yaml`:

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

ctgan:
  epochs: 500
  num_samples: 1400
  enforce_min_max_values: true
```

and the split settings in `configs/datasets.yaml`:

```yaml
lookback: 5
n_splits: 10
shuffle: false
```

---

## Rebuilding only the synthetic folds

If you already have the real folds and only want to redraw the CTGAN samples:

```bash
python scripts/prepare_data.py --dataset all --synthetic-only
```

---

## Citation

```bibtex
@article{kuo2026enhancing,
  title   = {Enhancing Temperature Forecasting for Sustainable Energy Systems Using
             CTGAN-Based Data Augmentation and Temporal Convolutional Networks},
  author  = {Kuo, Ping-Huan and Lin, Yu-Sian and Chiu, Yu-Chih},
  journal = {Energy Reports},
  year    = {2026},
  note    = {In press}
}
```

The original 2024 implementation that this framework was rebuilt from is preserved at the git tag
[`legacy-2024`](https://github.com/JamesLin102/Temperature-Prediction/tree/legacy-2024).

## License

Code released under the [MIT License](LICENSE). The bundled datasets remain under the terms of
their original sources.
