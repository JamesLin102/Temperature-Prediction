# Data

The two weather datasets belong to their original publishers and are **not**
redistributed in this repository. Download them yourself, put the CSVs in the
paths below, and one script rebuilds the exact fold assets the experiments read.

## 1. Download

| Dataset | Download from | Save as |
|---------|---------------|---------|
| Seattle | [Kaggle — Weather Prediction](https://www.kaggle.com/datasets/ananthr1/weather-prediction/data) | `data/seattle/seattle-weather.csv` |
| Seoul | [UCI — Bias correction of numerical prediction model temperature forecast](https://archive.ics.uci.edu/dataset/514/bias+correction+of+numerical+prediction+model+temperature+forecast) | `data/seoul/Bias_correction_ucl.csv` |

Keep the files exactly as downloaded — no renaming of columns, no reordering, no
cleaning. The Kaggle file is already named `seattle-weather.csv`. The UCI
download is a zip; the CSV inside it is `Bias_correction_ucl.csv`.

Expected contents:

| File | Rows | Columns used |
|------|------|--------------|
| `seattle-weather.csv` | 1,461 | `date`, `temp_max`, `temp_min` (plus `precipitation`, `wind`, `weather` for the multivariate experiment) |
| `Bias_correction_ucl.csv` | 7,752 | `Date`, `station`, `Next_Tmax`, `Next_Tmin` |

## 2. Build the fold assets

```bash
python scripts/prepare_data.py --dataset all
```

This writes, for each dataset and each target (`temp_max`, `temp_min`):

```
data/<dataset>/<target>/
├── ori_training_data_1.npy … _10.npy    real training windows, (N, 6)
├── testing_data_1.npy      … _10.npy    real test windows,     (N, 6)
└── synthetic_data_1.npy    … _10.npy    CTGAN samples,         (1400, 6)
```

Each row is five lookback values plus the next-day target. Sequences use the
off-by-one convention of the original implementation (Seattle: 1461 rows →
1455 sequences; Seoul: 7,750 valid rows → 7,600 station-safe sequences) and are
split with `KFold(n_splits=10, shuffle=False)`.

Flags:

| Flag | Effect |
|------|--------|
| `--dataset seattle\|seoul\|all` | Which dataset to prepare (default `all`) |
| `--skip-synthetic` | Real folds only — finishes in seconds, no CTGAN training |
| `--synthetic-only` | Keep the real folds, retrain CTGAN only |
| `--smoke` | Tiny CTGAN settings for a fast functional check |

## 3. What is and is not reproducible

**The real folds are exact.** `ori_training_data_*.npy` and `testing_data_*.npy`
regenerate bit-for-bit identical to the arrays used for the published results —
sequence building and `KFold(shuffle=False)` are fully deterministic.

**The synthetic folds are not.** CTGAN training is stochastic and depends on
library versions, so `synthetic_data_*.npy` differs from run to run. The
published metrics are therefore reproducible to within the usual CTGAN sampling
noise (of the order of ±0.01 MAE), not to the last decimal. The relative
ordering the paper reports — the processed workflow beating both the TCN
baseline and unprocessed augmentation — has been stable across regenerations.

`--skip-synthetic` is enough for the baseline comparison and the multivariate
experiment, which never touch the synthetic data. The workflow comparison needs
the synthetic folds.

## 4. Citing the data

- Seattle: Kaggle dataset "Weather Prediction" by ANANTH R.
- Seoul: Cho, D., Yoo, C., Im, J., Cha, D. (2020). *Comparative assessment of
  various machine learning-based bias correction methods for numerical weather
  prediction model forecasts of extreme air temperatures in urban areas.*
  Earth and Space Science. UCI Machine Learning Repository, CC BY 4.0.
