"""Canonical column names, target maps and feature lists for both datasets."""
from __future__ import annotations

# --- Seattle raw CSV --------------------------------------------------------
SEATTLE_DATE_COL = "date"
SEATTLE_COLUMNS = ["date", "precipitation", "temp_max", "temp_min", "wind", "weather"]
SEATTLE_MULTIVARIATE_FEATURES = ["temp_max", "temp_min", "precipitation", "wind"]
SEATTLE_CATEGORICAL_COL = "weather"

# --- New dataset (Bias_correction_ucl.csv) ----------------------------------
NEW_DATASET_DATE_COL = "Date"
NEW_DATASET_STATION_COL = "station"
NEW_DATASET_TARGET_MAP = {"Next_Tmax": "temp_max", "Next_Tmin": "temp_min"}

# --- Canonical targets shared downstream ------------------------------------
TARGETS = ["temp_max", "temp_min"]

# Aggregated-array filenames, kept identical to the original 2024 implementation.
LEGACY_PLOT_ARRAY_NAMES = [
    "T_y_true_all",
    "T_y_pred_all",
    "TC_y_true_all",
    "TC_y_pred_all",
    "TCP_y_true_all",
    "TCP_y_pred_all",
]


def resolve_target_column(dataset: str, target: str) -> str:
    """Return the source CSV column for a canonical target name.

    Seattle already uses canonical names; the new dataset uses ``Next_Tmax`` /
    ``Next_Tmin`` which map to ``temp_max`` / ``temp_min``.
    """
    if dataset == "seoul":
        reverse = {canonical: raw for raw, canonical in NEW_DATASET_TARGET_MAP.items()}
        if target not in reverse:
            raise KeyError(f"Unknown seoul target: {target}")
        return reverse[target]
    return target
