"""CTGAN synthetic-data augmenter (SDV).

Mirrors ``data_gan.py`` of the original 2024 implementation (git tag
``legacy-2024``): MinMax-scale an ``(N, 6)`` fold array, fit an SDV
``CTGANSynthesizer`` on it, sample synthetic rows, then inverse-transform back
to the original range.

Used only by ``scripts/prepare_data.py``, which builds the synthetic ``.npy``
folds under ``data/``. All SDV imports and training happen inside methods, so
importing this module never triggers a CTGAN run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

_COLUMNS = ["temp_1", "temp_2", "temp_3", "temp_4", "temp_5", "temp_6"]


class CTGANAugmenter:
    """Train CTGAN on a ``(N, 6)`` array and sample synthetic rows in the same format."""

    def __init__(
        self,
        epochs: int = 500,
        num_samples: int = 1400,
        enforce_min_max_values: bool = True,
        verbose: bool = False,
    ):
        self.epochs = epochs
        self.num_samples = num_samples
        self.enforce_min_max_values = enforce_min_max_values
        self.verbose = verbose

    def fit_sample(self, data_six_col: np.ndarray) -> np.ndarray:
        """Train CTGAN on a ``(N, 6)`` array; return ``(num_samples, 6)`` synthetic data."""
        from sdv.metadata import SingleTableMetadata
        from sdv.single_table import CTGANSynthesizer

        data = np.asarray(data_six_col, dtype=np.float64)
        if data.ndim != 2 or data.shape[1] != 6:
            raise ValueError(f"Expected a (N, 6) array, got shape {data.shape}")

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(data)
        frame = pd.DataFrame(scaled, columns=_COLUMNS)

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=frame)
        metadata.validate()

        synthesizer = CTGANSynthesizer(
            metadata,
            enforce_min_max_values=self.enforce_min_max_values,
            epochs=self.epochs,
            verbose=self.verbose,
        )
        synthesizer.fit(frame)
        synthetic = synthesizer.sample(num_rows=self.num_samples)
        synthetic = scaler.inverse_transform(synthetic)
        return np.asarray(synthetic, dtype=np.float64)
