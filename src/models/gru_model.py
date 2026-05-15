"""GRU baseline — PyTorch.

CPU device; no internal scaling — the baseline runner applies the same MinMax
preprocessing to all neural-net models for a fair comparison.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .base_model import BaseModel

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _GRUNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :]).squeeze(-1)  # last timestep


class GRUModel(BaseModel):
    """GRU regressor (last-timestep -> Dense(1))."""

    def __init__(
        self,
        hidden_size: int = 25,
        num_layers: int = 1,
        dropout: float = 0.0,
        epochs: int = 10,
        batch_size: int = 100,
        learning_rate: float = 0.001,
    ):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self._net: _GRUNet | None = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "GRUModel":
        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32).reshape(-1)
        input_size = X_train.shape[2]

        self._net = _GRUNet(input_size, self.hidden_size, self.num_layers, self.dropout).to(_DEVICE)
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        loader = DataLoader(
            TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
            batch_size=self.batch_size,
            shuffle=True,
        )

        self._net.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                xb, yb = xb.to(_DEVICE), yb.to(_DEVICE)
                optimizer.zero_grad()
                loss = criterion(self._net(xb), yb)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("GRUModel.predict called before fit.")
        X_test = np.asarray(X_test, dtype=np.float32)
        self._net.eval()
        with torch.no_grad():
            preds = self._net(torch.from_numpy(X_test).to(_DEVICE)).cpu().numpy()
        return preds.astype(np.float32).reshape(-1)
