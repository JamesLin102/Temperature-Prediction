"""TCN (Temporal Convolutional Network) — PyTorch.

Architecture faithfully reproduces keras-tcn ``TCN(25) + Dense(1)``:
  - nb_filters=25, kernel_size=3, dilations=(1,2,4,8,16,32), nb_stacks=1
  - use_batch_norm=False (keras-tcn default)
  - use_skip_connections=True: skip_out is taken BEFORE residual add (pre-residual
    conv output), summed across all blocks, fed to Dense(1) at last timestep
  - Xavier uniform init on all Conv1d weights; Adam eps=1e-7 (matches Keras default)
  - causal (left-padded) dilated convolutions, ReLU activations
  - MSE loss, Adam optimiser, epochs=50, batch=100
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .base_model import BaseModel

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _CausalConv1d(nn.Module):
    """1-D causal (left-padded) dilated convolution."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=0)
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.padding, 0))  # left-pad only -> causal
        return self.conv(x)


class _ResidualBlock(nn.Module):
    """Two dilated causal convs + ReLU + Dropout + residual.

    No BatchNorm — matches keras-tcn ``use_batch_norm=False`` default.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = _CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = _CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.drop = nn.Dropout(dropout)
        if in_channels != out_channels:
            ds = nn.Conv1d(in_channels, out_channels, kernel_size=1)
            nn.init.xavier_uniform_(ds.weight)
            nn.init.zeros_(ds.bias)
            self.downsample = ds
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.drop(F.relu(self.conv1(x)))
        out = self.drop(F.relu(self.conv2(out)))
        # skip_out: conv output before residual add (matches keras-tcn skip connection)
        skip_out = out
        main_out = F.relu(out + residual)
        return main_out, skip_out


class _TCNNet(nn.Module):
    """Stack of residual blocks with skip connections -> last timestep -> Linear(1).

    Implements keras-tcn ``use_skip_connections=True``: every residual block's
    output is accumulated (summed), and the sum feeds the final linear layer.
    """

    def __init__(self, input_size: int, num_filters: int, kernel_size: int, num_levels: int, dropout: float):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(num_levels):
            in_ch = input_size if i == 0 else num_filters
            self.blocks.append(_ResidualBlock(in_ch, num_filters, kernel_size, 2 ** i, dropout))
        self.fc = nn.Linear(num_filters, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size) -> (batch, input_size, seq_len)
        x = x.permute(0, 2, 1)
        skip_sum: torch.Tensor | None = None
        for block in self.blocks:
            x, skip_out = block(x)
            # accumulate skip outputs (pre-residual conv outputs, mirrors keras-tcn)
            skip_sum = skip_out if skip_sum is None else skip_sum + skip_out
        assert skip_sum is not None
        out = skip_sum[:, :, -1]            # last timestep -> (batch, num_filters)
        return self.fc(out).squeeze(-1)     # (batch,)


class TCNModel(BaseModel):
    """TCN regressor matching keras-tcn ``TCN(25) + Dense(1)``."""

    def __init__(
        self,
        hidden_size: int = 25,
        kernel_size: int = 3,
        num_levels: int = 6,
        dropout: float = 0.0,
        epochs: int = 10,
        batch_size: int = 100,
        learning_rate: float = 0.001,
    ):
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.num_levels = num_levels
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self._net: _TCNNet | None = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "TCNModel":
        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32).reshape(-1)
        input_size = X_train.shape[2]

        self._net = _TCNNet(
            input_size, self.hidden_size, self.kernel_size, self.num_levels, self.dropout
        ).to(_DEVICE)

        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.learning_rate, eps=1e-7)
        criterion = nn.MSELoss()
        dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

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
            raise RuntimeError("TCNModel.predict called before fit.")
        X_test = np.asarray(X_test, dtype=np.float32)
        self._net.eval()
        with torch.no_grad():
            preds = self._net(torch.from_numpy(X_test).to(_DEVICE)).cpu().numpy()
        return preds.astype(np.float32).reshape(-1)
