"""High-level architectures refactored from notebook 03."""

import torch
from torch import nn


class NeuralForecaster(nn.Module):
    def __init__(self, name: str, hidden_size: int = 64, lookback: int = 168):
        super().__init__()
        self.name = name
        if name in ("RNN", "LSTM"):
            cls = nn.RNN if name == "RNN" else nn.LSTM
            self.encoder = cls(1, hidden_size, num_layers=1, batch_first=True)
        else:
            self.projection = nn.Linear(1, hidden_size)
            self.position = nn.Parameter(torch.randn(1, lookback, hidden_size))
            layer = nn.TransformerEncoderLayer(hidden_size, 4, dim_feedforward=128, dropout=0.1, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.fc = nn.Linear(hidden_size, 24)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.name == "Transformer":
            final = self.encoder(self.projection(x) + self.position)[:, -1]
        else:
            _, hidden = self.encoder(x)
            final = (hidden[0] if self.name == "LSTM" else hidden)[-1]
        return self.fc(final)
