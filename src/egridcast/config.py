"""Portable runtime and training configuration."""

import os
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_NAMES = ("SARIMA", "XGBoost", "RNN", "LSTM", "Transformer")


@dataclass(frozen=True)
class Settings:
    data_path: Path = Path(os.getenv("EGRIDCAST_DATA", str(ROOT / "data/processed/PRICE_AND_DEMAND_2025_26_VIC1.csv")))
    artifact_dir: Path = Path(os.getenv("EGRIDCAST_ARTIFACTS", str(ROOT / "artifacts")))
    cors_origin: str = os.getenv("EGRIDCAST_CORS_ORIGIN", "http://localhost:5173")


@dataclass
class TrainConfig:
    lookback: int = 168
    horizon: int = 24
    epochs: int = 30
    patience: int = 5
    batch_size: int = 32
    hidden_size: int = 64
    learning_rate: float = 0.001
    seed: int = 42
    folds: int = 3
    stride: int = 24
    trees: int = 800
    sarima_maxiter: int = 100

    def __post_init__(self) -> None:
        if self.lookback != 168 or self.horizon != 24:
            raise ValueError("This protocol requires 168 hours of history and a 24-hour horizon")
        if (
            min(self.epochs, self.patience, self.batch_size, self.folds, self.stride, self.trees, self.sarima_maxiter)
            < 1
        ):
            raise ValueError("Training and evaluation counts must be positive")
        if self.hidden_size < 4 or self.hidden_size % 4:
            raise ValueError("Hidden size must be a positive multiple of four for attention heads")
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")

    def to_dict(self) -> dict:
        return asdict(self)
