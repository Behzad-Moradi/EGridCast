"""Expanding-window evaluation on common, complete 24-hour origins."""

import numpy as np
import pandas as pd
from egridcast.models import Forecaster


def forecast_origins(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp, stride: int = 24) -> list[int]:
    if stride < 1:
        raise ValueError("Stride must be positive")
    eligible = [i for i in range(168, len(series) - 24 + 1) if series.index[i] >= start and series.index[i + 23] <= end]
    if not eligible:
        raise ValueError("No complete 24-hour forecast origins in evaluation interval")
    return eligible[::stride]


def expanding_folds(series: pd.Series, folds: int) -> list[tuple[pd.Series, pd.Series]]:
    if folds < 1:
        raise ValueError("At least one expanding fold is required")
    edges = np.linspace(max(192, len(series) // 2), len(series), folds + 1, dtype=int)
    result = [(series.iloc[: edges[i]], series.iloc[edges[i] : edges[i + 1]]) for i in range(folds)]
    if any(len(val) < 24 for _, val in result):
        raise ValueError("Too many folds for available data")
    return result


def score_predictions(actual: np.ndarray, predicted: np.ndarray) -> dict:
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if actual.shape != predicted.shape or actual.ndim != 2 or actual.shape[1] != 24:
        raise ValueError("Metrics require matching N by 24 matrices")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Nonfinite metrics inputs")
    error = np.abs(actual - predicted)
    return {
        "mae": float(error.mean()),
        "rmse": float(np.sqrt(np.mean((actual - predicted) ** 2))),
        "mape": float(np.mean(error / np.maximum(np.abs(actual), 1e-8)) * 100),
        "mae_by_horizon": error.mean(axis=0).tolist(),
        "origins": len(actual),
        "horizon_hours": 24,
    }


def evaluate(model: Forecaster, series: pd.Series, origins: list[int], split: str) -> tuple[dict, list[dict]]:
    predictions, actuals, rows = [], [], []
    for i in origins:
        prediction = model.forecast(series.iloc[:i])
        actual = series.iloc[i : i + 24].to_numpy()
        predictions.append(prediction)
        actuals.append(actual)
        for h in range(24):
            rows.append(
                {
                    "split": split,
                    "forecast_origin": series.index[i - 1].isoformat(),
                    "timestamp": series.index[i + h].isoformat(),
                    "horizon": h + 1,
                    "actual_demand_mw": float(actual[h]),
                    "predicted_demand_mw": float(prediction[h]),
                }
            )
    return score_predictions(np.asarray(actuals), np.asarray(predictions)), rows
