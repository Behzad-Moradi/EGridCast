"""Notebook hourly means, strict validation, and origin-aligned features."""

from pathlib import Path
import numpy as np
import pandas as pd

TIMEZONE = "Etc/GMT-10"  # AEMO market time: fixed AEST, including summer.


def load_hourly(path: Path) -> tuple[pd.Series, dict]:
    df = pd.read_csv(path)
    required = {"SETTLEMENTDATE", "TOTALDEMAND", "REGION"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(df.columns))}")
    if not df.REGION.eq("VIC1").all():
        raise ValueError("Only VIC1 records are supported")
    index = pd.DatetimeIndex(pd.to_datetime(df.SETTLEMENTDATE, errors="raise"))
    if index.tz is None:
        index = index.tz_localize(TIMEZONE)
    else:
        index = index.tz_convert(TIMEZONE)
    values = pd.to_numeric(df.TOTALDEMAND, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Demand must be finite and positive")
    duplicates = int(index.duplicated().sum())
    if duplicates:
        raise ValueError(f"Duplicate source timestamps: {duplicates}; duplicate hourly timestamps after resampling: 0")
    series = pd.Series(values, index=index, name="demand_mw").sort_index()
    if ((series.index.minute % 5 != 0) | (series.index.second != 0)).any():
        raise ValueError("Source timestamps must align to five-minute intervals")
    hourly = series.resample("1h").mean()
    counts = series.resample("1h").count()
    missing = hourly.index[hourly.isna()]
    report = {
        "source_rows": len(series),
        "source_duplicates": duplicates,
        "hourly_duplicates": int(hourly.index.duplicated().sum()),
        "missing_hours": [t.isoformat() for t in missing],
        "partial_hours": [t.isoformat() for t in counts.index[counts != 12]],
        "timezone": "AEST (UTC+10), fixed NEM market time",
        "start": hourly.index[0].isoformat(),
        "end": hourly.index[-1].isoformat(),
    }
    if len(missing):
        raise ValueError(f"Missing hourly timestamps: {report['missing_hours']}")
    # Preserve notebook resampling, including partial boundary hours; report them explicitly.
    return hourly, report


def validate_history(history: pd.Series, lookback: int = 168) -> None:
    if len(history) < lookback:
        raise ValueError(f"At least {lookback} hours of history required")
    if not history.index.is_unique or not history.index.is_monotonic_increasing:
        raise ValueError("History must be unique and chronologically ordered")
    if not (history.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all():
        raise ValueError("History must have continuous hourly timestamps")
    if not np.isfinite(history.to_numpy()).all():
        raise ValueError("History contains nonfinite demand")


def create_sequences(data: np.ndarray, lookback: int = 168, horizon: int = 24) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(data).reshape(-1)
    if len(values) < lookback + horizon:
        raise ValueError("Insufficient data for complete sequences")
    windows = np.lib.stride_tricks.sliding_window_view(values, lookback + horizon)
    return windows[:, :lookback, None].astype(np.float32).copy(), windows[:, lookback:].astype(np.float32).copy()


def xgb_features(history: pd.Series, target_time: pd.Timestamp) -> np.ndarray:
    """All demand features are relative to the forecast origin, for every horizon."""
    v = history.to_numpy()
    if len(v) < 168:
        raise ValueError("XGBoost requires 168 observed hours")
    features = [v[-lag] for lag in (1, 2, 24, 48, 168)]
    for window in (24, 168):
        features.extend([v[-window:].mean(), v[-window:].std(ddof=1)])
    features.extend([target_time.hour, target_time.dayofweek, int(target_time.dayofweek >= 5), target_time.month])
    return np.asarray(features, dtype=np.float32)


def supervised_xgb(series: pd.Series) -> tuple[list[np.ndarray], np.ndarray]:
    origins = range(168, len(series) - 24 + 1)
    targets = np.asarray([series.iloc[i : i + 24].to_numpy() for i in origins])
    xs = [np.asarray([xgb_features(series.iloc[:i], series.index[i + h]) for i in origins]) for h in range(24)]
    if not len(targets):
        raise ValueError("Insufficient XGBoost training history")
    return xs, targets
