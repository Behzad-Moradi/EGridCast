import numpy as np
import pandas as pd
import pytest
from egridcast.data import create_sequences, load_hourly, supervised_xgb, xgb_features, validate_history
from egridcast.evaluation import expanding_folds, forecast_origins, score_predictions


@pytest.fixture
def series():
    return pd.Series(
        np.arange(400, dtype=float) + 4000, index=pd.date_range("2026-01-01", periods=400, freq="h", tz="Etc/GMT-10")
    )


def test_sequences_align_and_include_last_complete_window():
    x, y = create_sequences(np.arange(200))
    assert x.shape == (9, 168, 1) and y.shape == (9, 24)
    np.testing.assert_array_equal(x[0, :, 0], np.arange(168))
    np.testing.assert_array_equal(y[-1], np.arange(176, 200))


def test_direct_features_never_read_future(series):
    history = series.iloc[:200]
    features = xgb_features(history, series.index[223])
    assert features[0] == series.iloc[199]
    assert features[4] == series.iloc[32]
    assert features[5] == series.iloc[176:200].mean()
    xs, ys = supervised_xgb(series.iloc[:220])
    assert xs[0][0, 0] == series.iloc[167]
    assert xs[23][0, 0] == xs[0][0, 0]
    assert ys[0, 0] == series.iloc[168] and ys[0, 23] == series.iloc[191]


def test_future_perturbations_leave_origin_features_unchanged(series):
    changed = series.copy()
    changed.iloc[200:] = -999
    for h in range(24):
        np.testing.assert_array_equal(
            xgb_features(series.iloc[:200], series.index[200 + h]),
            xgb_features(changed.iloc[:200], changed.index[200 + h]),
        )


def test_folds_and_complete_targets(series):
    folds = expanding_folds(series, 3)
    for train, val in folds:
        assert train.index[-1] < val.index[0]
    origins = forecast_origins(series, series.index[250], series.index[300])
    assert all(i >= 250 and i + 23 <= 300 for i in origins)
    with pytest.raises(ValueError):
        forecast_origins(series, series.index[0], series.index[-1], 0)


def test_metrics_same_horizon():
    metrics = score_predictions(np.ones((2, 24)) * 100, np.ones((2, 24)) * 110)
    assert metrics["mae"] == 10 and metrics["rmse"] == 10
    assert metrics["mape"] == pytest.approx(10)
    assert len(metrics["mae_by_horizon"]) == 24
    with pytest.raises(ValueError):
        score_predictions(np.ones((2, 1)), np.ones((2, 1)))


def test_source_validation(tmp_path):
    times = pd.date_range("2026-01-01", periods=24, freq="5min")
    df = pd.DataFrame({"SETTLEMENTDATE": times, "REGION": "VIC1", "TOTALDEMAND": np.arange(24) + 4000})
    p = tmp_path / "data.csv"
    df.to_csv(p, index=False)
    hourly, report = load_hourly(p)
    assert len(hourly) == 2 and hourly.iloc[0] == 4005.5
    assert report["missing_hours"] == [] and report["partial_hours"] == []
    pd.concat([df, df.iloc[:1]]).to_csv(p, index=False)
    with pytest.raises(ValueError, match="Duplicate"):
        load_hourly(p)
    df.iloc[[0, 23]].to_csv(p, index=False)
    _, report = load_hourly(p)
    assert len(report["partial_hours"]) == 2
    df.iloc[[0]].assign(TOTALDEMAND=float("nan")).to_csv(p, index=False)
    with pytest.raises(ValueError, match="finite"):
        load_hourly(p)


def test_missing_hour_is_rejected(tmp_path):
    df = pd.DataFrame(
        {"SETTLEMENTDATE": ["2026-01-01 00:00", "2026-01-01 02:00"], "REGION": "VIC1", "TOTALDEMAND": 4000}
    )
    p = tmp_path / "data.csv"
    df.to_csv(p, index=False)
    with pytest.raises(ValueError, match="Missing hourly"):
        load_hourly(p)


def test_history_rejects_gap(series):
    with pytest.raises(ValueError, match="continuous"):
        validate_history(series.drop(series.index[200]))
