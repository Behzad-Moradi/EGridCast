import numpy as np
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient
from egridcast.api import create_app
from egridcast.artifacts import load_model, save_model, write_json
from egridcast.config import Settings, TrainConfig
from egridcast.models import ModelAdapter
from egridcast.models.neural import NeuralForecaster
from egridcast.schemas import ForecastResponse


@pytest.mark.parametrize("name", ["RNN", "LSTM", "Transformer"])
def test_neural_shape(name):
    net = NeuralForecaster(name, hidden_size=8)
    assert net(torch.zeros(2, 168, 1)).shape == (2, 24)


def test_scaler_train_only_and_checkpoint_roundtrip(tmp_path):
    index = pd.date_range("2026-01-01", periods=240, freq="h", tz="Etc/GMT-10")
    series = pd.Series(4000 + np.sin(np.arange(240) / 24) * 100, index=index)
    train, val = series.iloc[:210], series.iloc[210:]
    model = ModelAdapter("RNN", TrainConfig(epochs=1, hidden_size=8, batch_size=32)).fit(train, val)
    assert model.scaler.mean_[0] == pytest.approx(train.mean())
    save_model(tmp_path, model, series.index[-1].isoformat(), "digest", "evaluation")
    restored, meta = load_model(tmp_path, "RNN")
    np.testing.assert_allclose(model.forecast(series), restored.forecast(series))
    assert meta["trained_through"] == series.index[-1].isoformat()
    assert meta["packages"]["torch"]


def test_api_smoke_and_missing_artifacts(tmp_path):
    client = TestClient(create_app(Settings(data_path=tmp_path / "absent.csv", artifact_dir=tmp_path)))
    assert client.get("/api/health").json()["status"] == "degraded"
    assert len(client.get("/api/models").json()) == 5
    assert client.get("/api/metrics").json() == {"available": False, "evaluation": None}
    assert client.get("/api/history").status_code == 503
    assert client.post("/api/forecast", json={"model": "Unknown"}).status_code == 422
    assert client.get("/api/history?hours=1").status_code == 422


def test_real_artifact_forecast_api(tmp_path):
    path = tmp_path / "data.csv"
    index = pd.date_range("2026-01-01", periods=240 * 12, freq="5min")
    pd.DataFrame(
        {"SETTLEMENTDATE": index, "REGION": "VIC1", "TOTALDEMAND": 4000 + np.sin(np.arange(len(index)) / 100) * 100}
    ).to_csv(path, index=False)
    settings = Settings(data_path=path, artifact_dir=tmp_path / "artifacts")
    app = create_app(settings)
    series, _, digest = app.state.service.data()
    adapter = ModelAdapter("RNN", TrainConfig(epochs=1, hidden_size=8)).fit(series)
    save_model(settings.artifact_dir, adapter, series.index[-1].isoformat(), digest, "example")
    client = TestClient(app)
    assert client.get("/api/history").status_code == 200
    assert client.post("/api/forecast", json={"model": "LSTM"}).status_code == 503
    result = client.post("/api/forecast", json={"model": "RNN"})
    assert result.status_code == 200
    forecast = ForecastResponse(**result.json())
    assert len(forecast.points) == 24
    assert pd.Timestamp(forecast.points[0].timestamp) - pd.Timestamp(forecast.forecast_origin) == pd.Timedelta(hours=1)
    assert pd.Timestamp(forecast.points[-1].timestamp) - pd.Timestamp(forecast.forecast_origin) == pd.Timedelta(
        hours=24
    )
    assert all(p.timestamp.utcoffset().total_seconds() == 36000 for p in forecast.points)
    # Production inference must not call fit.
    adapter.fit = lambda *args, **kwargs: pytest.fail("Online retraining")
    app.state.service.adapter = lambda *args: (adapter, app.state.service.models()[2].metadata)
    assert client.post("/api/forecast", json={"model": "RNN"}).status_code == 200
    write_json(settings.artifact_dir / "production/RNN/current.json", {"version": "bad"})
    assert client.post("/api/forecast", json={"model": "RNN"}).status_code == 503


def test_schema_enforces_24_points():
    with pytest.raises(ValueError):
        ForecastResponse(
            model="RNN",
            forecast_origin="2026-01-01T00:00:00+10:00",
            points=[],
            model_version="v",
            trained_through="2026-01-01T00:00:00+10:00",
        )


def test_xgboost_direct_24_outputs_and_no_future_actuals():
    index = pd.date_range("2026-01-01", periods=240, freq="h", tz="Etc/GMT-10")
    series = pd.Series(4000 + np.sin(np.arange(240) / 24) * 100, index=index)
    model = ModelAdapter("XGBoost", TrainConfig(trees=2)).fit(series.iloc[:220])
    assert len(model.model) == 24
    prediction = model.forecast(series.iloc[:220])
    assert prediction.shape == (24,) and np.isfinite(prediction).all()
    with pytest.raises(ValueError, match="24-hour"):
        model.forecast(series, horizon=1)


def test_sarima_inference_filters_fixed_coefficients(monkeypatch):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    index = pd.date_range("2026-01-01", periods=240, freq="h", tz="Etc/GMT-10")
    series = pd.Series(4000 + np.sin(np.arange(240) / 24) * 100, index=index)
    adapter = ModelAdapter("SARIMA", TrainConfig())
    # Known-coefficient fixture exercises state filtering; no optimization or training claim.
    state_model = SARIMAX(series.iloc[:200], order=(1, 0, 0), seasonal_order=(0, 0, 0, 24))
    adapter.model = state_model.filter([0.99, 100.0])
    original_params = adapter.model.params.copy()
    expected = adapter.model.apply(series, refit=False).forecast(24).to_numpy()
    # Even older artifacts with smoother results must use filtering at inference.
    adapter.model = state_model.smooth(original_params)
    monkeypatch.setattr(SARIMAX, "smooth", lambda *a, **kw: pytest.fail("Unnecessary SARIMA smoothing"))
    predicted = adapter.forecast(series)
    assert predicted.shape == (24,) and np.isfinite(predicted).all()
    np.testing.assert_allclose(predicted, expected)
    np.testing.assert_array_equal(adapter.model.params, original_params)


def test_checksum_rejects_corrupted_artifact(tmp_path):
    index = pd.date_range("2026-01-01", periods=192, freq="h", tz="Etc/GMT-10")
    series = pd.Series(np.arange(192) + 4000.0, index=index)
    model = ModelAdapter("RNN", TrainConfig(epochs=1, hidden_size=8)).fit(series)
    meta = save_model(tmp_path, model, series.index[-1].isoformat(), "digest", "run")
    payload = tmp_path / "production/RNN" / meta["model_version"] / "model.pkl"
    payload.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_model(tmp_path, "RNN")
