"""Data/artifact loading and inference outside route handlers."""

import hashlib
import json
from functools import lru_cache
import pandas as pd
from egridcast.artifacts import load_model, model_metadata
from egridcast.config import MODEL_NAMES, Settings
from egridcast.data import load_hourly
from egridcast.schemas import ForecastResponse, ModelInfo


class ArtifactUnavailable(RuntimeError):
    pass


class ForecastService:
    def __init__(self, settings: Settings):
        self.settings = settings

    @lru_cache(maxsize=1)
    def data(self) -> tuple[pd.Series, dict, str]:
        series, report = load_hourly(self.settings.data_path)
        return series, report, hashlib.sha256(self.settings.data_path.read_bytes()).hexdigest()

    def models(self) -> list[ModelInfo]:
        items = []
        for name in MODEL_NAMES:
            try:
                metadata, folder = model_metadata(self.settings.artifact_dir, name)
                available = (folder / "model.pkl").is_file()
                items.append(
                    ModelInfo(
                        name=name,
                        available=available,
                        metadata=metadata,
                        error=None if available else "Artifact payload missing",
                    )
                )
            except (FileNotFoundError, ValueError, KeyError):
                items.append(
                    ModelInfo(
                        name=name, available=False, error="Run evaluate, then production-fit to create this artifact"
                    )
                )
        return items

    @lru_cache(maxsize=10)
    def adapter(self, name: str, version: str):
        adapter, metadata = load_model(self.settings.artifact_dir, name)
        if metadata["model_version"] != version:
            raise ArtifactUnavailable("Artifact changed during request; retry")
        return adapter, metadata

    def forecast(self, name: str) -> ForecastResponse:
        series, _, digest = self.data()
        try:
            meta, _ = model_metadata(self.settings.artifact_dir, name)
            adapter, meta = self.adapter(name, meta["model_version"])
        except (FileNotFoundError, ValueError, KeyError) as exc:
            raise ArtifactUnavailable(f"{name} artifact missing or invalid. Run production-fit.") from exc
        if meta["dataset_sha256"] != digest or pd.Timestamp(meta["trained_through"]) != series.index[-1]:
            raise ArtifactUnavailable(
                "Production artifact does not match the current dataset; rerun evaluation and production-fit"
            )
        values = adapter.forecast(series)
        timestamps = pd.date_range(series.index[-1] + pd.Timedelta(hours=1), periods=24, freq="h")
        return ForecastResponse(
            model=name,
            forecast_origin=series.index[-1].to_pydatetime(),
            points=[
                {"timestamp": t.to_pydatetime(), "predicted_demand_mw": float(v)}
                for t, v in zip(timestamps, values, strict=True)
            ],
            model_version=meta["model_version"],
            trained_through=meta["trained_through"],
        )

    def metrics(self) -> dict | None:
        pointer = self.settings.artifact_dir / "evaluation/current.json"
        if not pointer.exists():
            return None
        run = json.loads(pointer.read_text())["evaluation_id"]
        if not isinstance(run, str) or not run.isalnum():
            raise ValueError("Invalid evaluation identifier")
        return json.loads((self.settings.artifact_dir / "evaluation" / run / "metrics.json").read_text())
