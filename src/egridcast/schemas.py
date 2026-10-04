from datetime import datetime
from typing import Annotated, Literal
from pydantic import BaseModel, Field

ModelName = Literal["SARIMA", "XGBoost", "RNN", "LSTM", "Transformer"]


class ForecastRequest(BaseModel):
    model: ModelName = "LSTM"


class ForecastPoint(BaseModel):
    timestamp: datetime
    predicted_demand_mw: float


class HistoryPoint(BaseModel):
    timestamp: datetime
    demand_mw: float


class ForecastResponse(BaseModel):
    model: ModelName
    forecast_origin: datetime
    horizon_hours: Literal[24] = 24
    points: Annotated[list[ForecastPoint], Field(min_length=24, max_length=24)]
    model_version: str
    trained_through: datetime


class CompareResponse(BaseModel):
    forecasts: list[ForecastResponse]


class HistoryResponse(BaseModel):
    points: list[HistoryPoint]
    timezone: str
    data_quality: dict


class ModelInfo(BaseModel):
    name: ModelName
    available: bool
    metadata: dict | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: str
    data_available: bool
    available_models: list[str]


class MetricsResponse(BaseModel):
    available: bool
    evaluation: dict | None = None
