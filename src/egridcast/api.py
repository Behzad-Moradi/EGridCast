"""FastAPI routes; all training remains in the offline CLI."""

from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from egridcast.config import MODEL_NAMES, Settings
from egridcast.schemas import (
    CompareResponse,
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    HistoryResponse,
    MetricsResponse,
    ModelInfo,
)
from egridcast.services import ArtifactUnavailable, ForecastService


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    app = FastAPI(title="EGridCast", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.cors_origin],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    service = ForecastService(settings)
    app.state.service = service

    @app.exception_handler(ArtifactUnavailable)
    @app.exception_handler(FileNotFoundError)
    def unavailable(request: Request, exc: Exception):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(ValueError)
    def invalid_data(request: Request, exc: Exception):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.get("/api/health", response_model=HealthResponse)
    def health():
        try:
            service.data()
            data_available = True
        except (FileNotFoundError, ValueError):
            data_available = False
        available = [m.name for m in service.models() if m.available]
        return HealthResponse(
            status="ok" if data_available and len(available) == 5 else "degraded",
            data_available=data_available,
            available_models=available,
        )

    @app.get("/api/models", response_model=list[ModelInfo])
    def models():
        return service.models()

    @app.get("/api/history", response_model=HistoryResponse)
    def history(hours: int = Query(168, ge=24, le=2160)):
        series, report, _ = service.data()
        return HistoryResponse(
            points=[{"timestamp": t.to_pydatetime(), "demand_mw": float(v)} for t, v in series.iloc[-hours:].items()],
            timezone=report["timezone"],
            data_quality=report,
        )

    @app.get("/api/metrics", response_model=MetricsResponse)
    def metrics():
        result = service.metrics()
        return MetricsResponse(available=result is not None, evaluation=result)

    @app.post("/api/forecast", response_model=ForecastResponse)
    def forecast(payload: ForecastRequest):
        return service.forecast(payload.model)

    @app.post("/api/forecast/compare", response_model=CompareResponse)
    def compare():
        return CompareResponse(forecasts=[service.forecast(name) for name in MODEL_NAMES])

    return app


app = create_app()
