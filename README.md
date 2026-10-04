# EGridCast

An end-to-end Victoria electricity-demand forecasting studio: five models, a common next-24-hours protocol, FastAPI inference, and a React dashboard. Modelling is refactored from the existing notebooks, which remain intact.

## Architecture and data flow

```text
AEMO monthly CSVs → combine_files.py → canonical processed CSV
 → schema/quality validation → hourly TOTALDEMAND mean
 → chronological CV / validation → frozen configuration → August holdout
 → immutable evaluation JSON + forecast-level CSV
 → separate all-history production fit → versioned artifacts
 → FastAPI services → React / Radix UI / Recharts dashboard
```

Python modules separate data preparation, model adapters, evaluation, offline training, artifact publication, services, schemas and routes. Frontend data is always fetched from the backend. No sample forecasts or invented scores are embedded.

## Models and notebook provenance

| Model | Implementation | Forecast strategy |
|---|---|---|
| SARIMA | Notebook 01 SARIMAX (2,0,1) × (1,0,1,24) | 24-step forecast with fixed fitted coefficients; filter observed history at each origin |
| XGBoost | Notebook 02 lags 1/2/24/48/168, rolling mean/std 24/168, calendar features; depth 6, learning rate .03 | 24 horizon-specific XGBRegressors; observed lags remain fixed at origin |
| RNN | Notebook 03 nn.RNN, 64 units, one layer | 168 observations → 24 outputs |
| LSTM | Notebook 03 nn.LSTM, 64 units, one layer | 168 observations → 24 outputs |
| Transformer | Notebook 03 high-level encoder, learned position, 64 dimensions, 4 heads, 2 layers, feedforward 128 | 168 observations → 24 outputs |

Notebook one-step SARIMA/XGBoost scores are intentionally not imported into the 24-hour comparison. Neural best validation checkpoints are actually restored before scoring.

## Setup

Requires Python 3.11+, `uv`, Node 20.19+ or 22.12+, and npm. From the repository root:

```bash
uv sync --group dev
npm ci --prefix frontend
uv run python scripts/combine_files.py  # only if rebuilding processed data from raw monthly files
uv run egridcast inspect
```

Canonical CSV: `data/processed/PRICE_AND_DEMAND_2025_26_VIC1.csv`, with SETTLEMENTDATE, REGION=VIC1, TOTALDEMAND. Data is local and ignored by Git. The loader sorts chronologically, rejects duplicate source times and missing entire hours, and reports partial hours and hourly duplicates. It preserves the notebooks' hourly mean convention, including partial boundary bins.

AEMO market timestamps are interpreted as **fixed AEST (UTC+10)**, not Melbourne daylight-saving civil time. Forecast point timestamps are hourly bin labels. The current file ends at 1 September 2026 00:00 and includes only one source reading in that final hourly bin; this partial hour is explicitly reported.

## Offline evaluation and production training

These are the long-running steps to execute locally. Training is CPU-based and deterministic where practical. On macOS, the package limits OpenMP, PyTorch and XGBoost to one thread to avoid conflicting bundled OpenMP runtimes; other platforms use two training threads. Full Transformer training and repeated SARIMA fits can take substantial time.

```bash
uv run egridcast evaluate --folds 3 --epochs 30 --stride 24 --trees 800 --sarima-maxiter 100 && \
  uv run egridcast production-fit
```

The `&&` runs production training only after evaluation exits successfully. Wait for `Evaluation complete and published`; an incomplete run has no `evaluation/current.json` and cannot be used for production training. SARIMA optimization can be quiet for a substantial time between progress messages.

Training ends May 31, 2026; validation is June–July; August is the final holdout. Three configurable expanding folds run within pre-August data. Each fold fits its scaler only on its training prefix. All five models use the same complete forecast origins and 24-hour targets; default origin stride is 24 hours. Later origins may use actual observations already revealed before that origin. No actual within the forecast horizon is ever used.

Development configurations and neural epoch counts are frozen before any final test scoring. Final evaluation models refit on pre-August data for those frozen epoch counts. Outputs are `artifacts/evaluation/<run-id>/metrics.json`, `frozen_config.json`, and `predictions.csv`. A completed run is published through `evaluation/current.json`; incomplete runs are not shown in the UI. JSON includes per-fold CV, validation and test MAE/RMSE/MAPE and 24 horizon MAEs. MAPE is expressed in percent.

`production-fit` requires a complete published evaluation and an identical dataset hash. It uses the frozen settings to train on **all available history** without selecting parameters on the test set. Each model has a versioned folder under `artifacts/production/<model>/<version>/` containing preprocessing/model state and metadata: configuration, training cutoff, UTC training time, dataset hash, model checksum, Python/package versions and evaluation ID. A model pointer is published atomically after saving. Historical metrics remain separate. Models are trusted local pickle files; do not load artifacts from untrusted sources. Use the locked environment for compatibility.

For a faster pipeline exercise (real but deliberately undertrained results):

```bash
uv run egridcast evaluate --folds 1 --epochs 1 --stride 168 --trees 10 --sarima-maxiter 200
```

This still performs real fits on the repository data; it is not a substitute for the full portfolio evaluation. SARIMA nonconvergence stops the run; increase its iteration limit and rerun rather than publish an unverified fit.

## Run the application

In separate terminals:

```bash
uv run uvicorn egridcast.api:app --reload --host 127.0.0.1 --port 8000
npm run dev --prefix frontend
```

Open `http://localhost:5173`. API documentation: `http://127.0.0.1:8000/docs`. The dashboard includes historical and forecast demand, all-model comparison, split-specific metric cards/table, horizon error, artifact metadata, responsive design, and missing/error/loading states. Refresh the model registry after offline training. Restart the API after replacing the data file (history is cached per process).

Environment overrides:

| Variable | Default |
|---|---|
| EGRIDCAST_DATA | repository canonical CSV path |
| EGRIDCAST_ARTIFACTS | repository artifacts directory |
| EGRIDCAST_CORS_ORIGIN | http://localhost:5173 |
| VITE_API_URL | empty; Vite proxies /api to localhost:8000 |

For hosting, configure `VITE_API_URL` at frontend build time and the exact allowed CORS origin on the backend. Serve `frontend/dist` using your static host. Paths can be absolute to run outside the repository directory.

## API

| Endpoint | Behavior |
|---|---|
| GET /api/health | data readiness and available artifacts; degraded is an honest 200 status payload |
| GET /api/models | five-model registry and production metadata |
| GET /api/history?hours=168 | recent hourly means (24–2160 hours), fixed timezone and quality report |
| GET /api/metrics | published real evaluation; available=false before evaluation |
| POST /api/forecast | JSON `{"model":"LSTM"}`; 24 timestamped predicted_demand_mw points, origin, version and cutoff |
| POST /api/forecast/compare | all five aligned forecasts; requires every artifact |

Missing data/artifacts return 503 with actionable details. Invalid model names return 422. API requests only load artifacts and infer; they do not train. Production artifact hashes/cutoffs must match the current dataset.

## Verification

```bash
uv run pytest
uv run ruff check src scripts tests
uv run ruff format --check src scripts tests
npm run format:check --prefix frontend
npm test --prefix frontend
npm run build --prefix frontend
npm run test:e2e --prefix frontend  # with both servers running; uses local Chrome by default
```

Backend tests cover raw quality validation, hourly means, sequence alignment, origin-relative features, future perturbations, chronological folds, comparable metrics, scaler isolation, model shape, artifact roundtrip, online inference, missing models and API validation. Frontend tests cover historical/forecast boundaries, timestamp alignment and empty data.

## Screenshots

Verified screenshots of the dashboard using real repository history, before model training:

- [Desktop dashboard](docs/screenshots/dashboard-before-training.png)
- [Mobile dashboard](docs/screenshots/mobile-before-training.png)

Capture these additional views after full evaluation and production fitting:

- `docs/screenshots/forecast-studio.png` — history and selected 24-hour forecast (placeholder).
- `docs/screenshots/model-comparison.png` — five-model overlay and August test metrics (placeholder).

## Limitations and next improvements

This is a historical research application, not a live AEMO feed. Hourly means preserve partial boundary bins for notebook compatibility; a stricter coverage policy is a possible next experiment. Hyperparameters follow notebook baselines rather than exhaustive search. Daily evaluation origins reduce runtime and do not represent every possible hourly forecast origin. Production neural epoch counts are transferred from pre-test validation. Forecasts have no uncertainty bands or weather/holiday inputs; negative predictions are not silently clipped. API history stays cached until restart. Artifacts publish per model; complete comparison becomes available when all five finish. Training is intentionally offline and is not started by the UI. Add live ingestion, prediction intervals, weather covariates, monitored drift and scheduled retraining as future work.
