"""Offline CLI: evaluation freezes configuration before untouched August testing."""

import argparse
import hashlib
import json
import uuid
from dataclasses import replace
from pathlib import Path
import pandas as pd
from egridcast.artifacts import save_model, write_json
from egridcast.config import MODEL_NAMES, Settings, TrainConfig
from egridcast.data import load_hourly
from egridcast.evaluation import evaluate, expanding_folds, forecast_origins
from egridcast.models import ModelAdapter


def run_evaluation(settings: Settings, config: TrainConfig) -> Path:
    series, report = load_hourly(settings.data_path)
    dataset_hash = hashlib.sha256(settings.data_path.read_bytes()).hexdigest()
    run_id = uuid.uuid4().hex[:12]
    output = settings.artifact_dir / "evaluation" / run_id
    output.mkdir(parents=True)
    print(f"Evaluation run: {run_id} → {output}", flush=True)
    train = series.loc[:"2026-05-31 23:00"]
    val = series.loc["2026-06-01":"2026-07-31 23:00"]
    pretest = series.loc[:"2026-07-31 23:00"]
    if len(train) < 192 or len(val) < 24:
        raise ValueError("Data does not cover notebook training/validation dates")
    validation_origins = forecast_origins(series, val.index[0], val.index[-1], config.stride)
    test_origins = forecast_origins(
        series,
        pd.Timestamp("2026-08-01", tz=series.index.tz),
        pd.Timestamp("2026-08-31 23:00", tz=series.index.tz),
        config.stride,
    )
    results = {
        "evaluation_id": run_id,
        "dataset_sha256": dataset_hash,
        "data_quality": report,
        "protocol": {
            "train_end": train.index[-1].isoformat(),
            "validation_end": val.index[-1].isoformat(),
            "test": "August 2026",
            "stride_hours": config.stride,
            "horizon_hours": 24,
            "folds": config.folds,
        },
        "models": {},
    }
    selected, rows = {}, []
    # Complete all development work and freeze choices BEFORE evaluating any test prediction.
    for name in MODEL_NAMES:
        print(f"Development: {name}", flush=True)
        cv = []
        for n, (fold_train, fold_val) in enumerate(expanding_folds(pretest, config.folds)):
            print(
                f"{name}: CV fold {n + 1}/{config.folds}, fitting {len(fold_train):,} hours "
                f"through {fold_train.index[-1].isoformat()}",
                flush=True,
            )
            model = ModelAdapter(name, config).fit(fold_train, fold_val)
            origins = forecast_origins(pretest, fold_val.index[0], fold_val.index[-1], config.stride)
            print(f"{name}: scoring fold {n + 1} at {len(origins)} forecast origins", flush=True)
            metrics, records = evaluate(model, pretest, origins, f"cv_{n + 1}")
            cv.append({"fold": n + 1, "train_end": fold_train.index[-1].isoformat(), **metrics})
            rows.extend({"model": name, **r} for r in records)
        print(f"{name}: fitting development model, then scoring June–July validation", flush=True)
        model = ModelAdapter(name, config).fit(train, val)
        metrics, records = evaluate(model, series, validation_origins, "validation")
        rows.extend({"model": name, **r} for r in records)
        frozen = replace(config, epochs=model.best_epoch) if name in ("RNN", "LSTM", "Transformer") else config
        selected[name] = frozen.to_dict()
        results["models"][name] = {
            "cross_validation": cv,
            "validation": metrics,
            "selected_config": frozen.to_dict(),
            "training_log": model.training_log,
        }
    write_json(output / "frozen_config.json", {"models": selected, "dataset_sha256": dataset_hash})
    for name in MODEL_NAMES:
        print(f"Untouched test: {name}", flush=True)
        model = ModelAdapter(name, TrainConfig(**selected[name])).fit(pretest)
        metrics, records = evaluate(model, series, test_origins, "test")
        rows.extend({"model": name, **r} for r in records)
        results["models"][name]["test"] = metrics
    pd.DataFrame(rows).to_csv(output / "predictions.csv", index=False)
    write_json(output / "metrics.json", results)
    write_json(settings.artifact_dir / "evaluation" / "current.json", {"evaluation_id": run_id})
    print("Evaluation complete and published. production-fit can now run.", flush=True)
    return output


def production_fit(settings: Settings) -> None:
    pointer = settings.artifact_dir / "evaluation/current.json"
    if not pointer.is_file():
        raise RuntimeError(
            "No completed evaluation is published. Run 'uv run egridcast evaluate --folds 3 --epochs 30' "
            "and wait for 'Evaluation complete and published' before running production-fit. "
            "If evaluation exited early, inspect its output; production-fit cannot use an incomplete run."
        )
    run_id = json.loads(pointer.read_text())["evaluation_id"]
    folder = settings.artifact_dir / "evaluation" / run_id
    metrics = json.loads((folder / "metrics.json").read_text())
    if set(metrics["models"]) != set(MODEL_NAMES):
        raise ValueError("Complete five-model evaluation is required")
    frozen = json.loads((folder / "frozen_config.json").read_text())
    series, _ = load_hourly(settings.data_path)
    digest = hashlib.sha256(settings.data_path.read_bytes()).hexdigest()
    if digest != frozen["dataset_sha256"]:
        raise ValueError("Dataset changed since evaluation; rerun evaluate before production fit")
    for name in MODEL_NAMES:
        print(f"Production fit: {name}", flush=True)
        adapter = ModelAdapter(name, TrainConfig(**frozen["models"][name])).fit(series)
        save_model(settings.artifact_dir, adapter, series.index[-1].isoformat(), digest, run_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["inspect", "evaluate", "production-fit"])
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--stride", type=int, default=24)
    parser.add_argument("--trees", type=int, default=800)
    parser.add_argument("--sarima-maxiter", type=int, default=100)
    args = parser.parse_args()
    settings = Settings()
    config = TrainConfig(
        folds=args.folds, epochs=args.epochs, stride=args.stride, trees=args.trees, sarima_maxiter=args.sarima_maxiter
    )
    if min(config.folds, config.epochs, config.stride, config.trees, config.sarima_maxiter) < 1:
        parser.error("All numeric options must be positive")
    try:
        if args.command == "inspect":
            print(json.dumps(load_hourly(settings.data_path)[1], indent=2))
        elif args.command == "evaluate":
            print(run_evaluation(settings, config))
        else:
            production_fit(settings)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"egridcast: {exc}\n")


if __name__ == "__main__":
    main()
