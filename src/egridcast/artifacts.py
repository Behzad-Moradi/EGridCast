"""Versioned, local trusted artifacts and atomic publication."""

import hashlib
import importlib.metadata
import json
import os
import pickle
import platform
import uuid
from datetime import datetime, timezone
from pathlib import Path
from egridcast.models import ModelAdapter

PACKAGES = ("numpy", "pandas", "scikit-learn", "statsmodels", "xgboost", "torch")


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(temp, path)


def save_model(root: Path, adapter: ModelAdapter, trained_through: str, dataset_hash: str, evaluation_id: str) -> dict:
    version = uuid.uuid4().hex[:12]
    folder = root / "production" / adapter.name / version
    folder.mkdir(parents=True)
    payload = pickle.dumps(adapter, protocol=5)
    (folder / "model.pkl").write_bytes(payload)
    meta = {
        "name": adapter.name,
        "model_version": version,
        "trained_through": trained_through,
        "last_trained": datetime.now(timezone.utc).isoformat(),
        "config": adapter.config.to_dict(),
        "lookback": 168,
        "horizon_hours": 24,
        "evaluation_id": evaluation_id,
        "dataset_sha256": dataset_hash,
        "artifact_sha256": hashlib.sha256(payload).hexdigest(),
        "python_version": platform.python_version(),
        "packages": {p: importlib.metadata.version(p) for p in PACKAGES},
    }
    write_json(folder / "metadata.json", meta)
    write_json(root / "production" / adapter.name / "current.json", {"version": version})
    return meta


def model_metadata(root: Path, name: str) -> tuple[dict, Path]:
    base = root / "production" / name
    version = json.loads((base / "current.json").read_text())["version"]
    if not isinstance(version, str) or not version.isalnum():
        raise ValueError("Invalid artifact version")
    folder = base / version
    return json.loads((folder / "metadata.json").read_text()), folder


def load_model(root: Path, name: str) -> tuple[ModelAdapter, dict]:
    meta, folder = model_metadata(root, name)
    payload = (folder / "model.pkl").read_bytes()
    if hashlib.sha256(payload).hexdigest() != meta["artifact_sha256"]:
        raise ValueError("Artifact checksum mismatch")
    # Only load artifacts generated locally by the training CLI, never user uploads.
    return pickle.loads(payload), meta
