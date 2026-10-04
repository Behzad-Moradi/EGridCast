"""Protect orchestration against final-test-driven model selection."""

import json
import numpy as np
import pandas as pd
from egridcast.config import MODEL_NAMES, Settings, TrainConfig
from egridcast.evaluation import score_predictions
from egridcast.training import run_evaluation, production_fit


def test_all_development_is_frozen_before_any_test(monkeypatch, tmp_path):
    import egridcast.training as training

    index = pd.date_range("2025-01-01", "2026-09-01", freq="h", tz="Etc/GMT-10")
    series = pd.Series(np.ones(len(index)) * 4000, index=index)
    path = tmp_path / "source.csv"
    path.write_text("dataset fixture")
    settings = Settings(data_path=path, artifact_dir=tmp_path / "artifacts")
    events = []
    scored_origins = {"validation": [], "test": []}
    monkeypatch.setattr(training, "load_hourly", lambda p: (series, {}))

    class RecordingAdapter:
        def __init__(self, name, config):
            self.name, self.config = name, config
            self.best_epoch = 2
            self.training_log = []

        def fit(self, train, validation=None):
            assert train.index[-1] < pd.Timestamp("2026-08-01", tz=index.tz)
            if validation is not None:
                assert train.index[-1] < validation.index[0]
                assert validation.index[-1] < pd.Timestamp("2026-08-01", tz=index.tz)
            events.append(("fit", self.name))
            return self

    def recording_evaluate(model, history, origins, split):
        events.append((split, model.name))
        if split in scored_origins:
            scored_origins[split].append(origins)
        if split == "test":
            frozen = list(settings.artifact_dir.glob("evaluation/*/frozen_config.json"))
            assert len(frozen) == 1
            selected = json.loads(frozen[0].read_text())["models"]
            assert set(selected) == set(MODEL_NAMES)
            assert selected["LSTM"]["epochs"] == 2
        metrics = score_predictions(np.ones((len(origins), 24)) * 4000, np.ones((len(origins), 24)) * 4000)
        return metrics, []

    monkeypatch.setattr(training, "ModelAdapter", RecordingAdapter)
    monkeypatch.setattr(training, "evaluate", recording_evaluate)
    output = run_evaluation(settings, TrainConfig(folds=1, epochs=3, stride=168))
    first_test = next(i for i, e in enumerate(events) if e[0] == "test")
    assert all(event[0] not in ("validation", "cv_1") for event in events[first_test:])
    for origins in scored_origins.values():
        assert len(origins) == 5
        assert all(o == origins[0] for o in origins)
    assert json.loads((output / "metrics.json").read_text())["models"]["LSTM"]["selected_config"]["epochs"] == 2
    # Dataset changes invalidate a production fit before any model is trained.
    path.write_text("changed data")
    import pytest

    with pytest.raises(ValueError, match="Dataset changed"):
        production_fit(settings)


def test_production_requires_published_evaluation(tmp_path):
    import pytest

    settings = Settings(data_path=tmp_path / "data.csv", artifact_dir=tmp_path / "artifacts")
    # An incomplete run directory must not authorize production training.
    (settings.artifact_dir / "evaluation" / "incomplete").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="No completed evaluation is published"):
        production_fit(settings)


def test_cli_missing_evaluation_is_actionable_without_traceback(monkeypatch, tmp_path, capsys):
    import pytest
    import egridcast.training as training

    monkeypatch.setattr(training, "Settings", lambda: Settings(artifact_dir=tmp_path))
    monkeypatch.setattr("sys.argv", ["egridcast", "production-fit"])
    with pytest.raises(SystemExit) as error:
        training.main()
    assert error.value.code == 1
    message = capsys.readouterr().err
    assert "No completed evaluation is published" in message
    assert "Traceback" not in message
