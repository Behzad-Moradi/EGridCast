"""Common fit/forecast interface; online forecasts never optimize parameters."""

import copy
import random
import sys
from typing import Protocol
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor
from egridcast.config import TrainConfig
from egridcast.data import create_sequences, supervised_xgb, validate_history, xgb_features
from egridcast.models.neural import NeuralForecaster


class Forecaster(Protocol):
    def forecast(self, history: pd.Series, horizon: int = 24) -> np.ndarray: ...


class ModelAdapter:
    def __init__(self, name: str, config: TrainConfig):
        self.name, self.config = name, config
        self.best_epoch = config.epochs
        self.training_log: list[dict] = []

    def fit(self, train: pd.Series, validation: pd.Series | None = None) -> "ModelAdapter":
        c = self.config
        random.seed(c.seed)
        np.random.seed(c.seed)
        torch.manual_seed(c.seed)
        threads = 1 if sys.platform == "darwin" else 2
        torch.set_num_threads(threads)
        if self.name == "SARIMA":
            self.model = SARIMAX(train, order=(2, 0, 1), seasonal_order=(1, 0, 1, 24)).fit(
                disp=False, maxiter=c.sarima_maxiter, low_memory=True
            )
            if not self.model.mle_retvals.get("converged", False):
                raise RuntimeError("SARIMA did not converge; increase --sarima-maxiter")
        elif self.name == "XGBoost":
            xs, ys = supervised_xgb(train)
            self.model = []
            for h in range(24):
                reg = XGBRegressor(
                    n_estimators=c.trees,
                    learning_rate=0.03,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="reg:squarederror",
                    random_state=c.seed,
                    n_jobs=threads,
                )
                reg.fit(xs[h], ys[:, h])
                self.model.append(reg)
        else:
            self.scaler = StandardScaler().fit(train.to_numpy().reshape(-1, 1))
            x, y = create_sequences(self.scaler.transform(train.to_numpy().reshape(-1, 1)))
            loader = DataLoader(
                TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=c.batch_size, shuffle=False
            )
            self.model = NeuralForecaster(self.name, c.hidden_size, c.lookback)
            opt = torch.optim.Adam(self.model.parameters(), lr=c.learning_rate)
            criterion = torch.nn.MSELoss()
            if validation is not None:
                context = pd.concat([train.iloc[-168:], validation])
                vx, vy = create_sequences(self.scaler.transform(context.to_numpy().reshape(-1, 1)))
                val_loader = DataLoader(
                    TensorDataset(torch.from_numpy(vx), torch.from_numpy(vy)), batch_size=c.batch_size
                )
            best, stale, state = float("inf"), 0, None
            for epoch in range(c.epochs):
                self.model.train()
                losses = []
                for bx, by in loader:
                    opt.zero_grad()
                    loss = criterion(self.model(bx), by)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    opt.step()
                    losses.append(loss.item())
                self.model.eval()
                score = None
                if validation is not None:
                    total, count = 0.0, 0
                    with torch.no_grad():
                        for bx, by in val_loader:
                            total += criterion(self.model(bx), by).item() * len(bx)
                            count += len(bx)
                    score = total / count
                    if score < best:
                        best, stale = score, 0
                        state = copy.deepcopy(self.model.state_dict())
                        self.best_epoch = epoch + 1
                    else:
                        stale += 1
                self.training_log.append(
                    {"epoch": epoch + 1, "train_mse_scaled": float(np.mean(losses)), "validation_mse_scaled": score}
                )
                print(f"{self.name}: epoch {epoch + 1}/{c.epochs}, validation={score}", flush=True)
                if validation is not None and stale >= c.patience:
                    break
            if state is not None:
                self.model.load_state_dict(state)
            self.model.eval()
        return self

    def forecast(self, history: pd.Series, horizon: int = 24) -> np.ndarray:
        if horizon != 24:
            raise ValueError("Only a 24-hour horizon is supported")
        validate_history(history)
        if self.name == "SARIMA":
            # apply() can invoke the native smoother, which is unnecessary for
            # forecasting and can crash on macOS. Filter with fixed coefficients
            # and retain only the state needed for out-of-sample prediction.
            filtered = self.model.model.clone(history).filter(self.model.params, low_memory=True, cov_type="none")
            pred = filtered.forecast(24).to_numpy()
        elif self.name == "XGBoost":
            pred = np.asarray(
                [
                    reg.predict(xgb_features(history, history.index[-1] + pd.Timedelta(hours=h + 1))[None])[0]
                    for h, reg in enumerate(self.model)
                ]
            )
        else:
            x = self.scaler.transform(history.iloc[-168:].to_numpy().reshape(-1, 1)).astype(np.float32)
            self.model.eval()
            with torch.no_grad():
                scaled = self.model(torch.from_numpy(x[None])).numpy().reshape(-1, 1)
            pred = self.scaler.inverse_transform(scaled).reshape(-1)
        if np.asarray(pred).shape != (24,) or not np.isfinite(pred).all():
            raise ValueError("Model produced invalid forecasts")
        return pred
