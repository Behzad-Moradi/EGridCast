export const names = [
  "SARIMA",
  "XGBoost",
  "RNN",
  "LSTM",
  "Transformer",
] as const;
export type ModelName = (typeof names)[number];
export type Forecast = {
  model: ModelName;
  forecast_origin: string;
  points: { timestamp: string; predicted_demand_mw: number }[];
  model_version: string;
  trained_through: string;
};
export type ModelInfo = {
  name: ModelName;
  available: boolean;
  metadata: {
    config: Record<string, unknown>;
    lookback: number;
    trained_through: string;
    model_version: string;
    last_trained: string;
  } | null;
  error: string | null;
};
export type Score = {
  mae: number;
  rmse: number;
  mape: number;
  mae_by_horizon: number[];
  origins: number;
};
export type Evaluation = {
  evaluation_id: string;
  models: Record<ModelName, { validation: Score; test: Score }>;
};
export type History = {
  points: { timestamp: string; demand_mw: number }[];
  timezone: string;
  data_quality: { partial_hours: string[]; missing_hours: string[] };
};
