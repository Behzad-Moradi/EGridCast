import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import * as Tabs from "@radix-ui/react-tabs";
import {
  Activity,
  ArrowUpRight,
  BarChart3,
  Layers3,
  LoaderCircle,
  Zap,
} from "lucide-react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  ReferenceLine,
} from "recharts";
import {
  names,
  type ModelName,
  type ModelInfo,
  type History,
  type Forecast,
  type Evaluation,
} from "./types";
import { comparisonRows, demandRows } from "./chartData";
import "./style.css";
const colors = ["#0f766e", "#a96508", "#7953ba", "#2563b8", "#be406b"];
const base =
  (import.meta as ImportMeta & { env: Record<string, string> }).env
    .VITE_API_URL ?? "";
async function api<T>(path: string, body?: object): Promise<T> {
  const r = await fetch(
    `${base}/api/${path}`,
    body === undefined
      ? undefined
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
  );
  if (!r.ok) {
    const e = await r.json().catch(() => ({}));
    throw new Error(e.detail ?? `Request failed (${r.status})`);
  }
  return r.json();
}
const date = (s: string, full = false) =>
  new Intl.DateTimeFormat("en-AU", {
    timeZone: "Etc/GMT-10",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    ...(full ? { year: "numeric", minute: "2-digit" } : {}),
  }).format(new Date(s));
const number = (n: number | undefined, digits = 0) =>
  n === undefined
    ? "—"
    : n.toLocaleString("en-AU", { maximumFractionDigits: digits });
function App() {
  const [model, setModel] = useState<ModelName>("LSTM");
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [history, setHistory] = useState<History | null>(null);
  const [evaluation, setEvaluation] = useState<Evaluation | null>(null);
  const [forecasts, setForecasts] = useState<
    Partial<Record<ModelName, Forecast>>
  >({});
  const [compare, setCompare] = useState<Forecast[]>([]);
  const [split, setSplit] = useState<"validation" | "test">("test");
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  async function refresh() {
    setLoading(true);
    setError("");
    const results = await Promise.allSettled([
      api<ModelInfo[]>("models"),
      api<History>("history"),
      api<{ evaluation: Evaluation | null }>("metrics"),
    ]);
    if (results[0].status === "fulfilled") setModels(results[0].value);
    if (results[1].status === "fulfilled") setHistory(results[1].value);
    if (results[2].status === "fulfilled")
      setEvaluation(results[2].value.evaluation);
    const failure = results.find((r) => r.status === "rejected");
    if (failure?.status === "rejected")
      setError(String(failure.reason.message));
    setLoading(false);
  }
  useEffect(() => {
    void refresh();
  }, []);
  async function generate(all = false) {
    setBusy(all ? "compare" : "forecast");
    setError("");
    try {
      if (all) {
        const r = await api<{ forecasts: Forecast[] }>("forecast/compare", {});
        setCompare(r.forecasts);
        setForecasts(Object.fromEntries(r.forecasts.map((f) => [f.model, f])));
      } else {
        const f = await api<Forecast>("forecast", { model });
        setForecasts((old) => ({ ...old, [model]: f }));
      }
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy("");
    }
  }
  const selected = models.find((m) => m.name === model);
  const forecast = forecasts[model] ?? null;
  const score = evaluation?.models[model]?.[split];
  const rows = demandRows(history, forecast);
  return (
    <div className="app">
      <aside>
        <a className="brand" href="#">
          <span>
            <Zap size={23} />
          </span>
          EGridCast
        </a>
        <div className="nav-label">WORKSPACE</div>
        <a className="nav active" href="#overview">
          <Activity size={18} /> Forecast studio
        </a>
        <a className="nav" href="#performance">
          <BarChart3 size={18} /> Model performance
        </a>
        <a className="nav" href="#about">
          <Layers3 size={18} /> About this project
        </a>
      </aside>
      <main id="overview">
        <div className="title-row">
          <div>
            <h1>Electricity Demand Forecasting</h1>
          </div>
        </div>
        {error && (
          <div role="alert" className="notice error">
            {error}{" "}
            <button onClick={() => void refresh()}>Retry connection</button>
          </div>
        )}
        {loading ? (
          <div className="notice">
            <LoaderCircle className="spin" /> Loading demand history and model
            registry…
          </div>
        ) : (
          <>
            <section className="panel">
              <div className="panel-head">
                <div>
                  <h2>Hourly Demand Outlook</h2>
                </div>
                <div className="controls">
                  <label htmlFor="model">Model</label>
                  <select
                    id="model"
                    value={model}
                    onChange={(e) => setModel(e.target.value as ModelName)}
                  >
                    {names.map((n) => (
                      <option key={n}>{n}</option>
                    ))}
                  </select>
                  <button
                    disabled={!!busy || !selected?.available}
                    onClick={() => void generate()}
                  >
                    {busy === "forecast" ? (
                      <LoaderCircle className="spin" size={16} />
                    ) : (
                      <Zap size={16} />
                    )}{" "}
                    Generate forecast
                  </button>
                </div>
              </div>
              {!selected?.available && (
                <div className="notice">
                  {model} is awaiting a trained artifact. Run the offline
                  evaluation and production fit, then refresh the registry.{" "}
                  <button className="quiet" onClick={() => void refresh()}>
                    Refresh
                  </button>
                </div>
              )}
              <div className="chart">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={rows}>
                    <CartesianGrid stroke="#e2e8ef" vertical={false} />
                    <XAxis
                      dataKey="timestamp"
                      tickFormatter={(s) => date(String(s))}
                      minTickGap={65}
                    />
                    <YAxis
                      domain={["auto", "auto"]}
                      tickFormatter={(n) => number(Number(n))}
                      width={85}
                      label={{
                        value: "Demand (MW)",
                        angle: -90,
                        position: "insideLeft",
                        style: { textAnchor: "middle", fill: "#526773" },
                      }}
                    />
                    <Tooltip
                      labelFormatter={(s) => `${date(String(s), true)} AEST`}
                      formatter={(v, n) => [`${number(Number(v), 1)} MW`, n]}
                    />
                    <Legend itemSorter={null} />
                    <Line
                      isAnimationActive={false}
                      dataKey="history"
                      name="Observed demand"
                      stroke="#64748b"
                      dot={false}
                      strokeWidth={2}
                    />
                    {forecast && (
                      <Line
                        isAnimationActive={false}
                        dataKey="forecast"
                        name={`${model} forecast`}
                        stroke="#0f766e"
                        dot={false}
                        strokeWidth={2.5}
                      />
                    )}
                    {history?.points.at(-1) && (
                      <ReferenceLine
                        x={history.points.at(-1)!.timestamp}
                        stroke="#0f766e"
                        strokeDasharray="4 4"
                      />
                    )}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </section>
            <div className="two-col">
              <section className="panel">
                <div className="panel-head">
                  <h2>{model} Error by Forecast Horizon</h2>
                </div>
                {score ? (
                  <div className="chart small">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart
                        data={score?.mae_by_horizon.map((mae, i) => ({
                          hour: i + 1,
                          mae,
                        }))}
                      >
                        <CartesianGrid stroke="#e2e8ef" vertical={false} />
                        <XAxis
                          dataKey="hour"
                          height={50}
                          label={{
                            value: "Forecast Horizon (hours)",
                            position: "insideBottom",
                            offset: 0,
                            style: { fill: "#526773" },
                          }}
                        />
                        <YAxis
                          width={85}
                          label={{
                            value: "MAE (MW)",
                            angle: -90,
                            position: "insideLeft",
                            style: { textAnchor: "middle", fill: "#526773" },
                          }}
                        />
                        <Tooltip
                          formatter={(v) => [
                            `${number(Number(v), 2)} MW`,
                            "MAE",
                          ]}
                        />
                        <Line
                          isAnimationActive={false}
                          dataKey="mae"
                          stroke="#a96508"
                          strokeWidth={2}
                          dot={false}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <div className="empty">
                    <BarChart3 />
                    <h3>Evaluation has not been published</h3>
                    <p>
                      Forecast horizon errors appear after evaluation completes.
                    </p>
                  </div>
                )}
              </section>
              <section className="panel info">
                <div className="panel-head">
                  <div>
                    <h2>Inside {model}</h2>
                  </div>
                </div>
                <dl>
                  <dt>Architecture</dt>
                  <dd>
                    {model === "SARIMA"
                      ? "SARIMAX (2,0,1) × (1,0,1,24)"
                      : model === "XGBoost"
                        ? "24 direct horizon regressors"
                        : model === "Transformer"
                          ? "2 encoder layers · 4 attention heads"
                          : "1 recurrent layer · 64 hidden units"}
                  </dd>
                  <dt>History window</dt>
                  <dd>
                    {model === "SARIMA"
                      ? "Full observed history"
                      : "168 hourly observations"}
                  </dd>
                  <dt>Training cutoff</dt>
                  <dd>
                    {selected?.metadata
                      ? date(selected.metadata.trained_through, true) + " AEST"
                      : "Awaiting production fit"}
                  </dd>
                </dl>
                {selected?.metadata && (
                  <details>
                    <summary>Training configuration</summary>
                    <pre>
                      {JSON.stringify(selected.metadata.config, null, 2)}
                    </pre>
                  </details>
                )}
              </section>
            </div>
            <section className="panel">
              <div className="panel-head">
                <div>
                  <h2>24-Hour Model Comparison</h2>
                </div>
                <button
                  className="quiet"
                  disabled={
                    !!busy || models.filter((m) => m.available).length !== 5
                  }
                  onClick={() => void generate(true)}
                >
                  {busy === "compare" ? "Generating…" : "Compare all"}
                  <ArrowUpRight size={16} />
                </button>
              </div>
              {compare.length ? (
                <div className="chart small">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={comparisonRows(compare)}>
                      <CartesianGrid stroke="#e2e8ef" vertical={false} />
                      <XAxis
                        dataKey="timestamp"
                        tickFormatter={(s) => date(String(s))}
                        minTickGap={70}
                      />
                      <YAxis
                        domain={["auto", "auto"]}
                        width={85}
                        label={{
                          value: "Demand (MW)",
                          angle: -90,
                          position: "insideLeft",
                          style: { textAnchor: "middle", fill: "#526773" },
                        }}
                      />
                      <Tooltip
                        labelFormatter={(s) => `${date(String(s), true)} AEST`}
                      />
                      <Legend />
                      {names.map((n, i) => (
                        <Line
                          isAnimationActive={false}
                          key={n}
                          name={n}
                          dataKey={n}
                          stroke={colors[i]}
                          dot={false}
                          strokeWidth={2}
                        />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <div className="empty">
                  <Layers3 />
                  <h3>A shared view of tomorrow</h3>
                  <p>
                    When all five models are ready, generate aligned forecasts
                    here.
                  </p>
                </div>
              )}
            </section>
            <section className="panel" id="performance">
              <div className="panel-head">
                <div>
                  <h2>Model Performance Comparison</h2>
                </div>
                <Tabs.Root
                  value={split}
                  onValueChange={(v) => setSplit(v as typeof split)}
                >
                  <Tabs.List aria-label="Evaluation split" className="tabs">
                    <Tabs.Trigger value="validation">Validation</Tabs.Trigger>
                    <Tabs.Trigger value="test">Test</Tabs.Trigger>
                  </Tabs.List>
                </Tabs.Root>
              </div>
              {evaluation ? (
                <>
                  <div className="table-wrap">
                    <table>
                      <thead>
                        <tr>
                          <th>MODEL</th>
                          <th>MAE · MW</th>
                          <th>RMSE · MW</th>
                          <th>MAPE · %</th>
                          <th>ORIGINS</th>
                        </tr>
                      </thead>
                      <tbody>
                        {names.map((n, i) => {
                          const s = evaluation.models[n][split];
                          return (
                            <tr key={n}>
                              <td>
                                <span
                                  className="model-dot"
                                  style={{ background: colors[i] }}
                                />
                                {n}
                              </td>
                              <td>{number(s.mae, 2)}</td>
                              <td>{number(s.rmse, 2)}</td>
                              <td>{number(s.mape, 2)}</td>
                              <td>{s.origins}</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </>
              ) : (
                <div className="empty">
                  <BarChart3 />
                  <h3>Evaluation has not been published</h3>
                  <p>
                    Real validation and August holdout results appear after the
                    evaluation command completes.
                  </p>
                </div>
              )}
            </section>
            <section className="about" id="about">
              <div className="eyebrow">ABOUT EGRIDCAST</div>
              <p>
                EGridCast forecasts hourly electricity demand in Victoria. Five
                models—SARIMA, XGBoost, RNN, LSTM, and Transformer—are trained
                and tested using AEMO National Electricity Market data.
              </p>
            </section>
          </>
        )}
      </main>
    </div>
  );
}
createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
