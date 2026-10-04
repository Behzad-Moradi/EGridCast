import type { Forecast, History } from "./types";
export function demandRows(history: History | null, forecast: Forecast | null) {
  const rows: { timestamp: string; history?: number; forecast?: number }[] = (
    history?.points ?? []
  ).map((p) => ({ timestamp: p.timestamp, history: p.demand_mw }));
  if (forecast && rows.length)
    rows[rows.length - 1].forecast = rows[rows.length - 1].history;
  return [
    ...rows,
    ...(forecast?.points ?? []).map((p) => ({
      timestamp: p.timestamp,
      forecast: p.predicted_demand_mw,
    })),
  ];
}
export function comparisonRows(forecasts: Forecast[]) {
  const rows = new Map<string, Record<string, string | number>>();
  for (const f of forecasts)
    for (const p of f.points) {
      const row = rows.get(p.timestamp) ?? { timestamp: p.timestamp };
      row[f.model] = p.predicted_demand_mw;
      rows.set(p.timestamp, row);
    }
  return [...rows.values()].sort((a, b) =>
    String(a.timestamp).localeCompare(String(b.timestamp)),
  );
}
