import { describe, expect, it } from "vitest";
import { comparisonRows, demandRows } from "./chartData";
import type { Forecast, History } from "./types";
const forecast = {
  model: "LSTM",
  points: [
    { timestamp: "2026-09-01T01:00:00+10:00", predicted_demand_mw: 5000 },
  ],
} as Forecast;
describe("chart alignment", () => {
  it("joins history and forecast only at the observed boundary", () => {
    const history = {
      points: [{ timestamp: "2026-09-01T00:00:00+10:00", demand_mw: 4900 }],
    } as History;
    expect(demandRows(history, forecast)).toEqual([
      { timestamp: history.points[0].timestamp, history: 4900, forecast: 4900 },
      { timestamp: forecast.points[0].timestamp, forecast: 5000 },
    ]);
  });
  it("aligns comparison points by timestamp rather than array position", () => {
    const other = {
      ...forecast,
      model: "RNN",
      points: [
        { timestamp: "2026-09-01T02:00:00+10:00", predicted_demand_mw: 5100 },
      ],
    } as Forecast;
    const rows = comparisonRows([other, forecast]);
    expect(rows).toHaveLength(2);
    expect(rows[0].LSTM).toBe(5000);
    expect(rows[0].RNN).toBeUndefined();
  });
  it("returns no invented data before loading", () => {
    expect(demandRows(null, null)).toEqual([]);
    expect(comparisonRows([])).toEqual([]);
  });
});
