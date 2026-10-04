import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  use: {
    baseURL: process.env.EGRIDCAST_UI_URL ?? "http://127.0.0.1:5173",
    channel: process.env.EGRIDCAST_BROWSER_CHANNEL ?? "chrome",
    headless: true,
  },
});
