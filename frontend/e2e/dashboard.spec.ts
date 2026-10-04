import { test, expect } from "@playwright/test";

test("loads real history and honest unavailable-artifact states", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "Demand, ahead of time." }),
  ).toBeVisible();
  await expect(
    page.getByText("Loading demand history and model registry…"),
  ).toBeHidden();
  await expect(
    page.getByRole("heading", { name: "Hourly demand outlook" }),
  ).toBeVisible();
  await expect(page.locator(".recharts-line").first()).toBeVisible();
  const registry = await (await page.request.get("/api/models")).json();
  if (registry.every((m: { available: boolean }) => !m.available)) {
    await expect(
      page.getByRole("button", { name: "Generate forecast" }),
    ).toBeDisabled();
    await expect(
      page.getByText("LSTM is awaiting a trained artifact.", { exact: false }),
    ).toBeVisible();
    await page.getByLabel("Model", { exact: true }).selectOption("Transformer");
    await expect(
      page.getByRole("heading", { name: "Inside Transformer" }),
    ).toBeVisible();
    await expect(
      page.getByText("Transformer is awaiting a trained artifact.", {
        exact: false,
      }),
    ).toBeVisible();
  }
  await page.getByRole("tab", { name: "Validation" }).click();
  await expect(page.getByRole("tab", { name: "Validation" })).toHaveAttribute(
    "data-state",
    "active",
  );
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({
    path: "/private/tmp/egridcast-desktop.png",
    fullPage: true,
  });
  expect(errors).toEqual([]);
});

test("mobile layout fits the viewport", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "Hourly demand outlook" }),
  ).toBeVisible();
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth),
  ).toBeLessThanOrEqual(390);
  await page.screenshot({
    path: "/private/tmp/egridcast-mobile.png",
    fullPage: true,
  });
});
