import { expect, test } from "@playwright/test";

test("builder opens and JSON preview reports invalid edits", async ({ page }) => {
  await page.goto("/#/builder");

  await expect(page.getByRole("heading", { name: "Block Library" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Artifacts" })).toBeVisible();

  await page.getByRole("button", { name: "View" }).first().click();
  await expect(page.getByRole("heading", { name: "Graph JSON" })).toBeVisible();

  const textarea = page.locator("textarea.json-editor");
  await textarea.fill("{");
  await expect(page.getByText(/JSON parse error:/)).toBeVisible();
  await expect(page.getByRole("button", { name: "Apply to Graph" })).toBeDisabled();

  await page.getByRole("button", { name: "Close" }).click();
  await expect(page.getByRole("heading", { name: "Graph JSON" })).toHaveCount(0);
});

test("template import enables PyTorch preview for an exact family graph", async ({ page }) => {
  await page.goto("/#/builder");

  await page.getByLabel("Architecture Template").fill("GPT-2");
  await page.getByLabel("Template Blocks").fill("4");
  await page.getByRole("button", { name: "Load Template" }).click();

  const viewButtons = page.getByRole("button", { name: "View" });
  await viewButtons.nth(1).click();

  await expect(page.getByRole("heading", { name: "PyTorch Model" })).toBeVisible();
  await expect(page.locator(".cm-content")).toContainText("class GPT2Config");
  await page.getByRole("button", { name: "Close" }).click();
});

test("LLaMA template import enables exact-family PyTorch preview", async ({ page }) => {
  await page.goto("/#/builder");

  await page.getByLabel("Architecture Template").fill("LLaMA");
  await page.getByLabel("Template Blocks").fill("2");
  await page.getByRole("button", { name: "Load Template" }).click();

  const viewButtons = page.getByRole("button", { name: "View" });
  await viewButtons.nth(1).click();

  await expect(page.getByRole("heading", { name: "PyTorch Model" })).toBeVisible();
  await expect(page.locator(".cm-content")).toContainText("class LlamaConfig");
  await page.getByRole("button", { name: "Close" }).click();
});

test("dangling transformer blocks block export downloads", async ({ page }) => {
  await page.goto("/#/builder");

  await page.getByLabel("Find blocks or presets").fill("Transformer Block");
  await page.locator(".block-grid .block-card").filter({ hasText: "Transformer Block" }).first().click();

  await expect(page.getByText("Unconnected Output")).toBeVisible();
  await expect(page.getByText(/TransformerBlock has no outgoing connection\./)).toBeVisible();
  await expect(page.getByText(/Unavailable:/).first()).toBeVisible();
  await expect(page.getByRole("button", { name: "Download Project" })).toBeDisabled();
});

test("pruning page can inspect a mocked Hugging Face model and return to builder", async ({ page }) => {
  await page.route("http://127.0.0.1:8787/health", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ ok: true })
    });
  });

  await page.route("http://127.0.0.1:8787/api/hf/inspect", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        ok: true,
        result: {
          modelId: "microsoft/Phi-3-mini-4k-instruct",
          resolvedFamily: "phi3",
          config: {
            model_type: "phi3",
            num_hidden_layers: 32,
            hidden_size: 3072
          },
          weightIndex: {
            weight_map: {
              "model.layers.0.input_layernorm.weight": "model-00001-of-00002.safetensors"
            }
          },
          inspection: {
            layerCountHint: 32,
            layerCountKey: "num_hidden_layers",
            detectedLayerPrefix: "model.layers.",
            detectedLayerIndices: Array.from({ length: 32 }, (_, index) => index),
            broadPruningSupported: true,
            sampleLayerKeys: [
              "model.layers.0.input_layernorm.weight",
              "model.layers.0.mlp.down_proj.weight"
            ]
          }
        }
      })
    });
  });

  await page.goto("/#/builder");
  await page.getByRole("button", { name: "Try Pruning Tool" }).click();

  await expect(page.getByRole("heading", { name: "Pruning Tool" })).toBeVisible();
  await page.getByLabel("Hugging Face Model ID").fill("microsoft/Phi-3-mini-4k-instruct");
  await page.getByRole("button", { name: "Fetch Model Metadata" }).click();

  await expect(page.getByText("Resolved Model")).toBeVisible();
  await expect(page.getByText("microsoft/Phi-3-mini-4k-instruct")).toBeVisible();
  await expect(page.getByText("phi3", { exact: true })).toBeVisible();
  await expect(page.locator(".prune-badge").getByText("Likely supported")).toBeVisible();
  await expect(page.getByText("model.layers.", { exact: true })).toBeVisible();
  await expect(page.getByText("Transformer block stack detected")).toBeVisible();

  await page.getByRole("button", { name: "Back To Builder" }).click();
  await expect(page.getByRole("heading", { name: "Block Library" })).toBeVisible();
});

test("pruning presets update the block pruning summary", async ({ page }) => {
  await page.route("http://127.0.0.1:8787/health", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ ok: true })
    });
  });

  await page.route("http://127.0.0.1:8787/api/hf/inspect", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        ok: true,
        result: {
          modelId: "microsoft/Phi-3-mini-4k-instruct",
          resolvedFamily: "phi3",
          config: {
            model_type: "phi3",
            num_hidden_layers: 8,
            hidden_size: 3072
          },
          weightIndex: {
            weight_map: {
              "model.layers.0.input_layernorm.weight": "model-00001-of-00002.safetensors"
            }
          },
          inspection: {
            layerCountHint: 8,
            layerCountKey: "num_hidden_layers",
            detectedLayerPrefix: "model.layers.",
            detectedLayerIndices: Array.from({ length: 8 }, (_, index) => index),
            broadPruningSupported: true,
            sampleLayerKeys: [
              "model.layers.0.input_layernorm.weight",
              "model.layers.0.mlp.down_proj.weight"
            ]
          }
        }
      })
    });
  });

  await page.goto("/#/prune");
  await page.getByLabel("Hugging Face Model ID").fill("microsoft/Phi-3-mini-4k-instruct");
  await page.getByRole("button", { name: "Fetch Model Metadata" }).click();
  await expect(page.getByText("8 total")).toBeVisible();

  await page.getByLabel("Quick Preset").selectOption("keep-every-other");
  await expect(page.getByText("Dropped Layers")).toBeVisible();
  await expect(page.getByText(/1, 3, 5, 7/)).toBeVisible();
  await expect(page.getByText("Kept Layers")).toBeVisible();
  await expect(page.getByText(/0, 2, 4, 6/)).toBeVisible();
});

test("pruning page surfaces metadata fetch failures clearly", async ({ page }) => {
  await page.route("http://127.0.0.1:8787/health", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ ok: true })
    });
  });

  await page.route("http://127.0.0.1:8787/api/hf/inspect", async (route) => {
    await route.fulfill({
      status: 400,
      contentType: "application/json",
      body: JSON.stringify({
        ok: false,
        error: "Request failed with 401 for config.json"
      })
    });
  });

  await page.goto("/#/prune");
  await page.getByLabel("Hugging Face Model ID").fill("google/gemma-3-4b-it");
  await page.getByRole("button", { name: "Fetch Model Metadata" }).click();

  await expect(page.getByText("Metadata Error")).toBeVisible();
  await expect(page.getByText("Request failed with 401 for config.json")).toBeVisible();
});

test("pruning page surfaces missing prunable stack and blocks local prune", async ({ page }) => {
  await page.route("http://127.0.0.1:8787/health", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ ok: true })
    });
  });

  await page.route("http://127.0.0.1:8787/api/hf/inspect", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        ok: true,
        result: {
          modelId: "custom/opaque-model",
          resolvedFamily: "unknown",
          config: {
            hidden_size: 1024
          },
          weightIndex: null,
          inspection: {
            layerCountHint: null,
            layerCountKey: null,
            detectedLayerPrefix: null,
            detectedLayerIndices: [],
            broadPruningSupported: false,
            sampleLayerKeys: []
          }
        }
      })
    });
  });

  await page.goto("/#/prune");
  await page.getByLabel("Hugging Face Model ID").fill("custom/opaque-model");
  await page.getByRole("button", { name: "Fetch Model Metadata" }).click();

  await expect(page.getByText("No prunable stack detected")).toBeVisible();
  await expect(page.getByText("No prunable transformer block stack was detected from the fetched model metadata.")).toBeVisible();
  await expect(page.getByRole("button", { name: "Run Local Prune" })).toBeDisabled();
});
