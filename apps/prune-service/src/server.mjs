import { existsSync, readFileSync } from "node:fs";
import { spawn } from "node:child_process";
import { createServer } from "node:http";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(__dirname, "../../..");
const configPath = join(__dirname, "..", "base_conf.json");
const serviceConfig = existsSync(configPath)
  ? JSON.parse(readFileSync(configPath, "utf8"))
  : {};
const HOST = process.env.PRUNE_SERVICE_HOST || serviceConfig.host || "127.0.0.1";
const PORT = Number(process.env.PORT || process.env.PRUNE_SERVICE_PORT || serviceConfig.port || 8787);
const runnerPath = join(repoRoot, "scripts", "prune_checkpoint.py");
const preferredPython = process.env.PRUNE_PYTHON
  || String(serviceConfig.pythonPath ?? "").trim()
  || join(repoRoot, ".venv-prune", "bin", "python3");
const pythonCommand = existsSync(preferredPython) ? preferredPython : "python3";
const hfToken = process.env.HF_TOKEN?.trim() || String(serviceConfig.hfToken ?? "").trim();

function sendJson(response, statusCode, payload) {
  response.writeHead(statusCode, {
    "Content-Type": "application/json; charset=utf-8",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type"
  });
  response.end(JSON.stringify(payload));
}

function readJsonBody(request) {
  return new Promise((resolveBody, rejectBody) => {
    let body = "";
    request.on("data", (chunk) => {
      body += chunk;
      if (body.length > 1024 * 1024) {
        rejectBody(new Error("Request body too large."));
      }
    });
    request.on("end", () => {
      if (!body.trim()) {
        resolveBody({});
        return;
      }
      try {
        resolveBody(JSON.parse(body));
      } catch {
        rejectBody(new Error("Expected a JSON request body."));
      }
    });
    request.on("error", rejectBody);
  });
}

function normalizeModelId(modelId) {
  return String(modelId ?? "").trim().replace(/^https?:\/\/huggingface\.co\//, "").replace(/\/+$/, "");
}

function inferFamily(config) {
  const modelType = String(config?.model_type ?? "").toLowerCase();
  if (modelType === "gpt2") {
    return "gpt2";
  }
  if (modelType === "llama") {
    return "llama";
  }
  if (modelType === "phi3" || modelType === "phi-3") {
    return "phi3";
  }
  const architectures = Array.isArray(config?.architectures) ? config.architectures.map((value) => String(value).toLowerCase()) : [];
  if (architectures.some((value) => value.includes("phi3") || value.includes("phi-3"))) {
    return "phi3";
  }
  return "unknown";
}

function getNestedValue(data, path) {
  return path.split(".").reduce((current, segment) => {
    if (!current || typeof current !== "object" || !(segment in current)) {
      return undefined;
    }
    return current[segment];
  }, data);
}

function getLayerCountHint(config) {
  const candidates = [
    "num_hidden_layers",
    "n_layer",
    "num_layers",
    "n_layers",
    "text_config.num_hidden_layers",
    "language_config.num_hidden_layers"
  ];
  for (const key of candidates) {
    const value = getNestedValue(config, key);
    if (typeof value === "number" && Number.isFinite(value)) {
      return { value: Number(value), key };
    }
  }
  return { value: null, key: null };
}

function inspectWeightIndex(weightIndex, layerCountHint) {
  const weightMap = weightIndex?.weight_map;
  if (!weightMap || typeof weightMap !== "object" || Array.isArray(weightMap)) {
    return {
      detectedLayerPrefix: null,
      detectedLayerIndices: [],
      broadPruningSupported: false,
      sampleLayerKeys: []
    };
  }

  const keys = Object.keys(weightMap);
  const prefixMatches = new Map();

  for (const key of keys) {
    const match = key.match(/^(.*?)(\d+)\.(.+)$/);
    if (!match) {
      continue;
    }

    const prefix = match[1];
    const layerIndex = Number(match[2]);
    if (!Number.isInteger(layerIndex)) {
      continue;
    }

    if (!prefixMatches.has(prefix)) {
      prefixMatches.set(prefix, new Set());
    }
    prefixMatches.get(prefix).add(layerIndex);
  }

  const rankedPrefixes = [...prefixMatches.entries()]
    .map(([prefix, indices]) => ({
      prefix,
      indices: [...indices].sort((a, b) => a - b)
    }))
    .sort((left, right) => right.indices.length - left.indices.length);

  const detected = rankedPrefixes[0];
  if (!detected) {
    return {
      detectedLayerPrefix: null,
      detectedLayerIndices: [],
      broadPruningSupported: false,
      sampleLayerKeys: []
    };
  }

  const sampleLayerKeys = keys.filter((key) => key.startsWith(detected.prefix)).slice(0, 6);
  const broadPruningSupported =
    detected.indices.length >= 2 &&
    (layerCountHint === null ||
      detected.indices.length === layerCountHint ||
      detected.indices.length >= Math.max(2, Math.floor(layerCountHint * 0.75)));

  return {
    detectedLayerPrefix: detected.prefix,
    detectedLayerIndices: detected.indices,
    broadPruningSupported,
    sampleLayerKeys
  };
}

async function fetchHfJson(modelId, filename) {
  const response = await fetch(`https://huggingface.co/${modelId}/resolve/main/${filename}`, {
    headers: {
      Accept: "application/json",
      ...(hfToken ? { Authorization: `Bearer ${hfToken}` } : {})
    }
  });
  if (!response.ok) {
    throw new Error(`Request failed with ${response.status} for ${filename}`);
  }
  const data = await response.json();
  if (!data || typeof data !== "object" || Array.isArray(data)) {
    throw new Error(`Expected JSON object for ${filename}`);
  }
  return data;
}

async function inspectHuggingFaceModel(modelIdInput) {
  const modelId = normalizeModelId(modelIdInput);
  if (!modelId) {
    throw new Error("Enter a Hugging Face model id first.");
  }

  const config = await fetchHfJson(modelId, "config.json");
  let weightIndex = null;
  try {
    weightIndex = await fetchHfJson(modelId, "model.safetensors.index.json");
  } catch {
    weightIndex = null;
  }

  const layerCount = getLayerCountHint(config);
  const inspection = inspectWeightIndex(weightIndex, layerCount.value);

  return {
    modelId,
    config,
    weightIndex,
    resolvedFamily: inferFamily(config),
    inspection: {
      layerCountHint: layerCount.value,
      layerCountKey: layerCount.key,
      ...inspection
    }
  };
}

function validatePruneRequest(payload) {
  if (!payload || typeof payload !== "object") {
    throw new Error("Expected a JSON object.");
  }

  const inputDir = typeof payload.inputDir === "string" ? payload.inputDir.trim() : "";
  const outputDir = typeof payload.outputDir === "string" ? payload.outputDir.trim() : "";
  const modelId = typeof payload.modelId === "string" ? payload.modelId.trim() : "";
  const droppedLayerIndices = Array.isArray(payload.droppedLayerIndices)
    ? payload.droppedLayerIndices.filter((value) => Number.isInteger(value)).map((value) => Number(value))
    : [];

  if (!inputDir && !modelId) {
    throw new Error("Provide either `inputDir` or `modelId`.");
  }
  if (!outputDir) {
    throw new Error("`outputDir` is required.");
  }

  return {
    inputDir,
    outputDir,
    modelId,
    cacheDir: typeof payload.cacheDir === "string" ? payload.cacheDir.trim() : "",
    droppedLayerIndices,
    layerPrefix: typeof payload.layerPrefix === "string" ? payload.layerPrefix : "",
    layerCountKey: typeof payload.layerCountKey === "string" ? payload.layerCountKey : "",
    strategy: typeof payload.strategy === "string" ? payload.strategy : "drop"
  };
}

function runPruneCommand(requestPayload) {
  return new Promise((resolveRun, rejectRun) => {
    const args = [
      runnerPath,
      "--output-dir",
      requestPayload.outputDir,
      "--strategy",
      requestPayload.strategy,
      "--dropped-layers",
      requestPayload.droppedLayerIndices.join(",")
    ];

    if (requestPayload.inputDir) {
      args.push("--input-dir", requestPayload.inputDir);
    }
    if (requestPayload.modelId) {
      args.push("--model-id", requestPayload.modelId);
    }
    if (hfToken) {
      args.push("--hf-token", hfToken);
    }
    if (requestPayload.cacheDir) {
      args.push("--cache-dir", requestPayload.cacheDir);
    }
    if (requestPayload.layerPrefix) {
      args.push("--layer-prefix", requestPayload.layerPrefix);
    }
    if (requestPayload.layerCountKey) {
      args.push("--layer-count-key", requestPayload.layerCountKey);
    }

    const child = spawn(pythonCommand, args, {
      cwd: repoRoot,
      stdio: ["ignore", "pipe", "pipe"]
    });

    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });
    child.on("error", (error) => {
      rejectRun(error);
    });
    child.on("close", (code) => {
      if (code !== 0) {
        rejectRun(new Error(stderr.trim() || stdout.trim() || `Prune runner exited with code ${code}.`));
        return;
      }

      try {
        resolveRun(JSON.parse(stdout));
      } catch {
        rejectRun(new Error(`Prune runner returned invalid JSON.\n${stdout}`));
      }
    });
  });
}

function sendNdjsonEvent(response, payload) {
  response.write(`${JSON.stringify(payload)}\n`);
}

function parseJsonLines(buffer, onLine) {
  const lines = buffer.split("\n");
  const rest = lines.pop() ?? "";
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) {
      continue;
    }
    onLine(trimmed);
  }
  return rest;
}

function streamPruneCommand(response, requestPayload) {
  return new Promise((resolveStream, rejectStream) => {
    const args = [
      runnerPath,
      "--output-dir",
      requestPayload.outputDir,
      "--strategy",
      requestPayload.strategy,
      "--dropped-layers",
      requestPayload.droppedLayerIndices.join(",")
    ];

    if (requestPayload.inputDir) {
      args.push("--input-dir", requestPayload.inputDir);
    }
    if (requestPayload.modelId) {
      args.push("--model-id", requestPayload.modelId);
    }
    if (hfToken) {
      args.push("--hf-token", hfToken);
    }
    if (requestPayload.cacheDir) {
      args.push("--cache-dir", requestPayload.cacheDir);
    }
    if (requestPayload.layerPrefix) {
      args.push("--layer-prefix", requestPayload.layerPrefix);
    }
    if (requestPayload.layerCountKey) {
      args.push("--layer-count-key", requestPayload.layerCountKey);
    }

    const child = spawn(pythonCommand, args, {
      cwd: repoRoot,
      stdio: ["ignore", "pipe", "pipe"]
    });

    let stdoutBuffer = "";
    let stderrBuffer = "";
    let finalResult = null;
    let rawError = "";

    child.stdout.on("data", (chunk) => {
      stdoutBuffer += chunk.toString();
      stdoutBuffer = parseJsonLines(stdoutBuffer, (line) => {
        try {
          finalResult = JSON.parse(line);
        } catch {
          rawError += `${line}\n`;
        }
      });
    });

    child.stderr.on("data", (chunk) => {
      stderrBuffer += chunk.toString();
      stderrBuffer = parseJsonLines(stderrBuffer, (line) => {
        try {
          const payload = JSON.parse(line);
          sendNdjsonEvent(response, payload);
        } catch {
          sendNdjsonEvent(response, {
            type: "log",
            stage: "stderr",
            message: line
          });
        }
      });
    });

    child.on("error", (error) => {
      rejectStream(error);
    });

    child.on("close", (code) => {
      if (stderrBuffer.trim()) {
        try {
          sendNdjsonEvent(response, JSON.parse(stderrBuffer.trim()));
        } catch {
          sendNdjsonEvent(response, {
            type: "log",
            stage: "stderr",
            message: stderrBuffer.trim()
          });
        }
      }

      if (code !== 0) {
        rejectStream(new Error(rawError.trim() || `Prune runner exited with code ${code}.`));
        return;
      }
      if (!finalResult) {
        rejectStream(new Error("Prune runner did not return a final result."));
        return;
      }

      sendNdjsonEvent(response, { type: "result", result: finalResult });
      resolveStream(finalResult);
    });
  });
}

const server = createServer(async (request, response) => {
  if (!request.url) {
    sendJson(response, 404, { ok: false, error: "Missing request URL." });
    return;
  }

  if (request.method === "OPTIONS") {
    response.writeHead(204, {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type"
    });
    response.end();
    return;
  }

  if (request.method === "GET" && request.url === "/health") {
    sendJson(response, 200, {
      ok: true,
      service: "prune-service",
      pythonCommand,
      runnerPath,
      hfTokenConfigured: Boolean(hfToken)
    });
    return;
  }

  if (request.method === "POST" && request.url === "/api/hf/inspect") {
    try {
      const payload = await readJsonBody(request);
      const result = await inspectHuggingFaceModel(payload.modelId);
      sendJson(response, 200, { ok: true, result });
    } catch (error) {
      sendJson(response, 400, {
        ok: false,
        error: error instanceof Error ? error.message : "Model inspection failed."
      });
    }
    return;
  }

  if (request.method === "POST" && request.url === "/api/prune/run") {
    try {
      const payload = validatePruneRequest(await readJsonBody(request));
      response.writeHead(200, {
        "Content-Type": "application/x-ndjson; charset=utf-8",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type"
      });
      await streamPruneCommand(response, payload);
      response.end();
    } catch (error) {
      if (!response.headersSent) {
        sendJson(response, 400, {
          ok: false,
          error: error instanceof Error ? error.message : "Prune request failed."
        });
      } else {
        sendNdjsonEvent(response, {
          type: "error",
          error: error instanceof Error ? error.message : "Prune request failed."
        });
        response.end();
      }
    }
    return;
  }

  sendJson(response, 404, { ok: false, error: "Route not found." });
});

server.listen(PORT, HOST, () => {
  console.log(`prune-service listening on http://${HOST}:${PORT}`);
});
