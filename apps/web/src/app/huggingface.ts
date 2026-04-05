export type HuggingFaceFetchResult = {
  modelId: string;
  config: Record<string, unknown>;
  weightIndex: Record<string, unknown> | null;
  resolvedFamily: "gpt2" | "llama" | "phi3" | "unknown";
  inspection: HuggingFaceModelInspection;
};

export type HuggingFaceModelInspection = {
  layerCountHint: number | null;
  layerCountKey: string | null;
  detectedLayerPrefix: string | null;
  detectedLayerIndices: number[];
  broadPruningSupported: boolean;
  sampleLayerKeys: string[];
};

function normalizeModelId(modelId: string): string {
  return modelId.trim().replace(/^https?:\/\/huggingface\.co\//, "").replace(/\/+$/, "");
}

function getNestedValue(data: Record<string, unknown>, path: string): unknown {
  return path.split(".").reduce<unknown>((current, segment) => {
    if (!current || typeof current !== "object" || Array.isArray(current) || !(segment in current)) {
      return undefined;
    }
    return (current as Record<string, unknown>)[segment];
  }, data);
}

function inferFamily(config: Record<string, unknown>): "gpt2" | "llama" | "phi3" | "unknown" {
  const modelType = String(config.model_type ?? "").toLowerCase();
  if (modelType === "gpt2") {
    return "gpt2";
  }
  if (modelType === "llama") {
    return "llama";
  }
  if (modelType === "phi3" || modelType === "phi-3") {
    return "phi3";
  }

  const architectures = Array.isArray(config.architectures) ? config.architectures.map((value) => String(value).toLowerCase()) : [];
  if (architectures.some((value) => value.includes("phi3") || value.includes("phi-3"))) {
    return "phi3";
  }
  return "unknown";
}

function getLayerCountHint(config: Record<string, unknown>): { value: number | null; key: string | null } {
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

function inspectWeightIndex(
  weightIndex: Record<string, unknown> | null,
  layerCountHint: number | null
): Omit<HuggingFaceModelInspection, "layerCountHint" | "layerCountKey"> {
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
  const prefixMatches = new Map<string, Set<number>>();

  for (const key of keys) {
    const match = key.match(/^(.*?)(\d+)\.(.+)$/);
    if (!match) {
      continue;
    }

    const prefix = match[1]!;
    const layerIndex = Number(match[2]);
    if (!Number.isInteger(layerIndex)) {
      continue;
    }

    if (!prefixMatches.has(prefix)) {
      prefixMatches.set(prefix, new Set<number>());
    }
    prefixMatches.get(prefix)!.add(layerIndex);
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
    (layerCountHint === null || detected.indices.length === layerCountHint || detected.indices.length >= Math.max(2, Math.floor(layerCountHint * 0.75)));

  return {
    detectedLayerPrefix: detected.prefix,
    detectedLayerIndices: detected.indices,
    broadPruningSupported,
    sampleLayerKeys
  };
}

async function fetchJson(url: string): Promise<Record<string, unknown>> {
  const response = await fetch(url, {
    headers: {
      Accept: "application/json"
    }
  });

  if (!response.ok) {
    throw new Error(`Request failed with ${response.status} for ${url}`);
  }

  const data = await response.json();
  if (!data || typeof data !== "object" || Array.isArray(data)) {
    throw new Error(`Expected JSON object from ${url}`);
  }

  return data as Record<string, unknown>;
}

const DEFAULT_PRUNE_SERVICE_URL =
  (import.meta as ImportMeta & { env?: Record<string, string | undefined> }).env?.VITE_PRUNE_SERVICE_URL ??
  "http://127.0.0.1:8787";

export async function fetchHuggingFaceModelViaService(
  modelIdInput: string,
  baseUrl = DEFAULT_PRUNE_SERVICE_URL
): Promise<HuggingFaceFetchResult> {
  const modelId = normalizeModelId(modelIdInput);
  if (!modelId) {
    throw new Error("Enter a Hugging Face model id first.");
  }

  const response = await fetch(`${baseUrl}/api/hf/inspect`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ modelId })
  });
  const payload = (await response.json()) as { ok?: boolean; error?: string; result?: HuggingFaceFetchResult };
  if (!response.ok || !payload.ok || !payload.result) {
    throw new Error(payload.error ?? "Failed to inspect model metadata.");
  }
  return payload.result;
}

export async function fetchHuggingFaceModel(modelIdInput: string): Promise<HuggingFaceFetchResult> {
  const modelId = normalizeModelId(modelIdInput);
  if (!modelId) {
    throw new Error("Enter a Hugging Face model id first.");
  }

  const baseUrl = `https://huggingface.co/${modelId}/resolve/main`;
  const config = await fetchJson(`${baseUrl}/config.json`);

  let weightIndex: Record<string, unknown> | null = null;
  try {
    weightIndex = await fetchJson(`${baseUrl}/model.safetensors.index.json`);
  } catch {
    weightIndex = null;
  }

  const layerCount = getLayerCountHint(config);
  const weightInspection = inspectWeightIndex(weightIndex, layerCount.value);

  return {
    modelId,
    config,
    weightIndex,
    resolvedFamily: inferFamily(config),
    inspection: {
      layerCountHint: layerCount.value,
      layerCountKey: layerCount.key,
      ...weightInspection
    }
  };
}
