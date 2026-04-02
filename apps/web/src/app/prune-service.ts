export type RunLocalPruneRequest = {
  inputDir?: string;
  outputDir: string;
  modelId: string;
  cacheDir?: string;
  droppedLayerIndices: number[];
  layerPrefix?: string;
  layerCountKey?: string;
  strategy?: "drop";
};

export type RunLocalPruneResult = {
  ok: true;
  downloadedFromHub: boolean;
  inputDir: string;
  outputDir: string;
  layerPrefix: string | null;
  layerCountKey: string | null;
  keptLayerCount: number;
  droppedLayerCount: number;
  writtenFiles: string[];
  copiedSupportFiles: string[];
  validation: {
    ok: boolean;
    config: {
      ok: boolean;
      layerCountKey: string | null;
      expectedLayerCount: number;
      configuredLayerCount: number | null;
    };
    checkpoint: {
      ok: boolean;
      expectedLayerPrefix: string;
      detectedLayerPrefix: string | null;
      expectedLayerIndices: number[];
      detectedLayerIndices: number[];
      writtenWeightBytes: number;
      originalLayerCount: number;
      keptLayerCount: number;
    };
    load: {
      ok: boolean;
      originalConfigLoadOk: boolean;
      prunedConfigLoadOk: boolean;
      originalModelLoadOk: boolean;
      prunedModelLoadOk: boolean;
      skippedModelLoadReason: string | null;
      warnings: string[];
      memory: {
        device: string;
        originalModelBytes: number | null;
        prunedModelBytes: number | null;
        originalModelMiB: number | null;
        prunedModelMiB: number | null;
      };
    };
  };
};

export type PruneServiceLogEvent = {
  type: "log";
  stage: string;
  message: string;
  [key: string]: unknown;
};

type PruneServiceResultEvent = {
  type: "result";
  result: RunLocalPruneResult;
};

type PruneServiceErrorEvent = {
  type: "error";
  error: string;
};

type PruneServiceEvent = PruneServiceLogEvent | PruneServiceResultEvent | PruneServiceErrorEvent;

const DEFAULT_PRUNE_SERVICE_URL = "http://127.0.0.1:8787";

export async function checkPruneServiceHealth(baseUrl = DEFAULT_PRUNE_SERVICE_URL): Promise<boolean> {
  try {
    const response = await fetch(`${baseUrl}/health`);
    return response.ok;
  } catch {
    return false;
  }
}

export async function runLocalPrune(
  request: RunLocalPruneRequest,
  onLog?: (event: PruneServiceLogEvent) => void,
  baseUrl = DEFAULT_PRUNE_SERVICE_URL
): Promise<RunLocalPruneResult> {
  const response = await fetch(`${baseUrl}/api/prune/run`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      ...request,
      inputDir: request.inputDir ?? "",
      cacheDir: request.cacheDir ?? "",
      strategy: request.strategy ?? "drop"
    })
  });

  if (!response.ok || !response.body) {
    let payload: { ok?: boolean; error?: string } | null = null;
    try {
      payload = (await response.json()) as { ok?: boolean; error?: string };
    } catch {
      payload = null;
    }
    throw new Error(payload?.error ?? "Local prune run failed.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalResult: RunLocalPruneResult | null = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) {
        continue;
      }
      const event = JSON.parse(trimmed) as PruneServiceEvent;
      if (event.type === "log") {
        onLog?.(event);
      } else if (event.type === "result") {
        finalResult = event.result;
      } else if (event.type === "error") {
        throw new Error(event.error);
      }
    }
  }

  if (buffer.trim()) {
    const event = JSON.parse(buffer.trim()) as PruneServiceEvent;
    if (event.type === "log") {
      onLog?.(event);
    } else if (event.type === "result") {
      finalResult = event.result;
    } else if (event.type === "error") {
      throw new Error(event.error);
    }
  }

  if (!finalResult) {
    throw new Error("Prune run finished without a final result.");
  }

  return finalResult;
}
