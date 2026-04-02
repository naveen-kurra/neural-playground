import type { HuggingFaceFetchResult } from "./huggingface";
import type { LayerRemapEntry } from "./pruning";

export type LayerPruningStrategy = "drop";

export type PruningManifest = {
  version: 1;
  modelId: string;
  family: "gpt2" | "llama" | "unknown";
  layerCountKey: string | null;
  originalLayerCount: number;
  keptLayerIndices: number[];
  droppedLayerIndices: number[];
  remap: LayerRemapEntry[];
  detectedLayerPrefix: string | null;
  strategy: LayerPruningStrategy;
};

export function buildUpdatedConfig(
  result: HuggingFaceFetchResult,
  remap: LayerRemapEntry[]
): Record<string, unknown> {
  const nextConfig: Record<string, unknown> = { ...result.config };
  const layerCount = remap.length;
  if (result.inspection.layerCountKey) {
    nextConfig[result.inspection.layerCountKey] = layerCount;
  }
  return nextConfig;
}

export function buildPruningManifest(
  result: HuggingFaceFetchResult,
  keptLayerIndices: number[],
  droppedLayerIndices: number[],
  remap: LayerRemapEntry[]
): PruningManifest {
  return {
    version: 1,
    modelId: result.modelId,
    family: result.resolvedFamily,
    layerCountKey: result.inspection.layerCountKey,
    originalLayerCount: result.inspection.layerCountHint ?? result.inspection.detectedLayerIndices.length,
    keptLayerIndices,
    droppedLayerIndices,
    remap,
    detectedLayerPrefix: result.inspection.detectedLayerPrefix,
    strategy: "drop"
  };
}

function escapePythonString(value: string): string {
  return value.replaceAll("\\", "\\\\").replaceAll('"', '\\"');
}

export function buildWeightRemapScript(manifest: PruningManifest): string {
  const remapJson = JSON.stringify(manifest.remap, null, 2);
  const detectedLayerPrefix = manifest.detectedLayerPrefix ? escapePythonString(manifest.detectedLayerPrefix) : "";
  const layerCountKey = manifest.layerCountKey ? escapePythonString(manifest.layerCountKey) : "";

  return `from __future__ import annotations

import json
from pathlib import Path

try:
    from safetensors.torch import load_file, save_file
except ImportError as exc:
    raise SystemExit("Install safetensors first: pip install safetensors") from exc


REMAP = ${remapJson}
LAYER_PREFIX = "${detectedLayerPrefix}"
LAYER_COUNT_KEY = "${layerCountKey}"


def remap_key(key: str, index_map: dict[int, int]) -> str | None:
    if not LAYER_PREFIX or not key.startswith(LAYER_PREFIX):
        return key

    suffix = key[len(LAYER_PREFIX):]
    parts = suffix.split(".", 1)
    if not parts or not parts[0].isdigit():
        return key

    old_index = int(parts[0])
    if old_index not in index_map:
        return None

    rest = parts[1] if len(parts) > 1 else ""
    new_index = index_map[old_index]
    return f"{LAYER_PREFIX}{new_index}.{rest}" if rest else f"{LAYER_PREFIX}{new_index}"


def main():
    source_weights = Path("original.safetensors")
    target_weights = Path("pruned.safetensors")
    source_config = Path("config.json")
    target_config = Path("pruned-config.json")

    index_map = {entry["oldIndex"]: entry["newIndex"] for entry in REMAP}
    tensors = load_file(str(source_weights))

    remapped_tensors = {}
    for key, value in tensors.items():
        new_key = remap_key(key, index_map)
        if new_key is None:
            continue
        remapped_tensors[new_key] = value

    save_file(remapped_tensors, str(target_weights))

    config = json.loads(source_config.read_text(encoding="utf-8"))
    if LAYER_COUNT_KEY:
        config[LAYER_COUNT_KEY] = len(REMAP)
    target_config.write_text(json.dumps(config, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
`;
}

