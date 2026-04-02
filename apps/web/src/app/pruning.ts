export type LayerRemapEntry = {
  newIndex: number;
  oldIndex: number;
};

export function getEffectiveLayerIndices(
  detectedLayerIndices: number[],
  layerCountHint: number | null
): number[] {
  if (detectedLayerIndices.length > 0) {
    return detectedLayerIndices;
  }

  if (layerCountHint && layerCountHint > 0) {
    return Array.from({ length: layerCountHint }, (_, index) => index);
  }

  return [];
}

export function buildLayerRemap(keptLayerIndices: number[]): LayerRemapEntry[] {
  return keptLayerIndices.map((oldIndex, newIndex) => ({ newIndex, oldIndex }));
}

