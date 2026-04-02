import type { ModelGraph } from "@neural-playground/block-schema";

export function numberConfig(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string" && value.trim() !== "") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

export function inferSequenceDimensions(graph: ModelGraph): Map<string, number | null> {
  const inferred = new Map<string, number | null>();
  let changed = true;
  let iterations = 0;

  while (changed && iterations < graph.nodes.length + 2) {
    changed = false;
    iterations += 1;

    for (const node of graph.nodes) {
      const explicit = getExplicitSequenceDim(node);
      const previous = inferred.get(node.id);
      const next = explicit ?? inferFromIncoming(graph, node.id, inferred);

      if (previous !== next) {
        inferred.set(node.id, next);
        changed = true;
      }
    }
  }

  return inferred;
}

export function inferNodeSequenceDim(
  node: ModelGraph["nodes"][number],
  inferredSequenceDims: Map<string, number | null>
): number | null {
  return inferredSequenceDims.get(node.id) ?? null;
}

export function getInputSequenceDim(
  node: ModelGraph["nodes"][number],
  inferredSequenceDims: Map<string, number | null>
): number | null {
  switch (node.type) {
    case "TransformerBlock":
    case "GPT2Block":
    case "LlamaBlock":
      return numberConfig(node.config.dModel);
    case "LayerNorm":
    case "LlamaFinalRMSNorm":
    case "GPT2FinalLayerNorm":
    case "MLP":
    case "Add":
    case "Dropout":
    case "LlamaLMHead":
    case "GPT2LMHead":
    case "Output":
      return inferNodeSequenceDim(node, inferredSequenceDims);
    default:
      return null;
  }
}

export function getOutputSequenceDim(
  node: ModelGraph["nodes"][number],
  inferredSequenceDims: Map<string, number | null>
): number | null {
  switch (node.type) {
    case "Embedding":
    case "LlamaTokenEmbedding":
    case "GPT2TokenEmbedding":
    case "GPT2PositionEmbedding":
      return numberConfig(node.config.embeddingDim);
    case "TransformerBlock":
    case "GPT2Block":
    case "LlamaBlock":
      return numberConfig(node.config.dModel);
    case "LayerNorm":
    case "LlamaFinalRMSNorm":
    case "GPT2FinalLayerNorm":
    case "MLP":
    case "Add":
    case "Dropout":
      return inferNodeSequenceDim(node, inferredSequenceDims);
    default:
      return null;
  }
}

function getExplicitSequenceDim(node: ModelGraph["nodes"][number]): number | null {
  switch (node.type) {
    case "Embedding":
    case "LlamaTokenEmbedding":
    case "GPT2TokenEmbedding":
    case "GPT2PositionEmbedding":
      return numberConfig(node.config.embeddingDim);
    case "TransformerBlock":
    case "GPT2Block":
    case "LlamaBlock":
      return numberConfig(node.config.dModel);
    default:
      return null;
  }
}

function inferFromIncoming(
  graph: ModelGraph,
  nodeId: string,
  inferredSequenceDims: Map<string, number | null>
): number | null {
  const incoming = graph.edges.filter((edge) => edge.target === nodeId);
  const dims = incoming
    .map((edge) => inferredSequenceDims.get(edge.source) ?? null)
    .filter((value): value is number => value !== null);

  if (dims.length === 0) {
    return null;
  }

  const first = dims[0]!;
  return dims.every((dim) => dim === first) ? first : null;
}
