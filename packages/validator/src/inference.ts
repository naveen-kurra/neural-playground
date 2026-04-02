import { getBlockDefinition, type ModelGraph } from "@neural-playground/block-schema";

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
  const def = getBlockDefinition(node.type);
  // If block declares its own explicit dim field, use it directly
  if (def.sequenceDimField) {
    return numberConfig((node.config as Record<string, unknown>)[def.sequenceDimField]);
  }
  // If block accepts sequence input, its expected input dim is inferred from incoming edges
  const hasSequenceInput = def.inputContracts.some((c) => c.kind === "sequence");
  if (hasSequenceInput) {
    return inferNodeSequenceDim(node, inferredSequenceDims);
  }
  return null;
}

export function getOutputSequenceDim(
  node: ModelGraph["nodes"][number],
  inferredSequenceDims: Map<string, number | null>
): number | null {
  const def = getBlockDefinition(node.type);
  // If block declares its own explicit dim field, its output is that dim
  if (def.sequenceDimField) {
    return numberConfig((node.config as Record<string, unknown>)[def.sequenceDimField]);
  }
  // If block produces sequence output, its output dim is inferred from incoming edges
  const hasSequenceOutput = def.outputContracts.some((c) => c.kind === "sequence");
  if (hasSequenceOutput) {
    return inferNodeSequenceDim(node, inferredSequenceDims);
  }
  return null;
}

function getExplicitSequenceDim(node: ModelGraph["nodes"][number]): number | null {
  const def = getBlockDefinition(node.type);
  if (!def.sequenceDimField) return null;
  return numberConfig((node.config as Record<string, unknown>)[def.sequenceDimField]);
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
