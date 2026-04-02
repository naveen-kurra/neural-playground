import { type BlockNode, type ModelGraph } from "@neural-playground/block-schema";
import type { ExportContext } from "./types";

function findSingleInput(graph: ModelGraph): BlockNode {
  const inputs = graph.nodes.filter((node) => node.type === "Input");
  if (inputs.length !== 1) {
    throw new Error("Project export currently requires exactly one Input node.");
  }
  return inputs[0]!;
}

function topologicalSort(graph: ModelGraph): BlockNode[] {
  const indegree = new Map<string, number>();
  const outgoing = new Map<string, string[]>();

  for (const node of graph.nodes) {
    indegree.set(node.id, 0);
    outgoing.set(node.id, []);
  }

  for (const edge of graph.edges) {
    indegree.set(edge.target, (indegree.get(edge.target) ?? 0) + 1);
    outgoing.get(edge.source)?.push(edge.target);
  }

  const ready: string[] = graph.nodes.filter((node) => (indegree.get(node.id) ?? 0) === 0).map((node) => node.id);
  const orderedIds: string[] = [];

  while (ready.length > 0) {
    const nextId = ready.shift()!;
    orderedIds.push(nextId);

    for (const targetId of outgoing.get(nextId) ?? []) {
      const nextDegree = (indegree.get(targetId) ?? 0) - 1;
      indegree.set(targetId, nextDegree);
      if (nextDegree === 0) {
        ready.push(targetId);
      }
    }
  }

  if (orderedIds.length !== graph.nodes.length) {
    throw new Error("Project export requires an acyclic graph.");
  }

  return orderedIds
    .map((id) => graph.nodes.find((node) => node.id === id))
    .filter((node): node is BlockNode => node !== undefined);
}

export function sanitizeNodeId(nodeId: string): string {
  return nodeId.replace(/[^a-zA-Z0-9_]/g, "_").toLowerCase();
}

export function normalizeActivation(value: unknown): string {
  return String(value ?? "gelu").toLowerCase();
}

export function normalizeOptimizer(value: unknown): string {
  const normalized = String(value ?? "AdamW").toLowerCase();
  return normalized === "sgd" ? "sgd" : "adamw";
}

export function normalizeLoss(value: unknown): string {
  const normalized = String(value ?? "CrossEntropy").toLowerCase();
  return normalized === "crossentropy" ? "cross_entropy" : normalized;
}

export function buildContext(graph: ModelGraph): ExportContext {
  const orderedNodes = topologicalSort(graph);
  const inputNode = findSingleInput(graph);
  const embeddingNode = orderedNodes.find((node) => node.type === "Embedding");
  if (!embeddingNode) {
    throw new Error("Project export currently requires an Embedding node.");
  }

  const warnings: string[] = [];
  if (
    graph.nodes.some(
      (node) => graph.edges.filter((edge) => edge.target === node.id).length > 1 && node.type !== "Add"
    )
  ) {
    warnings.push("Non-Add branch merges are flattened in export order. Full multi-input graph execution is not implemented yet.");
  }

  const firstTransformer = orderedNodes.find((node) => node.type === "TransformerBlock");
  const firstMlp = orderedNodes.find((node) => node.type === "MLP");

  let currentDim = Number(embeddingNode.config.embeddingDim ?? 768);
  for (const node of orderedNodes) {
    if (node.type === "TransformerBlock") {
      currentDim = Number(node.config.dModel ?? currentDim);
    }
  }

  return {
    graph,
    orderedNodes,
    warnings,
    vocabSize: Number(embeddingNode.config.vocabSize ?? 32000),
    sequenceLength: Number(inputNode.config.sequenceLength ?? 1024),
    embeddingDim: Number(embeddingNode.config.embeddingDim ?? 768),
    transformerCount: orderedNodes.filter((node) => node.type === "TransformerBlock").length,
    defaultHeads: Number(firstTransformer?.config.numHeads ?? 12),
    defaultFfnHidden: Number(firstTransformer?.config.ffnHidden ?? firstMlp?.config.hiddenDim ?? currentDim * 4),
    defaultActivation: normalizeActivation(firstMlp?.config.activation ?? graph.training.activation ?? "gelu"),
    optimizerName: normalizeOptimizer(graph.training.optimizer),
    lossName: normalizeLoss(graph.training.loss),
    lastSequenceDim: currentDim
  };
}

export function exportActivationName(graph: ModelGraph, fallback: string): string {
  if (graph.training.activation === "Custom") {
    const customName = (graph.training.activationCustomName ?? "").trim();
    return customName || "custom_activation";
  }
  return normalizeActivation(graph.training.activation ?? fallback);
}

export function exportLossName(graph: ModelGraph, fallback: string): string {
  if (graph.training.loss === "Custom") {
    const customName = (graph.training.lossCustomName ?? "").trim();
    return customName || "custom_loss";
  }
  return normalizeLoss(graph.training.loss ?? fallback);
}

export function exportOptimizerName(graph: ModelGraph, fallback: string): string {
  if (graph.training.optimizer === "Custom") {
    const customName = (graph.training.optimizerCustomName ?? "").trim();
    return customName || "custom_optimizer";
  }
  return normalizeOptimizer(graph.training.optimizer ?? fallback);
}
