import type { ModelGraph } from "@neural-playground/block-schema";

import { createValidationIssue, type ValidationIssue, type ValidationMode } from "./issues";

export function validateTopology(graph: ModelGraph, mode: ValidationMode): ValidationIssue[] {
  const issues: ValidationIssue[] = [];

  if (hasCycle(graph)) {
    issues.push(createValidationIssue("graph_cycle", "Graph contains a cycle, which is not supported."));
  }

  issues.push(...validateNodeArity(graph));

  if (mode === "playground-valid") {
    return issues;
  }

  const orderedNodes = topologicalNodes(graph);

  const inputNodes = graph.nodes.filter((node) => node.type === "Input");
  const outputNodes = graph.nodes.filter((node) => node.type === "Output");
  const embeddingNodes = graph.nodes.filter((node) =>
    node.type === "Embedding" || node.type === "GPT2TokenEmbedding" || node.type === "LlamaTokenEmbedding"
  );

  if (inputNodes.length !== 1) {
    issues.push(createValidationIssue("export_input_count", "Export requires exactly one Input node."));
  }

  if (outputNodes.length !== 1) {
    issues.push(createValidationIssue("export_output_count", "Export requires exactly one Output node."));
  }

  if (embeddingNodes.length !== 1) {
    issues.push(createValidationIssue("export_embedding_count", "Export requires exactly one Embedding node."));
  }

  issues.push(...validateStageOrdering(orderedNodes));

  if (mode === "decoder-training-valid") {
    const output = outputNodes[0];
    if (output) {
      const headType = String(output.config.headType ?? "LanguageModel");
      if (headType !== "LanguageModel") {
        issues.push(
          createValidationIssue(
            "decoder_output_head",
            "Decoder training export requires the Output head type to be LanguageModel."
          )
        );
      }
    }
  }

  return issues;
}

type NodeArityRule = {
  minIncoming?: number;
  maxIncoming?: number;
  minOutgoing?: number;
  maxOutgoing?: number;
};

function validateNodeArity(graph: ModelGraph): ValidationIssue[] {
  const issues: ValidationIssue[] = [];

  for (const node of graph.nodes) {
    const incoming = graph.edges.filter((edge) => edge.target === node.id).length;
    const outgoing = graph.edges.filter((edge) => edge.source === node.id).length;
    const rule = getNodeArityRule(node.type);

    if (rule.minIncoming !== undefined && incoming < rule.minIncoming) {
      issues.push({
        code: "node_input_arity",
        message: `${node.type} requires ${formatArityExpectation(rule.minIncoming, "incoming")} but found ${incoming}.`,
        nodeId: node.id
      });
    }

    if (rule.maxIncoming !== undefined && incoming > rule.maxIncoming) {
      issues.push({
        code: "node_input_arity",
        message: `${node.type} allows ${formatArityExpectation(rule.maxIncoming, "incoming")} but found ${incoming}.`,
        nodeId: node.id
      });
    }

    if (rule.minOutgoing !== undefined && outgoing < rule.minOutgoing) {
      issues.push({
        code: "node_output_arity",
        message: `${node.type} requires ${formatArityExpectation(rule.minOutgoing, "outgoing")} but found ${outgoing}.`,
        nodeId: node.id
      });
    }

    if (rule.maxOutgoing !== undefined && outgoing > rule.maxOutgoing) {
      issues.push({
        code: "node_output_arity",
        message: `${node.type} allows ${formatArityExpectation(rule.maxOutgoing, "outgoing")} but found ${outgoing}.`,
        nodeId: node.id
      });
    }
  }

  return issues;
}

function getNodeArityRule(type: ModelGraph["nodes"][number]["type"]): NodeArityRule {
  switch (type) {
    case "Input":
      return { maxIncoming: 0 };
    case "Add":
      return { minIncoming: 2, maxIncoming: 2, minOutgoing: 1, maxOutgoing: 1 };
    case "Output":
      return { maxIncoming: 1, maxOutgoing: 0 };
    default:
      return { maxIncoming: 1, maxOutgoing: 1 };
  }
}

function formatArityExpectation(count: number, direction: "incoming" | "outgoing"): string {
  return `exactly ${count} ${direction} connection${count === 1 ? "" : "s"}`;
}

function topologicalNodes(graph: ModelGraph): ModelGraph["nodes"] {
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

  const queue = graph.nodes.filter((node) => (indegree.get(node.id) ?? 0) === 0).map((node) => node.id);
  const orderedIds: string[] = [];

  while (queue.length > 0) {
    const next = queue.shift()!;
    orderedIds.push(next);

    for (const target of outgoing.get(next) ?? []) {
      const nextDegree = (indegree.get(target) ?? 0) - 1;
      indegree.set(target, nextDegree);
      if (nextDegree === 0) {
        queue.push(target);
      }
    }
  }

  return orderedIds.map((id) => graph.nodes.find((node) => node.id === id)!).filter(Boolean);
}

function validateStageOrdering(nodes: ModelGraph["nodes"]): ValidationIssue[] {
  const issues: ValidationIssue[] = [];

  let seenEmbedding = false;
  let seenHiddenStage = false;
  let seenOutputStage = false;

  for (const node of nodes) {
    const type = node.type;

    if (isEmbeddingStage(type)) {
      if (seenHiddenStage || seenOutputStage) {
        issues.push(
          createValidationIssue(
            "embedding_stage_order",
            "Embedding stages must appear before transformer, feedforward, normalization, and output stages."
          )
        );
        break;
      }
      seenEmbedding = true;
      continue;
    }

    if (isHiddenStage(type)) {
      if (!seenEmbedding) {
        issues.push(
          createValidationIssue(
            "hidden_before_embedding",
            "Transformer, feedforward, and normalization stages must come after an embedding stage."
          )
        );
        break;
      }
      if (seenOutputStage) {
        issues.push(
          createValidationIssue(
            "hidden_after_output",
            "Transformer, feedforward, and normalization stages cannot appear after output stages."
          )
        );
        break;
      }
      seenHiddenStage = true;
      continue;
    }

    if (isOutputStage(type)) {
      if (!seenEmbedding) {
        issues.push(
          createValidationIssue(
            "output_before_embedding",
            "Output stages must come after an embedding stage."
          )
        );
        break;
      }
      seenOutputStage = true;
    }
  }

  return issues;
}

function isEmbeddingStage(type: ModelGraph["nodes"][number]["type"]): boolean {
  return type === "Embedding" || type === "GPT2TokenEmbedding" || type === "LlamaTokenEmbedding" || type === "GPT2PositionEmbedding";
}

function isHiddenStage(type: ModelGraph["nodes"][number]["type"]): boolean {
  return (
    type === "Add" ||
    type === "Dropout" ||
    type === "TransformerBlock" ||
    type === "GPT2Block" ||
    type === "LlamaBlock" ||
    type === "MoE" ||
    type === "MLP" ||
    type === "LayerNorm" ||
    type === "GPT2FinalLayerNorm" ||
    type === "LlamaFinalRMSNorm"
  );
}

function isOutputStage(type: ModelGraph["nodes"][number]["type"]): boolean {
  return type === "GPT2LMHead" || type === "LlamaLMHead" || type === "Softmax" || type === "Output";
}

function hasCycle(graph: ModelGraph): boolean {
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

  const queue = graph.nodes.filter((node) => (indegree.get(node.id) ?? 0) === 0).map((node) => node.id);
  let visited = 0;

  while (queue.length > 0) {
    const next = queue.shift()!;
    visited += 1;

    for (const target of outgoing.get(next) ?? []) {
      const nextDegree = (indegree.get(target) ?? 0) - 1;
      indegree.set(target, nextDegree);
      if (nextDegree === 0) {
        queue.push(target);
      }
    }
  }

  return visited !== graph.nodes.length;
}
