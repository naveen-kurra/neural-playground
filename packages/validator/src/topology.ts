import type { ModelGraph } from "@neural-playground/block-schema";

import { createValidationIssue, type ValidationIssue, type ValidationMode } from "./issues";

export function validateTopology(graph: ModelGraph, mode: ValidationMode): ValidationIssue[] {
  const issues: ValidationIssue[] = [];

  if (hasCycle(graph)) {
    issues.push(createValidationIssue("graph_cycle", "Graph contains a cycle, which is not supported."));
  }

  if (mode === "playground-valid") {
    return issues;
  }

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
