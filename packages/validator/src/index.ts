import type { ModelGraph } from "@neural-playground/block-schema";

import { validateConfigCompatibility } from "./config-compatibility";
import { validateEdge } from "./edge-validation";
import { inferSequenceDimensions } from "./inference";
import { createValidationIssue, type ValidationIssue, type ValidationMode } from "./issues";
import { validateNodeConfig } from "./node-validation";
import { validateTopology } from "./topology";

export * from "./issues";
export * from "./inference";
export * from "./node-validation";
export * from "./edge-validation";
export * from "./topology";
export * from "./config-compatibility";

export function validateGraph(graph: ModelGraph, mode: ValidationMode = "playground-valid"): ValidationIssue[] {
  const issues: ValidationIssue[] = [];
  const nodeIds = new Set(graph.nodes.map((node) => node.id));
  const inferredSequenceDims = inferSequenceDimensions(graph);

  issues.push(...validateTopology(graph, mode));
  issues.push(...validateConfigCompatibility(graph));

  const inputCount = graph.nodes.filter((node) => node.type === "Input").length;
  const outputCount = graph.nodes.filter((node) => node.type === "Output").length;

  if (inputCount === 0) {
    issues.push(createValidationIssue("missing_input", "Graph needs at least one Input node."));
  }

  if (outputCount === 0) {
    issues.push(createValidationIssue("missing_output", "Graph needs at least one Output node."));
  }

  for (const edge of graph.edges) {
    issues.push(...validateEdge(graph, edge, inferredSequenceDims, nodeIds));
  }

  for (const node of graph.nodes) {
    const incoming = graph.edges.filter((edge) => edge.target === node.id);
    const outgoing = graph.edges.filter((edge) => edge.source === node.id);

    if (node.type !== "Input" && incoming.length === 0) {
      issues.push({
        code: "missing_incoming_edge",
        message: `${node.type} has no incoming connection.`,
        nodeId: node.id
      });
    }

    if (node.type !== "Output" && outgoing.length === 0) {
      issues.push({
        code: "missing_outgoing_edge",
        message: `${node.type} has no outgoing connection.`,
        nodeId: node.id
      });
    }

    issues.push(...validateNodeConfig(node, inferredSequenceDims));
  }

  return issues;
}
