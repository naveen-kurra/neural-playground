import { getBlockDefinition, type ModelGraph } from "@neural-playground/block-schema";

export type ValidationIssue = {
  code: string;
  message: string;
  nodeId?: string;
  edgeId?: string;
};

export function createValidationIssue(code: string, message: string): ValidationIssue {
  return { code, message };
}

export function validateGraph(graph: ModelGraph): ValidationIssue[] {
  const issues: ValidationIssue[] = [];
  const nodeIds = new Set(graph.nodes.map((node) => node.id));

  const inputCount = graph.nodes.filter((node) => node.type === "Input").length;
  const outputCount = graph.nodes.filter((node) => node.type === "Output").length;

  if (inputCount === 0) {
    issues.push(createValidationIssue("missing_input", "Graph needs at least one Input node."));
  }

  if (outputCount === 0) {
    issues.push(createValidationIssue("missing_output", "Graph needs at least one Output node."));
  }

  for (const edge of graph.edges) {
    if (!nodeIds.has(edge.source) || !nodeIds.has(edge.target)) {
      issues.push({
        code: "dangling_edge",
        message: `Edge ${edge.id} references a missing node.`,
        edgeId: edge.id
      });
      continue;
    }

    const sourceNode = graph.nodes.find((node) => node.id === edge.source);
    const targetNode = graph.nodes.find((node) => node.id === edge.target);
    if (!sourceNode || !targetNode) {
      continue;
    }

    const sourceDef = getBlockDefinition(sourceNode.type);
    const targetDef = getBlockDefinition(targetNode.type);
    const compatible = sourceDef.outputs.some((shape) => targetDef.inputs.includes(shape));
    if (!compatible) {
      issues.push({
        code: "shape_mismatch",
        message: `${sourceNode.type} cannot connect to ${targetNode.type} with the current block signatures.`,
        edgeId: edge.id
      });
    }
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
  }

  return issues;
}
