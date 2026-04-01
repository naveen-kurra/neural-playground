import { getBlockDefinition, type ModelGraph, type ShapeContract } from "@neural-playground/block-schema";

import { getInputSequenceDim, getOutputSequenceDim } from "./inference";
import type { ValidationIssue } from "./issues";

export function validateEdge(
  graph: ModelGraph,
  edge: ModelGraph["edges"][number],
  inferredSequenceDims: Map<string, number | null>,
  nodeIds: Set<string>
): ValidationIssue[] {
  const issues: ValidationIssue[] = [];

  if (!nodeIds.has(edge.source) || !nodeIds.has(edge.target)) {
    issues.push({
      code: "dangling_edge",
      message: `Edge ${edge.id} references a missing node.`,
      edgeId: edge.id
    });
    return issues;
  }

  const sourceNode = graph.nodes.find((node) => node.id === edge.source);
  const targetNode = graph.nodes.find((node) => node.id === edge.target);
  if (!sourceNode || !targetNode) {
    return issues;
  }

  const sourceDef = getBlockDefinition(sourceNode.type);
  const targetDef = getBlockDefinition(targetNode.type);
  const compatible = contractsAreCompatible(sourceDef.outputContracts, targetDef.inputContracts);
  if (!compatible) {
    issues.push({
      code: "shape_mismatch",
      message: `${sourceNode.type} cannot connect to ${targetNode.type} with the current block signatures.`,
      edgeId: edge.id
    });
    return issues;
  }

  const sourceSequenceDim = getOutputSequenceDim(sourceNode, inferredSequenceDims);
  const targetSequenceDim = getInputSequenceDim(targetNode, inferredSequenceDims);
  if (
    sourceSequenceDim !== null &&
    targetSequenceDim !== null &&
    sourceSequenceDim !== targetSequenceDim
  ) {
    issues.push({
      code: "dimension_mismatch",
      message: `${sourceNode.type} outputs sequence dim ${sourceSequenceDim}, but ${targetNode.type} expects ${targetSequenceDim}.`,
      edgeId: edge.id
    });
  }

  return issues;
}

function contractsAreCompatible(outputs: ShapeContract[], inputs: ShapeContract[]): boolean {
  return outputs.some((output) =>
    inputs.some((input) => {
      if (output.kind !== input.kind) {
        return false;
      }
      if (output.dims.length !== input.dims.length) {
        return false;
      }
      return output.dims.every((dim, index) => {
        const expected = input.dims[index];
        return expected === dim || expected === "unknown" || dim === "unknown";
      });
    })
  );
}
