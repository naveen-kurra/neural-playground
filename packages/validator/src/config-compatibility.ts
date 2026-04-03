import { getBlockDefinition, type ModelGraph } from "@neural-playground/block-schema";
import { numberConfig } from "./inference";
import type { ValidationIssue } from "./issues";

type Node = ModelGraph["nodes"][number];

function getNodeVocabSize(node: Node): number | null {
  const def = getBlockDefinition(node.type);
  if (!def.vocabSizeField) return null;
  return numberConfig((node.config as Record<string, unknown>)[def.vocabSizeField]);
}

function getNodeSequenceDim(node: Node): number | null {
  const def = getBlockDefinition(node.type);
  if (!def.sequenceDimField) return null;
  return numberConfig((node.config as Record<string, unknown>)[def.sequenceDimField]);
}

export function validateConfigCompatibility(graph: ModelGraph): ValidationIssue[] {
  const issues: ValidationIssue[] = [];
  const nodeMap = new Map(graph.nodes.map((n) => [n.id, n]));

  // --- Edge-level: vocab size mismatch between connected nodes ---
  for (const edge of graph.edges) {
    const source = nodeMap.get(edge.source);
    const target = nodeMap.get(edge.target);
    if (!source || !target) continue;

    const sourceVocab = getNodeVocabSize(source);
    const targetVocab = getNodeVocabSize(target);
    if (sourceVocab !== null && targetVocab !== null && sourceVocab !== targetVocab) {
      issues.push({
        code: "config_vocab_mismatch",
        message: `${source.type} (vocabSize ${sourceVocab}) connects to ${target.type} (vocabSize ${targetVocab}) — vocab sizes must match.`,
        edgeId: edge.id
      });
    }
  }

  // --- Node-level: merge nodes must receive inputs with matching dimensions ---
  for (const node of graph.nodes) {
    const def = getBlockDefinition(node.type);
    const isMergeNode = def.inputContracts.length > 0 &&
      def.inputContracts.every((c) => c.kind === "sequence") &&
      !def.sequenceDimField;
    const incoming = graph.edges.filter((e) => e.target === node.id);

    if (node.type === "Add" && incoming.length !== 2) {
      issues.push({
        code: "add_input_arity",
        message: "Add requires exactly two incoming sequence connections.",
        nodeId: node.id
      });
    }

    if (isMergeNode) {
      const dims = incoming
        .map((e) => nodeMap.get(e.source))
        .filter((n): n is Node => n !== undefined)
        .map((n) => getNodeSequenceDim(n))
        .filter((d): d is number => d !== null);

      if (dims.length > 1) {
        const first = dims[0]!;
        if (!dims.every((d) => d === first)) {
          issues.push({
            code: "config_dim_mismatch",
            message: `${node.type} inputs have mismatched dimensions (${dims.join(", ")}). All inputs must have the same dimension.`,
            nodeId: node.id
          });
        }
      }
    }
  }

  return issues;
}
