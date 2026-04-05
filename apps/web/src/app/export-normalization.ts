import type { BlockEdge, BlockNode, ModelGraph } from "@neural-playground/block-schema";

type SafeGraph = { ok: true; value: ModelGraph } | { ok: false; error: string };

function topologicalNodes(graph: ModelGraph): BlockNode[] {
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

  if (orderedIds.length !== graph.nodes.length) {
    throw new Error("Mixed-family export requires an acyclic graph.");
  }

  return orderedIds.map((id) => graph.nodes.find((node) => node.id === id)!).filter(Boolean);
}

function makeNode(type: BlockNode["type"], index: number, config: BlockNode["config"]): BlockNode {
  return {
    id: `export-${type}-${index}`,
    type,
    position: { x: 40 + index * 220, y: 96 },
    config
  };
}

export function normalizeGraphForGenericExport(graph: ModelGraph): SafeGraph {
  try {
    const ordered = topologicalNodes(graph);
    const inputNode = ordered.find((node) => node.type === "Input");
    const outputNode = ordered.find((node) => node.type === "Output");

    if (!inputNode || !outputNode) {
      throw new Error("Mixed-family export requires exactly one Input and one Output.");
    }

    const normalizedNodes: BlockNode[] = [];
    let index = 0;
    let embeddingAdded = false;

    normalizedNodes.push(
      makeNode("Input", index++, {
        sequenceLength: Number(inputNode.config.sequenceLength ?? 1024)
      })
    );

    for (const node of ordered) {
      if (node.type === "Input" || node.type === "GPT2PositionEmbedding" || node.type === "Add" || node.type === "Dropout") {
        continue;
      }

      if (node.type === "Embedding" || node.type === "GPT2TokenEmbedding" || node.type === "LlamaTokenEmbedding" || node.type === "MistralTokenEmbedding" || node.type === "Phi3TokenEmbedding" || node.type === "Gemma4TokenEmbedding") {
        if (embeddingAdded) {
          continue;
        }
        normalizedNodes.push(
          makeNode("Embedding", index++, {
            vocabSize: Number(node.config.vocabSize ?? 32000),
            embeddingDim: Number(node.config.embeddingDim ?? 768)
          })
        );
        embeddingAdded = true;
        continue;
      }

      if (node.type === "TransformerBlock") {
        normalizedNodes.push(
          makeNode("TransformerBlock", index++, {
            dModel: Number(node.config.dModel ?? 768),
            numHeads: Number(node.config.numHeads ?? 12),
            ffnHidden: Number(node.config.ffnHidden ?? 3072),
            activation: String(node.config.activation ?? graph.training.activation ?? "GELU"),
            dropout: Number(node.config.dropout ?? 0.1)
          })
        );
        continue;
      }

      if (node.type === "GPT2Block") {
        normalizedNodes.push(
          makeNode("TransformerBlock", index++, {
            dModel: Number(node.config.dModel ?? 768),
            numHeads: Number(node.config.numHeads ?? 12),
            ffnHidden: Number(node.config.ffnHidden ?? 3072),
            activation: "GELU",
            dropout: Number(node.config.dropout ?? 0.1)
          })
        );
        continue;
      }

      if (node.type === "LlamaBlock" || node.type === "MistralBlock" || node.type === "Phi3Block" || node.type === "Gemma4Block") {
        normalizedNodes.push(
          makeNode("TransformerBlock", index++, {
            dModel: Number(node.config.dModel ?? (node.type === "MistralBlock" ? 4096 : node.type === "Phi3Block" ? 3072 : node.type === "Gemma4Block" ? 5376 : 4096)),
            numHeads: Number(node.config.numHeads ?? 32),
            ffnHidden: Number(node.config.ffnHidden ?? (node.type === "MistralBlock" ? 14336 : node.type === "Phi3Block" ? 8192 : node.type === "Gemma4Block" ? 21504 : 11008)),
            activation: node.type === "Gemma4Block" ? "GELU" : "SiLU",
            dropout: Number(node.config.dropout ?? 0)
          })
        );
        continue;
      }

      if (node.type === "MLP") {
        normalizedNodes.push(
          makeNode("MLP", index++, {
            hiddenDim: Number(node.config.hiddenDim ?? 3072),
            activation: String(node.config.activation ?? graph.training.activation ?? "GELU")
          })
        );
        continue;
      }

      if (node.type === "MoE") {
        normalizedNodes.push(
          makeNode("MoE", index++, {
            numExperts: Number(node.config.numExperts ?? 8),
            topK: Number(node.config.topK ?? 2),
            expertHidden: Number(node.config.expertHidden ?? 3072),
            activation: String(node.config.activation ?? graph.training.activation ?? "GELU")
          })
        );
        continue;
      }

      if (node.type === "LayerNorm" || node.type === "GPT2FinalLayerNorm" || node.type === "LlamaFinalRMSNorm" || node.type === "MistralFinalRMSNorm" || node.type === "Phi3FinalRMSNorm" || node.type === "Gemma4FinalRMSNorm") {
        normalizedNodes.push(
          makeNode("LayerNorm", index++, {
            epsilon: Number(node.config.epsilon ?? 0.00001)
          })
        );
        continue;
      }

      if (node.type === "Softmax") {
        normalizedNodes.push(
          makeNode("Softmax", index++, {
            axis: Number(node.config.axis ?? -1)
          })
        );
        continue;
      }

      if (node.type === "GPT2LMHead" || node.type === "LlamaLMHead" || node.type === "MistralLMHead" || node.type === "Phi3LMHead" || node.type === "Gemma4LMHead") {
        continue;
      }

      if (node.type === "Output") {
        continue;
      }
    }

    if (!embeddingAdded) {
      throw new Error("Mixed-family export requires one embedding stage.");
    }

    normalizedNodes.push(
      makeNode("Output", index++, {
        headType: String(outputNode.config.headType ?? "LanguageModel")
      })
    );

    const normalizedEdges: BlockEdge[] = normalizedNodes.slice(1).map((node, nodeIndex) => ({
      id: `export-edge-${nodeIndex + 1}`,
      source: normalizedNodes[nodeIndex]!.id,
      target: node.id
    }));

    return {
      ok: true,
      value: {
        nodes: normalizedNodes,
        edges: normalizedEdges,
        training: graph.training
      }
    };
  } catch (error) {
    return {
      ok: false,
      error: error instanceof Error ? error.message : "Failed to normalize graph for generic export."
    };
  }
}
