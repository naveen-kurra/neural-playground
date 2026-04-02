import type { BlockNode, ModelGraph } from "@neural-playground/block-schema";
import {
  mapModelGraphToGPT2Ir,
  mapModelGraphToHybridIr,
  mapModelGraphToLlamaIr,
  type GPT2ArchitectureSpec,
  type HybridDecoderArchitectureSpec,
  type LlamaArchitectureSpec
} from "@neural-playground/ir-schema";

function num(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

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

  return orderedIds.map((id) => graph.nodes.find((node) => node.id === id)!).filter(Boolean);
}

function formatCompact(value: number): string {
  if (value >= 1_000_000_000) {
    return `${(value / 1_000_000_000).toFixed(2)}B`;
  }
  if (value >= 1_000_000) {
    return `${(value / 1_000_000).toFixed(2)}M`;
  }
  if (value >= 1_000) {
    return `${(value / 1_000).toFixed(1)}K`;
  }
  return `${value}`;
}

export function estimateParameterCount(graph: ModelGraph): number {
  try {
    return estimateGPT2SpecParameters(mapModelGraphToGPT2Ir(graph));
  } catch {}

  try {
    return estimateLlamaSpecParameters(mapModelGraphToLlamaIr(graph));
  } catch {}

  try {
    return estimateHybridSpecParameters(mapModelGraphToHybridIr(graph));
  } catch {}

  return estimateGenericGraphParameters(graph);
}

function estimateGenericGraphParameters(graph: ModelGraph): number {
  const ordered = topologicalNodes(graph);
  let hiddenSize = 768;
  let vocabSize = 32000;
  let parameterCount = 0;
  let finalNormAccounted = false;

  for (const node of ordered) {
    switch (node.type) {
      case "Embedding":
      case "GPT2TokenEmbedding":
      case "LlamaTokenEmbedding": {
        vocabSize = num(node.config.vocabSize, vocabSize);
        hiddenSize = num(node.config.embeddingDim, hiddenSize);
        parameterCount += vocabSize * hiddenSize;
        break;
      }
      case "GPT2PositionEmbedding": {
        const seqLen = num(node.config.sequenceLength, 1024);
        const dim = num(node.config.embeddingDim, hiddenSize);
        parameterCount += seqLen * dim;
        break;
      }
      case "TransformerBlock": {
        const dModel = num(node.config.dModel, hiddenSize);
        parameterCount += 4 * dModel * dModel + 4 * dModel;
        parameterCount += 2 * dModel;
        hiddenSize = dModel;
        break;
      }
      case "MoE": {
        const expertHidden = num(node.config.expertHidden, hiddenSize * 4);
        const numExperts = num(node.config.numExperts, 8);
        parameterCount += hiddenSize * numExperts + numExperts;
        parameterCount += numExperts * (2 * hiddenSize * expertHidden + (expertHidden + hiddenSize));
        break;
      }
      case "GPT2Block": {
        const dModel = num(node.config.dModel, hiddenSize);
        const ffnHidden = num(node.config.ffnHidden, dModel * 4);
        parameterCount += 4 * dModel * dModel + 4 * dModel;
        parameterCount += 2 * dModel * ffnHidden + (ffnHidden + dModel);
        parameterCount += 4 * dModel;
        hiddenSize = dModel;
        break;
      }
      case "LlamaBlock": {
        const dModel = num(node.config.dModel, hiddenSize);
        const numHeads = num(node.config.numHeads, 32);
        const numKvHeads = num(node.config.numKeyValueHeads, numHeads);
        const headDim = num(node.config.headDim, Math.floor(dModel / Math.max(1, numHeads)));
        const ffnHidden = num(node.config.ffnHidden, dModel * 4);
        parameterCount += dModel * (numHeads * headDim);
        parameterCount += dModel * (numKvHeads * headDim) * 2;
        parameterCount += dModel * (numHeads * headDim);
        parameterCount += dModel * ffnHidden * 2 + ffnHidden * dModel;
        parameterCount += 2 * dModel;
        hiddenSize = dModel;
        break;
      }
      case "MLP": {
        const ffnHidden = num(node.config.hiddenDim, hiddenSize * 4);
        parameterCount += 2 * hiddenSize * ffnHidden + (ffnHidden + hiddenSize);
        break;
      }
      case "LayerNorm":
      case "GPT2FinalLayerNorm":
      case "LlamaFinalRMSNorm": {
        parameterCount += 2 * hiddenSize;
        finalNormAccounted = true;
        break;
      }
      case "GPT2LMHead":
      case "LlamaLMHead": {
        const tiedWeights = Boolean(node.config.tiedWeights ?? false);
        const lmVocabSize = num(node.config.vocabSize, vocabSize);
        if (!tiedWeights) {
          parameterCount += hiddenSize * lmVocabSize;
        }
        break;
      }
      case "Output":
        if (String(node.config.headType ?? "LanguageModel") === "LanguageModel") {
          parameterCount += hiddenSize * vocabSize;
        }
        break;
      default:
        break;
    }
  }

  if (!finalNormAccounted && hiddenSize > 0) {
    parameterCount += 0;
  }

  return Math.max(0, Math.floor(parameterCount));
}

function estimateGPT2SpecParameters(spec: GPT2ArchitectureSpec): number {
  const d = spec.config.hiddenSize;
  const vocab = spec.config.vocabSize;
  const pos = spec.config.maxPositionEmbeddings;
  const layers = spec.config.numHiddenLayers;
  const ffn = spec.config.intermediateSize;

  const tokenEmbedding = vocab * d;
  const positionEmbedding = pos * d;
  const blockAttention = 4 * d * d + 4 * d;
  const blockMlp = 2 * d * ffn + ffn + d;
  const blockNorms = 4 * d;
  const finalNorm = 2 * d;
  const lmHead = spec.config.tieWordEmbeddings ? 0 : d * vocab;

  return tokenEmbedding + positionEmbedding + layers * (blockAttention + blockMlp + blockNorms) + finalNorm + lmHead;
}

function estimateLlamaSpecParameters(spec: LlamaArchitectureSpec): number {
  const d = spec.config.hiddenSize;
  const vocab = spec.config.vocabSize;
  const layers = spec.config.numHiddenLayers;
  const ffn = spec.config.intermediateSize;
  const heads = spec.config.numAttentionHeads;
  const kvHeads = spec.config.numKeyValueHeads;
  const headDim = spec.config.headDim;

  const tokenEmbedding = vocab * d;
  const qProj = d * (heads * headDim);
  const kProj = d * (kvHeads * headDim);
  const vProj = d * (kvHeads * headDim);
  const oProj = (heads * headDim) * d;
  const attention = qProj + kProj + vProj + oProj;
  const mlp = d * ffn * 2 + ffn * d;
  const norms = 2 * d;
  const finalNorm = d;
  const lmHead = spec.config.tieWordEmbeddings ? 0 : d * vocab;

  return tokenEmbedding + layers * (attention + mlp + norms) + finalNorm + lmHead;
}

function estimateHybridSpecParameters(spec: HybridDecoderArchitectureSpec): number {
  const d = spec.config.hiddenSize;
  const vocab = spec.config.vocabSize;
  let count = 0;

  if (spec.operators.embedding.family === "gpt2") {
    count += vocab * d;
    count += spec.operators.embedding.maxPositionEmbeddings * d;
  } else {
    count += vocab * d;
  }

  for (const block of spec.operators.blocks) {
    if (block.family === "gpt2") {
      count += 4 * block.hiddenSize * block.hiddenSize + 4 * block.hiddenSize;
      count += 2 * block.hiddenSize * block.intermediateSize + block.intermediateSize + block.hiddenSize;
      count += 4 * block.hiddenSize;
    } else {
      const qProj = block.hiddenSize * (block.numHeads * block.headDim);
      const kProj = block.hiddenSize * (block.numKeyValueHeads * block.headDim);
      const vProj = block.hiddenSize * (block.numKeyValueHeads * block.headDim);
      const oProj = (block.numHeads * block.headDim) * block.hiddenSize;
      count += qProj + kProj + vProj + oProj;
      count += block.hiddenSize * block.intermediateSize * 2 + block.intermediateSize * block.hiddenSize;
      count += 2 * block.hiddenSize;
    }
  }

  count += spec.operators.finalNorm.family === "gpt2" ? 2 * d : d;

  if (!spec.config.tieWordEmbeddings) {
    count += d * vocab;
  }

  return Math.max(0, Math.floor(count));
}

export function formatParameterCount(graph: ModelGraph): string {
  const count = estimateParameterCount(graph);
  return `${formatCompact(count)} params`;
}
