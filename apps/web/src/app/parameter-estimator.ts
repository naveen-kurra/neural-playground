import type { BlockNode, ModelGraph } from "@neural-playground/block-schema";
import {
  mapModelGraphToGPT2Ir,
  mapModelGraphToHybridIr,
  mapModelGraphToLlamaIr,
  type GPT2ArchitectureSpec,
  type HybridDecoderArchitectureSpec,
  type LlamaArchitectureSpec
} from "@neural-playground/ir-schema";
import {
  countEmbedding,
  countGpt2Attention,
  countGpt2Mlp,
  countLayerNorm,
  countLlamaAttention,
  countLlamaMlp,
  countTopKExperts,
  countTopKRouter
} from "./parameter-estimator-helpers";

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
        parameterCount += countEmbedding(vocabSize, hiddenSize);
        break;
      }
      case "GPT2PositionEmbedding": {
        const seqLen = num(node.config.sequenceLength, 1024);
        const dim = num(node.config.embeddingDim, hiddenSize);
        parameterCount += countEmbedding(seqLen, dim);
        break;
      }
      case "TransformerBlock": {
        const dModel = num(node.config.dModel, hiddenSize);
        parameterCount += countGpt2Attention(dModel);
        parameterCount += countLayerNorm(dModel);
        hiddenSize = dModel;
        break;
      }
      case "MoE": {
        const expertHidden = num(node.config.expertHidden, hiddenSize * 4);
        const numExperts = num(node.config.numExperts, 8);
        parameterCount += countTopKRouter(hiddenSize, numExperts);
        parameterCount += countTopKExperts(hiddenSize, expertHidden, numExperts, "gpt2");
        break;
      }
      case "GPT2Block": {
        const dModel = num(node.config.dModel, hiddenSize);
        const ffnHidden = num(node.config.ffnHidden, dModel * 4);
        parameterCount += countGpt2Attention(dModel);
        if (String(node.config.feedforwardType ?? "mlp") === "moe") {
          const expertHidden = num(node.config.expertHidden, ffnHidden);
          const numExperts = num(node.config.numExperts, 8);
          parameterCount += countTopKRouter(dModel, numExperts);
          parameterCount += countTopKExperts(dModel, expertHidden, numExperts, "gpt2");
        } else {
          parameterCount += countGpt2Mlp(dModel, ffnHidden);
        }
        parameterCount += countLayerNorm(dModel) * 2;
        hiddenSize = dModel;
        break;
      }
      case "LlamaBlock": {
        const dModel = num(node.config.dModel, hiddenSize);
        const numHeads = num(node.config.numHeads, 32);
        const numKvHeads = num(node.config.numKeyValueHeads, numHeads);
        const headDim = num(node.config.headDim, Math.floor(dModel / Math.max(1, numHeads)));
        const feedforwardType = String(node.config.feedforwardType ?? "mlp");
        const ffnHidden = num(node.config.ffnHidden, dModel * 4);
        parameterCount += countLlamaAttention(dModel, numHeads, numKvHeads, headDim, Boolean(node.config.attentionBias ?? false));
        if (feedforwardType === "moe") {
          const expertHidden = num(node.config.expertHidden, ffnHidden);
          const numExperts = num(node.config.numExperts, 8);
          parameterCount += countTopKRouter(dModel, numExperts, false);
          parameterCount += countTopKExperts(dModel, expertHidden, numExperts, "llama");
        } else {
          parameterCount += countLlamaMlp(dModel, ffnHidden, Boolean(node.config.mlpBias ?? false));
        }
        parameterCount += countLayerNorm(dModel, false) * 2;
        hiddenSize = dModel;
        break;
      }
      case "MLP": {
        const ffnHidden = num(node.config.hiddenDim, hiddenSize * 4);
        parameterCount += countGpt2Mlp(hiddenSize, ffnHidden);
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

  const tokenEmbedding = countEmbedding(vocab, d);
  const positionEmbedding = countEmbedding(pos, d);
  const blockAttention = countGpt2Attention(d);
  const blockMlp = countGpt2Mlp(d, ffn);
  const blockNorms = countLayerNorm(d) * 2;
  const finalNorm = countLayerNorm(d);
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

  const tokenEmbedding = countEmbedding(vocab, d);
  const attention = countLlamaAttention(d, heads, kvHeads, headDim, spec.config.attentionBias);
  const mlp = countLlamaMlp(d, ffn, spec.config.mlpBias);
  const norms = countLayerNorm(d, false) * 2;
  const finalNorm = countLayerNorm(d, false);
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
      count += countGpt2Attention(block.hiddenSize);
      if (block.feedforwardType === "moe") {
        count += countTopKRouter(block.hiddenSize, block.numExperts);
        count += countTopKExperts(block.hiddenSize, block.expertHidden, block.numExperts, "gpt2");
      } else {
        count += countGpt2Mlp(block.hiddenSize, block.intermediateSize);
      }
      count += countLayerNorm(block.hiddenSize) * 2;
    } else {
      count += countLlamaAttention(block.hiddenSize, block.numHeads, block.numKeyValueHeads, block.headDim, block.attentionBias);
      if (block.feedforwardType === "moe") {
        count += countTopKRouter(block.hiddenSize, block.numExperts, false);
        count += countTopKExperts(block.hiddenSize, block.expertHidden, block.numExperts, "llama");
      } else {
        count += countLlamaMlp(block.hiddenSize, block.intermediateSize, block.mlpBias);
      }
      count += countLayerNorm(block.hiddenSize, false) * 2;
    }
  }

  count += spec.operators.finalNorm.family === "gpt2" ? countLayerNorm(d) : countLayerNorm(d, false);

  if (!spec.config.tieWordEmbeddings) {
    count += d * vocab;
  }

  return Math.max(0, Math.floor(count));
}

export function formatParameterCount(graph: ModelGraph): string {
  const count = estimateParameterCount(graph);
  return `${formatCompact(count)} params`;
}
