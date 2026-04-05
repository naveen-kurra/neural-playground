import type { ModelGraph } from "@neural-playground/block-schema";
import { isCustomizedGpt2Block, isCustomizedLlamaBlock, isCustomizedPhi3Block } from "./customization";
import { buildGPT2ArchitectureSpec, type GPT2ConfigInput } from "./gpt2";
import { buildGemma4ArchitectureSpec, type Gemma4ConfigInput } from "./gemma4";
import { buildLlamaArchitectureSpec, type LlamaConfigInput } from "./llama";
import { buildPhi3ArchitectureSpec, type Phi3ConfigInput } from "./phi3";
import type { Gemma4ArchitectureSpec, GPT2ArchitectureSpec, HybridDecoderArchitectureSpec, LlamaArchitectureSpec, Phi3ArchitectureSpec } from "./types";

function topologicalNodes(graph: ModelGraph) {
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
    throw new Error("GPT-2 IR mapping requires an acyclic graph.");
  }

  return orderedIds.map((id) => graph.nodes.find((node) => node.id === id)!).filter(Boolean);
}

function getIncoming(graph: ModelGraph, nodeId: string): string[] {
  return graph.edges.filter((edge) => edge.target === nodeId).map((edge) => edge.source);
}

export function mapModelGraphToGPT2Ir(graph: ModelGraph): GPT2ArchitectureSpec {
  const ordered = topologicalNodes(graph);
  const exactTypes = ["Input", "GPT2TokenEmbedding", "GPT2PositionEmbedding", "Add", "Dropout", "GPT2Block", "GPT2FinalLayerNorm", "GPT2LMHead", "Output"];
  const genericTypes = ["Input", "Embedding", "TransformerBlock", "Output"];
  const hasExactProjection = ordered.some((node) => node.type === "GPT2TokenEmbedding");
  const allowedTypes = hasExactProjection ? exactTypes : genericTypes;
  const invalidTypes = ordered.filter((node) => !allowedTypes.includes(node.type));
  if (invalidTypes.length > 0) {
    throw new Error(`GPT-2 IR mapping only supports ${allowedTypes.join(", ")}. Found ${invalidTypes[0]!.type}.`);
  }

  if (hasExactProjection) {
    return mapExactGpt2ProjectionToIr(graph, ordered);
  }

  const inputNodes = ordered.filter((node) => node.type === "Input");
  const embeddingNodes = ordered.filter((node) => node.type === "Embedding");
  const outputNodes = ordered.filter((node) => node.type === "Output");
  const blockNodes = ordered.filter((node) => node.type === "TransformerBlock");

  if (inputNodes.length !== 1 || embeddingNodes.length !== 1 || outputNodes.length !== 1) {
    throw new Error("GPT-2 IR mapping requires exactly one Input, one Embedding, and one Output node.");
  }

  const output = outputNodes[0]!;
  if (String(output.config.headType ?? "LanguageModel") !== "LanguageModel") {
    throw new Error("GPT-2 IR mapping requires Output head type LanguageModel.");
  }

  const embedding = embeddingNodes[0]!;
  const firstBlock = blockNodes[0];
  const hiddenSize = Number(embedding.config.embeddingDim ?? firstBlock?.config.dModel ?? 768);
  const config: GPT2ConfigInput = {
    name: "GPT-2 (from canvas)",
    vocabSize: Number(embedding.config.vocabSize ?? 50257),
    maxPositionEmbeddings: Number(inputNodes[0]!.config.sequenceLength ?? 1024),
    hiddenSize,
    numHiddenLayers: blockNodes.length,
    numAttentionHeads: Number(firstBlock?.config.numHeads ?? 12),
    intermediateSize: Number(firstBlock?.config.ffnHidden ?? hiddenSize * 4),
    activationFunction: normalizeActivationName(graph.training.activation),
    attnDropout: Number(firstBlock?.config.dropout ?? 0.1),
    residDropout: Number(firstBlock?.config.dropout ?? 0.1),
    embdDropout: 0.1,
    layerNormEpsilon: 1e-5,
    scaleAttnWeights: true,
    scaleAttnByInverseLayerIdx: false,
    reorderAndUpcastAttn: false,
    tieWordEmbeddings: true
  };

  return buildGPT2ArchitectureSpec(config);
}

function mapExactGpt2ProjectionToIr(graph: ModelGraph, ordered: ModelGraph["nodes"]): GPT2ArchitectureSpec {
  const inputNode = ordered.find((node) => node.type === "Input");
  const tokenEmbedding = ordered.find((node) => node.type === "GPT2TokenEmbedding");
  const positionEmbedding = ordered.find((node) => node.type === "GPT2PositionEmbedding");
  const addNode = ordered.find((node) => node.type === "Add");
  const dropoutNode = ordered.find((node) => node.type === "Dropout");
  const finalNormNode = ordered.find((node) => node.type === "GPT2FinalLayerNorm");
  const lmHeadNode = ordered.find((node) => node.type === "GPT2LMHead");
  const outputNode = ordered.find((node) => node.type === "Output");
  const blocks = ordered.filter((node) => node.type === "GPT2Block");

  if (!inputNode || !tokenEmbedding || !positionEmbedding || !addNode || !dropoutNode || !finalNormNode || !lmHeadNode || !outputNode) {
    throw new Error("Exact GPT-2 projection is incomplete. Expected Input, GPT2TokenEmbedding, GPT2PositionEmbedding, Add, Dropout, GPT2Block*, GPT2FinalLayerNorm, GPT2LMHead, and Output.");
  }

  if (String(outputNode.config.headType ?? "LanguageModel") !== "LanguageModel") {
    throw new Error("GPT-2 IR mapping requires Output head type LanguageModel.");
  }

  const hiddenSize = Number(tokenEmbedding.config.embeddingDim ?? 768);
  const firstBlock = blocks[0];
  if (blocks.some(isCustomizedGpt2Block)) {
    throw new Error("Customized GPT-2 block internals are exported through the hybrid path.");
  }

  return buildGPT2ArchitectureSpec({
    name: "GPT-2 (exact canvas)",
    vocabSize: Number(tokenEmbedding.config.vocabSize ?? lmHeadNode.config.vocabSize ?? 50257),
    maxPositionEmbeddings: Number(positionEmbedding.config.sequenceLength ?? inputNode.config.sequenceLength ?? 1024),
    hiddenSize,
    numHiddenLayers: blocks.length,
    numAttentionHeads: Number(firstBlock?.config.numHeads ?? 12),
    intermediateSize: Number(firstBlock?.config.ffnHidden ?? hiddenSize * 4),
    activationFunction: String(firstBlock?.config.activation ?? normalizeActivationName(graph.training.activation)),
    attnDropout: Number(firstBlock?.config.dropout ?? dropoutNode.config.dropout ?? 0.1),
    residDropout: Number(firstBlock?.config.dropout ?? dropoutNode.config.dropout ?? 0.1),
    embdDropout: Number(dropoutNode.config.dropout ?? 0.1),
    layerNormEpsilon: Number(firstBlock?.config.layerNormEpsilon ?? finalNormNode.config.epsilon ?? 1e-5),
    scaleAttnWeights: Boolean(firstBlock?.config.scaleAttnWeights ?? true),
    scaleAttnByInverseLayerIdx: Boolean(firstBlock?.config.scaleAttnByInverseLayerIdx ?? false),
    reorderAndUpcastAttn: Boolean(firstBlock?.config.reorderAndUpcastAttn ?? false),
    tieWordEmbeddings: Boolean(lmHeadNode.config.tiedWeights ?? true)
  });
}

export function mapGPT2ConfigToIr(
  config: Record<string, unknown>,
  options: { modelId?: string; name?: string } = {}
): GPT2ArchitectureSpec {
  const hiddenSize = numberField(config.n_embd) ?? numberField(config.hidden_size) ?? 768;

  return buildGPT2ArchitectureSpec({
    name: options.name ?? "GPT-2",
    modelId: options.modelId,
    vocabSize: numberField(config.vocab_size) ?? 50257,
    maxPositionEmbeddings: numberField(config.n_positions) ?? numberField(config.n_ctx) ?? 1024,
    hiddenSize,
    numHiddenLayers: numberField(config.n_layer) ?? numberField(config.num_hidden_layers) ?? 12,
    numAttentionHeads: numberField(config.n_head) ?? numberField(config.num_attention_heads) ?? 12,
    intermediateSize: numberField(config.n_inner) ?? hiddenSize * 4,
    activationFunction: stringField(config.activation_function) ?? "gelu_new",
    embdDropout: numberField(config.embd_pdrop) ?? 0.1,
    attnDropout: numberField(config.attn_pdrop) ?? 0.1,
    residDropout: numberField(config.resid_pdrop) ?? 0.1,
    layerNormEpsilon: numberField(config.layer_norm_epsilon) ?? 1e-5,
    scaleAttnWeights: booleanField(config.scale_attn_weights) ?? true,
    scaleAttnByInverseLayerIdx: booleanField(config.scale_attn_by_inverse_layer_idx) ?? false,
    reorderAndUpcastAttn: booleanField(config.reorder_and_upcast_attn) ?? false,
    tieWordEmbeddings: booleanField(config.tie_word_embeddings) ?? true
  });
}

export function projectGPT2IrToModelGraph(spec: GPT2ArchitectureSpec): ModelGraph {
  const inputNode = {
    id: "Input-gpt2",
    type: "Input" as const,
    position: { x: 40, y: 92 },
    config: {
      sequenceLength: spec.config.maxPositionEmbeddings
    }
  };

  const embeddingNode = {
    id: "GPT2TokenEmbedding-gpt2",
    type: "GPT2TokenEmbedding" as const,
    position: { x: 280, y: 40 },
    config: {
      vocabSize: spec.config.vocabSize,
      embeddingDim: spec.config.hiddenSize
    }
  };

  const positionEmbeddingNode = {
    id: "GPT2PositionEmbedding-gpt2",
    type: "GPT2PositionEmbedding" as const,
    position: { x: 280, y: 168 },
    config: {
      sequenceLength: spec.config.maxPositionEmbeddings,
      embeddingDim: spec.config.hiddenSize
    }
  };

  const addNode = {
    id: "Add-gpt2",
    type: "Add" as const,
    position: { x: 540, y: 104 },
    config: {}
  };

  const dropoutNode = {
    id: "Dropout-gpt2",
    type: "Dropout" as const,
    position: { x: 760, y: 104 },
    config: {
      dropout: spec.config.embdDropout
    }
  };

  const blockNodes = Array.from({ length: spec.config.numHiddenLayers }, (_, index) => ({
    id: `GPT2Block-gpt2-${index}`,
    type: "GPT2Block" as const,
    position: { x: 1000 + index * 240, y: 104 + (index % 2) * 72 },
    config: {
      dModel: spec.config.hiddenSize,
      numHeads: spec.config.numAttentionHeads,
      ffnHidden: spec.config.intermediateSize,
      feedforwardType: "mlp",
      numExperts: 8,
      topK: 2,
      expertHidden: spec.config.intermediateSize,
      activation: spec.config.activationFunction,
      layerNormEpsilon: spec.config.layerNormEpsilon,
      dropout: spec.config.residDropout,
      scaleAttnWeights: spec.config.scaleAttnWeights,
      scaleAttnByInverseLayerIdx: spec.config.scaleAttnByInverseLayerIdx,
      reorderAndUpcastAttn: spec.config.reorderAndUpcastAttn
    }
  }));

  const finalNormNode = {
    id: "GPT2FinalLayerNorm-gpt2",
    type: "GPT2FinalLayerNorm" as const,
    position: { x: 1000 + spec.config.numHiddenLayers * 240, y: 104 },
    config: {
      epsilon: spec.config.layerNormEpsilon
    }
  };

  const lmHeadNode = {
    id: "GPT2LMHead-gpt2",
    type: "GPT2LMHead" as const,
    position: { x: 1240 + spec.config.numHiddenLayers * 240, y: 104 },
    config: {
      vocabSize: spec.config.vocabSize,
      tiedWeights: spec.config.tieWordEmbeddings
    }
  };

  const outputNode = {
    id: "Output-gpt2",
    type: "Output" as const,
    position: { x: 1480 + spec.config.numHiddenLayers * 240, y: 104 },
    config: {
      headType: "LanguageModel"
    }
  };

  const nodes = [inputNode, embeddingNode, positionEmbeddingNode, addNode, dropoutNode, ...blockNodes, finalNormNode, lmHeadNode, outputNode];
  const edges = [
    { id: "edge-gpt2-1", source: inputNode.id, target: embeddingNode.id },
    { id: "edge-gpt2-2", source: inputNode.id, target: positionEmbeddingNode.id },
    { id: "edge-gpt2-3", source: embeddingNode.id, target: addNode.id },
    { id: "edge-gpt2-4", source: positionEmbeddingNode.id, target: addNode.id },
    { id: "edge-gpt2-5", source: addNode.id, target: dropoutNode.id },
    ...blockNodes.map((node, index) => ({
      id: `edge-gpt2-block-${index + 1}`,
      source: index === 0 ? dropoutNode.id : blockNodes[index - 1]!.id,
      target: node.id
    })),
    {
      id: "edge-gpt2-final-ln",
      source: blockNodes.length > 0 ? blockNodes[blockNodes.length - 1]!.id : dropoutNode.id,
      target: finalNormNode.id
    },
    { id: "edge-gpt2-lm-head", source: finalNormNode.id, target: lmHeadNode.id },
    { id: "edge-gpt2-output", source: lmHeadNode.id, target: outputNode.id }
  ];

  return {
    nodes,
    edges,
    training: {
      optimizer: "AdamW",
      loss: "CrossEntropy",
      learningRate: 0.0003,
      activation: spec.config.activationFunction.toLowerCase().includes("gelu")
        ? "GELU"
        : spec.config.activationFunction.toLowerCase() === "relu"
          ? "ReLU"
          : spec.config.activationFunction.toLowerCase() === "silu"
            ? "SiLU"
            : "Custom",
      optimizerCustomName: "",
      lossCustomName: "",
      activationCustomName: spec.config.activationFunction.toLowerCase().includes("gelu")
        ? ""
        : spec.config.activationFunction.toLowerCase() === "relu" || spec.config.activationFunction.toLowerCase() === "silu"
          ? ""
          : spec.config.activationFunction
    }
  };
}

export function mapLlamaConfigToIr(
  config: Record<string, unknown>,
  options: { modelId?: string; name?: string } = {}
): LlamaArchitectureSpec {
  const hiddenSize = numberField(config.hidden_size) ?? 4096;
  const numAttentionHeads = numberField(config.num_attention_heads) ?? 32;
  const headDim = numberField(config.head_dim) ?? Math.floor(hiddenSize / numAttentionHeads);

  return buildLlamaArchitectureSpec({
    name: options.name ?? "LLaMA",
    modelId: options.modelId,
    vocabSize: numberField(config.vocab_size) ?? 32000,
    hiddenSize,
    intermediateSize: numberField(config.intermediate_size) ?? 11008,
    numHiddenLayers: numberField(config.num_hidden_layers) ?? 32,
    numAttentionHeads,
    numKeyValueHeads: numberField(config.num_key_value_heads) ?? numAttentionHeads,
    headDim,
    hiddenActivation: stringField(config.hidden_act) ?? "silu",
    maxPositionEmbeddings: numberField(config.max_position_embeddings) ?? 2048,
    rmsNormEpsilon: numberField(config.rms_norm_eps) ?? 1e-6,
    ropeTheta: numberField(config.rope_theta) ?? numberField((config.rope_scaling as Record<string, unknown> | undefined)?.rope_theta) ?? 10000,
    attentionBias: booleanField(config.attention_bias) ?? false,
    attentionDropout: numberField(config.attention_dropout) ?? 0,
    mlpBias: booleanField(config.mlp_bias) ?? false,
    tieWordEmbeddings: booleanField(config.tie_word_embeddings) ?? false
  });
}

export function mapPhi3ConfigToIr(
  config: Record<string, unknown>,
  options: { modelId?: string; name?: string } = {}
): Phi3ArchitectureSpec {
  const hiddenSize = numberField(config.hidden_size) ?? 3072;
  const numAttentionHeads = numberField(config.num_attention_heads) ?? 32;
  const headDim = numberField(config.head_dim) ?? Math.floor(hiddenSize / numAttentionHeads);

  return buildPhi3ArchitectureSpec({
    name: options.name ?? "Phi-3",
    modelId: options.modelId,
    vocabSize: numberField(config.vocab_size) ?? 32064,
    hiddenSize,
    intermediateSize: numberField(config.intermediate_size) ?? 8192,
    numHiddenLayers: numberField(config.num_hidden_layers) ?? 32,
    numAttentionHeads,
    numKeyValueHeads: numberField(config.num_key_value_heads) ?? numAttentionHeads,
    headDim,
    hiddenActivation: stringField(config.hidden_act) ?? "silu",
    maxPositionEmbeddings: numberField(config.max_position_embeddings) ?? 4096,
    rmsNormEpsilon: numberField(config.rms_norm_eps) ?? 1e-5,
    ropeTheta: numberField(config.rope_theta) ?? numberField((config.rope_scaling as Record<string, unknown> | undefined)?.rope_theta) ?? 10000,
    attentionBias: booleanField(config.attention_bias) ?? false,
    attentionDropout: numberField(config.attention_dropout) ?? 0,
    mlpBias: booleanField(config.mlp_bias) ?? false,
    tieWordEmbeddings: booleanField(config.tie_word_embeddings) ?? false
  });
}

export function mapGemma4ConfigToIr(
  config: Record<string, unknown>,
  options: { modelId?: string; name?: string } = {}
): Gemma4ArchitectureSpec {
  const textConfig = typeof config.text_config === "object" && config.text_config && !Array.isArray(config.text_config)
    ? (config.text_config as Record<string, unknown>)
    : config;
  const hiddenSize = numberField(textConfig.hidden_size) ?? 5376;
  const numAttentionHeads = numberField(textConfig.num_attention_heads) ?? 32;
  const layerTypes = Array.isArray(textConfig.layer_types)
    ? textConfig.layer_types.filter((value): value is string => typeof value === "string")
    : undefined;
  const ropeParameters = typeof textConfig.rope_parameters === "object" && textConfig.rope_parameters && !Array.isArray(textConfig.rope_parameters)
    ? Object.fromEntries(
        Object.entries(textConfig.rope_parameters as Record<string, unknown>)
          .map(([key, value]) => {
            if (!value || typeof value !== "object" || Array.isArray(value)) return null;
            const record = value as Record<string, unknown>;
            return [
              key,
              {
                ropeType: stringField(record.rope_type) ?? "default",
                ropeTheta: numberField(record.rope_theta) ?? 1_000_000
              }
            ] as const;
          })
          .filter((entry): entry is readonly [string, { ropeType: string; ropeTheta: number }] => entry !== null)
      )
    : undefined;
  return buildGemma4ArchitectureSpec({
    name: options.name ?? "Gemma 4",
    modelId: options.modelId,
    vocabSize: numberField(textConfig.vocab_size) ?? 262144,
    hiddenSize,
    intermediateSize: numberField(textConfig.intermediate_size) ?? 21504,
    numHiddenLayers: numberField(textConfig.num_hidden_layers) ?? 60,
    numAttentionHeads,
    numKeyValueHeads: numberField(textConfig.num_key_value_heads) ?? 16,
    headDim: numberField(textConfig.head_dim) ?? Math.floor(hiddenSize / numAttentionHeads),
    hiddenActivation: stringField(textConfig.hidden_activation) ?? stringField(textConfig.hidden_act) ?? "gelu_pytorch_tanh",
    maxPositionEmbeddings: numberField(textConfig.max_position_embeddings) ?? 262144,
    rmsNormEpsilon: numberField(textConfig.rms_norm_eps) ?? 1e-6,
    ropeTheta: numberField(textConfig.rope_theta) ?? 1_000_000,
    attentionBias: booleanField(textConfig.attention_bias) ?? false,
    attentionDropout: numberField(textConfig.attention_dropout) ?? 0,
    mlpBias: booleanField(textConfig.mlp_bias) ?? false,
    tieWordEmbeddings: booleanField(textConfig.tie_word_embeddings) ?? true,
    slidingWindow: numberField(textConfig.sliding_window) ?? 1024,
    layerTypes,
    ropeParameters,
    numGlobalKeyValueHeads: numberField(textConfig.num_global_key_value_heads) ?? 4,
    globalHeadDim: numberField(textConfig.global_head_dim) ?? numberField(textConfig.head_dim) ?? Math.floor(hiddenSize / numAttentionHeads),
    attentionKEqV: booleanField(textConfig.attention_k_eq_v) ?? false,
    numKvSharedLayers: numberField(textConfig.num_kv_shared_layers) ?? 0
  });
}

export function mapModelGraphToPhi3Ir(graph: ModelGraph): Phi3ArchitectureSpec {
  const ordered = topologicalNodes(graph);
  const exactTypes = ["Input", "Phi3TokenEmbedding", "Phi3Block", "Phi3FinalRMSNorm", "Phi3LMHead", "Output"];
  const invalidTypes = ordered.filter((node) => !exactTypes.includes(node.type));
  if (invalidTypes.length > 0) {
    throw new Error(`Phi-3 IR mapping only supports ${exactTypes.join(", ")}. Found ${invalidTypes[0]!.type}.`);
  }

  const inputNode = ordered.find((node) => node.type === "Input");
  const embeddingNode = ordered.find((node) => node.type === "Phi3TokenEmbedding");
  const finalNormNode = ordered.find((node) => node.type === "Phi3FinalRMSNorm");
  const lmHeadNode = ordered.find((node) => node.type === "Phi3LMHead");
  const outputNode = ordered.find((node) => node.type === "Output");
  const blocks = ordered.filter((node) => node.type === "Phi3Block");

  if (!inputNode || !embeddingNode || !finalNormNode || !lmHeadNode || !outputNode) {
    throw new Error(
      "Exact Phi-3 projection is incomplete. Expected Input, Phi3TokenEmbedding, Phi3Block*, Phi3FinalRMSNorm, Phi3LMHead, and Output."
    );
  }

  if (String(outputNode.config.headType ?? "LanguageModel") !== "LanguageModel") {
    throw new Error("Phi-3 IR mapping requires Output head type LanguageModel.");
  }

  const hiddenSize = Number(embeddingNode.config.embeddingDim ?? 3072);
  const firstBlock = blocks[0];
  if (blocks.some(isCustomizedPhi3Block)) {
    throw new Error("Customized Phi-3 block internals are not exported through the exact Phi-3 path yet.");
  }

  return buildPhi3ArchitectureSpec({
    name: "Phi-3 (exact canvas)",
    vocabSize: Number(embeddingNode.config.vocabSize ?? lmHeadNode.config.vocabSize ?? 32064),
    hiddenSize,
    intermediateSize: Number(firstBlock?.config.ffnHidden ?? 8192),
    numHiddenLayers: blocks.length,
    numAttentionHeads: Number(firstBlock?.config.numHeads ?? 32),
    numKeyValueHeads: Number(firstBlock?.config.numKeyValueHeads ?? firstBlock?.config.numHeads ?? 32),
    headDim: Number(firstBlock?.config.headDim ?? Math.floor(hiddenSize / Number(firstBlock?.config.numHeads ?? 32))),
    hiddenActivation: String(firstBlock?.config.activation ?? normalizeActivationName(graph.training.activation)),
    maxPositionEmbeddings: Number(inputNode.config.sequenceLength ?? 4096),
    rmsNormEpsilon: Number(firstBlock?.config.rmsNormEpsilon ?? finalNormNode.config.epsilon ?? 1e-5),
    ropeTheta: Number(firstBlock?.config.ropeTheta ?? 10000),
    attentionBias: Boolean(firstBlock?.config.attentionBias ?? false),
    attentionDropout: Number(firstBlock?.config.dropout ?? 0),
    mlpBias: Boolean(firstBlock?.config.mlpBias ?? false),
    tieWordEmbeddings: Boolean(lmHeadNode.config.tiedWeights ?? false),
    modelId: undefined
  });
}

export function mapModelGraphToGemma4Ir(graph: ModelGraph): Gemma4ArchitectureSpec {
  const ordered = topologicalNodes(graph);
  const exactTypes = ["Input", "Gemma4TokenEmbedding", "Gemma4Block", "Gemma4FinalRMSNorm", "Gemma4LMHead", "Output"];
  const invalidTypes = ordered.filter((node) => !exactTypes.includes(node.type));
  if (invalidTypes.length > 0) {
    throw new Error(`Gemma 4 IR mapping only supports ${exactTypes.join(", ")}. Found ${invalidTypes[0]!.type}.`);
  }
  const inputNode = ordered.find((node) => node.type === "Input");
  const embeddingNode = ordered.find((node) => node.type === "Gemma4TokenEmbedding");
  const finalNormNode = ordered.find((node) => node.type === "Gemma4FinalRMSNorm");
  const lmHeadNode = ordered.find((node) => node.type === "Gemma4LMHead");
  const outputNode = ordered.find((node) => node.type === "Output");
  const blocks = ordered.filter((node) => node.type === "Gemma4Block");
  if (!inputNode || !embeddingNode || !finalNormNode || !lmHeadNode || !outputNode) {
    throw new Error("Exact Gemma 4 projection is incomplete. Expected Input, Gemma4TokenEmbedding, Gemma4Block*, Gemma4FinalRMSNorm, Gemma4LMHead, and Output.");
  }
  if (String(outputNode.config.headType ?? "LanguageModel") !== "LanguageModel") {
    throw new Error("Gemma 4 IR mapping requires Output head type LanguageModel.");
  }
  const hiddenSize = Number(embeddingNode.config.embeddingDim ?? 5376);
  const firstBlock = blocks[0];
  return buildGemma4ArchitectureSpec({
    name: "Gemma 4 (exact canvas)",
    vocabSize: Number(embeddingNode.config.vocabSize ?? lmHeadNode.config.vocabSize ?? 262144),
    hiddenSize,
    intermediateSize: Number(firstBlock?.config.ffnHidden ?? 21504),
    numHiddenLayers: blocks.length,
    numAttentionHeads: Number(firstBlock?.config.numHeads ?? 32),
    numKeyValueHeads: Number(firstBlock?.config.numKeyValueHeads ?? 16),
    headDim: Number(firstBlock?.config.headDim ?? Math.floor(hiddenSize / Number(firstBlock?.config.numHeads ?? 32))),
    hiddenActivation: String(firstBlock?.config.activation ?? "gelu_pytorch_tanh"),
    maxPositionEmbeddings: Number(inputNode.config.sequenceLength ?? 262144),
    rmsNormEpsilon: Number(firstBlock?.config.rmsNormEpsilon ?? finalNormNode.config.epsilon ?? 1e-6),
    ropeTheta: Number(firstBlock?.config.ropeTheta ?? 1_000_000),
    attentionBias: Boolean(firstBlock?.config.attentionBias ?? false),
    attentionDropout: Number(firstBlock?.config.dropout ?? 0),
    mlpBias: Boolean(firstBlock?.config.mlpBias ?? false),
    tieWordEmbeddings: Boolean(lmHeadNode.config.tiedWeights ?? true),
    slidingWindow: Number(firstBlock?.config.slidingWindow ?? 1024),
    layerTypes: blocks.map((block, index) => String(block.config.layerType ?? (index === blocks.length - 1 ? "full_attention" : "sliding_attention"))),
    ropeParameters: {
      sliding_attention: {
        ropeType: "default",
        ropeTheta: Number(firstBlock?.config.ropeTheta ?? 1_000_000)
      },
      full_attention: {
        ropeType: "default",
        ropeTheta: Number(firstBlock?.config.ropeTheta ?? 1_000_000)
      }
    },
    numGlobalKeyValueHeads: Number(firstBlock?.config.numGlobalKeyValueHeads ?? 4),
    globalHeadDim: Number(firstBlock?.config.globalHeadDim ?? firstBlock?.config.headDim ?? 512),
    attentionKEqV: Boolean(firstBlock?.config.attentionKEqV ?? false),
    numKvSharedLayers: Number(firstBlock?.config.numKvSharedLayers ?? 0)
  });
}

export function mapModelGraphToLlamaIr(graph: ModelGraph): LlamaArchitectureSpec {
  const ordered = topologicalNodes(graph);
  const exactTypes = ["Input", "LlamaTokenEmbedding", "LlamaBlock", "LlamaFinalRMSNorm", "LlamaLMHead", "Output"];
  const invalidTypes = ordered.filter((node) => !exactTypes.includes(node.type));
  if (invalidTypes.length > 0) {
    throw new Error(`LLaMA IR mapping only supports ${exactTypes.join(", ")}. Found ${invalidTypes[0]!.type}.`);
  }

  const inputNode = ordered.find((node) => node.type === "Input");
  const embeddingNode = ordered.find((node) => node.type === "LlamaTokenEmbedding");
  const finalNormNode = ordered.find((node) => node.type === "LlamaFinalRMSNorm");
  const lmHeadNode = ordered.find((node) => node.type === "LlamaLMHead");
  const outputNode = ordered.find((node) => node.type === "Output");
  const blocks = ordered.filter((node) => node.type === "LlamaBlock");

  if (!inputNode || !embeddingNode || !finalNormNode || !lmHeadNode || !outputNode) {
    throw new Error(
      "Exact LLaMA projection is incomplete. Expected Input, LlamaTokenEmbedding, LlamaBlock*, LlamaFinalRMSNorm, LlamaLMHead, and Output."
    );
  }

  if (String(outputNode.config.headType ?? "LanguageModel") !== "LanguageModel") {
    throw new Error("LLaMA IR mapping requires Output head type LanguageModel.");
  }

  const hiddenSize = Number(embeddingNode.config.embeddingDim ?? 4096);
  const firstBlock = blocks[0];
  if (blocks.some(isCustomizedLlamaBlock)) {
    throw new Error("Customized LLaMA block internals are exported through the hybrid path.");
  }

  return buildLlamaArchitectureSpec({
    name: "LLaMA (exact canvas)",
    vocabSize: Number(embeddingNode.config.vocabSize ?? lmHeadNode.config.vocabSize ?? 32000),
    hiddenSize,
    intermediateSize: Number(firstBlock?.config.ffnHidden ?? hiddenSize * 4),
    numHiddenLayers: blocks.length,
    numAttentionHeads: Number(firstBlock?.config.numHeads ?? 32),
    numKeyValueHeads: Number(firstBlock?.config.numKeyValueHeads ?? firstBlock?.config.numHeads ?? 32),
    headDim: Number(firstBlock?.config.headDim ?? Math.floor(hiddenSize / Number(firstBlock?.config.numHeads ?? 32))),
    hiddenActivation: String(firstBlock?.config.activation ?? normalizeActivationName(graph.training.activation)),
    maxPositionEmbeddings: Number(inputNode.config.sequenceLength ?? 2048),
    rmsNormEpsilon: Number(firstBlock?.config.rmsNormEpsilon ?? finalNormNode.config.epsilon ?? 1e-6),
    ropeTheta: Number(firstBlock?.config.ropeTheta ?? 10000),
    attentionBias: Boolean(firstBlock?.config.attentionBias ?? false),
    attentionDropout: Number(firstBlock?.config.dropout ?? 0),
    mlpBias: Boolean(firstBlock?.config.mlpBias ?? false),
    tieWordEmbeddings: Boolean(lmHeadNode.config.tiedWeights ?? false),
    modelId: undefined
  });
}

export function mapModelGraphToHybridIr(graph: ModelGraph): HybridDecoderArchitectureSpec {
  const ordered = topologicalNodes(graph);
  const allowedTypes = [
    "Input",
    "GPT2TokenEmbedding",
    "GPT2PositionEmbedding",
    "Add",
    "Dropout",
    "LlamaTokenEmbedding",
    "Phi3TokenEmbedding",
    "GPT2Block",
    "LlamaBlock",
    "Phi3Block",
    "GPT2FinalLayerNorm",
    "LlamaFinalRMSNorm",
    "Phi3FinalRMSNorm",
    "GPT2LMHead",
    "LlamaLMHead",
    "Phi3LMHead",
    "Output"
  ];
  const invalid = ordered.filter((node) => !allowedTypes.includes(node.type));
  if (invalid.length > 0) {
    throw new Error(`Hybrid export only supports exact GPT-2/LLaMA/Phi-3 decoder stages. Found ${invalid[0]!.type}.`);
  }

  const inputNode = ordered.find((node) => node.type === "Input");
  const outputNode = ordered.find((node) => node.type === "Output");
  if (!inputNode || !outputNode) {
    throw new Error("Hybrid export requires exactly one Input and one Output.");
  }
  if (String(outputNode.config.headType ?? "LanguageModel") !== "LanguageModel") {
    throw new Error("Hybrid export requires Output head type LanguageModel.");
  }

  const gpt2Embedding = ordered.find((node) => node.type === "GPT2TokenEmbedding");
  const llamaEmbedding = ordered.find((node) => node.type === "LlamaTokenEmbedding");
  const phi3Embedding = ordered.find((node) => node.type === "Phi3TokenEmbedding");
  const embeddingCandidates = [gpt2Embedding, llamaEmbedding, phi3Embedding].filter(Boolean);
  if (embeddingCandidates.length !== 1) {
    throw new Error("Hybrid export requires exactly one exact embedding stage.");
  }

  const embeddingNode = embeddingCandidates[0]!;
  const embeddingFamily =
    embeddingNode.type === "GPT2TokenEmbedding" ? "gpt2" : embeddingNode.type === "LlamaTokenEmbedding" ? "llama" : "phi3";
  const hiddenSize = Number(embeddingNode.config.embeddingDim ?? 768);
  const vocabSize = Number(embeddingNode.config.vocabSize ?? 32000);
  const maxPositionEmbeddings = Number(inputNode.config.sequenceLength ?? 1024);

  const exactBlocks = ordered.filter(
    (node) => node.type === "GPT2Block" || node.type === "LlamaBlock" || node.type === "Phi3Block"
  );
  if (exactBlocks.length === 0) {
    throw new Error("Hybrid export requires at least one exact GPT-2, LLaMA, or Phi-3 block.");
  }

  const blockOps = exactBlocks.map((node, index) => {
    const dModel = Number(node.config.dModel ?? hiddenSize);
    if (dModel !== hiddenSize) {
      throw new Error(`Hybrid export requires consistent hidden size across blocks. Found ${dModel} and ${hiddenSize}.`);
    }

    if (node.type === "GPT2Block") {
      const feedforwardType = String(node.config.feedforwardType ?? "mlp") === "moe" ? ("moe" as const) : ("mlp" as const);
      return {
        family: "gpt2" as const,
        kind: "hybrid_block" as const,
        id: node.id,
        input: index === 0 ? "hidden.embeddings" : `hidden.block_${index - 1}`,
        output: `hidden.block_${index}`,
        hiddenSize,
        intermediateSize: Number(node.config.ffnHidden ?? hiddenSize * 4),
        numHeads: Number(node.config.numHeads ?? 12),
        layerNormEpsilon: Number(node.config.layerNormEpsilon ?? 1e-5),
        activation: String(node.config.activation ?? "gelu_new"),
        feedforwardType,
        numExperts: Number(node.config.numExperts ?? 8),
        topK: Number(node.config.topK ?? 2),
        expertHidden: Number(node.config.expertHidden ?? node.config.ffnHidden ?? hiddenSize * 4),
        attnDropout: Number(node.config.dropout ?? 0.1),
        residDropout: Number(node.config.dropout ?? 0.1),
        scaleAttnWeights: Boolean(node.config.scaleAttnWeights ?? true),
        scaleAttnByInverseLayerIdx: Boolean(node.config.scaleAttnByInverseLayerIdx ?? false),
        reorderAndUpcastAttn: Boolean(node.config.reorderAndUpcastAttn ?? false),
        expandTo: {} as never
      };
    }

    return {
      family: (node.type === "Phi3Block" ? "phi3" : "llama") as "llama" | "phi3",
      kind: "hybrid_block" as const,
      id: node.id,
      input: index === 0 ? "hidden.embeddings" : `hidden.block_${index - 1}`,
      output: `hidden.block_${index}`,
      hiddenSize,
      intermediateSize: Number(node.config.ffnHidden ?? hiddenSize * 4),
      numHeads: Number(node.config.numHeads ?? 32),
      numKeyValueHeads: Number(node.config.numKeyValueHeads ?? node.config.numHeads ?? 32),
      headDim: Number(node.config.headDim ?? Math.floor(hiddenSize / Number(node.config.numHeads ?? 32))),
      rmsNormEpsilon: Number(node.config.rmsNormEpsilon ?? (node.type === "Phi3Block" ? 1e-5 : 1e-6)),
      ropeTheta: Number(node.config.ropeTheta ?? 10000),
      activation: String(node.config.activation ?? "silu"),
      feedforwardType: String(node.config.feedforwardType ?? "mlp") === "moe" ? ("moe" as const) : ("mlp" as const),
      numExperts: Number(node.config.numExperts ?? 8),
      topK: Number(node.config.topK ?? 2),
      expertHidden: Number(node.config.expertHidden ?? node.config.ffnHidden ?? hiddenSize * 4),
      attentionBias: Boolean(node.config.attentionBias ?? false),
      attentionDropout: Number(node.config.dropout ?? 0),
      mlpBias: Boolean(node.config.mlpBias ?? false),
      expandTo: {} as never
    };
  });

  const finalNormNode = ordered.find(
    (node) => node.type === "GPT2FinalLayerNorm" || node.type === "LlamaFinalRMSNorm" || node.type === "Phi3FinalRMSNorm"
  );
  if (!finalNormNode) {
    throw new Error("Hybrid export requires a final normalization stage.");
  }
  const finalNormFamily =
    finalNormNode.type === "GPT2FinalLayerNorm" ? "gpt2" : finalNormNode.type === "LlamaFinalRMSNorm" ? "llama" : "phi3";
  const finalNormOp =
    finalNormFamily === "gpt2"
      ? {
          family: "gpt2" as const,
          kind: "hybrid_final_norm" as const,
          id: finalNormNode.id,
          input: `hidden.block_${blockOps.length - 1}`,
          output: "hidden.final_norm",
          hiddenSize,
          epsilon: Number(finalNormNode.config.epsilon ?? 1e-5)
        }
      : {
          family: finalNormFamily as "llama" | "phi3",
          kind: "hybrid_final_norm" as const,
          id: finalNormNode.id,
          input: `hidden.block_${blockOps.length - 1}`,
          output: "hidden.final_norm",
          hiddenSize,
          epsilon: Number(finalNormNode.config.epsilon ?? (finalNormFamily === "phi3" ? 1e-5 : 1e-6))
        };

  const lmHeadNode = ordered.find((node) => node.type === "GPT2LMHead" || node.type === "LlamaLMHead" || node.type === "Phi3LMHead");
  if (!lmHeadNode) {
    throw new Error("Hybrid export requires an LM head stage.");
  }
  const tieWordEmbeddings = Boolean(lmHeadNode.config.tiedWeights ?? (embeddingFamily === "gpt2"));
  const lmHeadFamily = lmHeadNode.type === "GPT2LMHead" ? "gpt2" : lmHeadNode.type === "LlamaLMHead" ? "llama" : "phi3";
  const lmHeadOp =
    lmHeadFamily === "gpt2"
      ? {
          family: "gpt2" as const,
          kind: "hybrid_lm_head" as const,
          id: lmHeadNode.id,
          input: "hidden.final_norm",
          output: "logits",
          hiddenSize,
          vocabSize,
          tiedToEmbedding: embeddingNode.id
        }
      : {
          family: lmHeadFamily as "llama" | "phi3",
          kind: "hybrid_lm_head" as const,
          id: lmHeadNode.id,
          input: "hidden.final_norm",
          output: "logits",
          hiddenSize,
          vocabSize,
          tiedToEmbedding: embeddingNode.id
        };

  const embeddingOp =
    embeddingFamily === "gpt2"
      ? {
          family: "gpt2" as const,
          kind: "hybrid_embeddings" as const,
          id: embeddingNode.id,
          input: "input.tokens",
          output: "hidden.embeddings",
          vocabSize,
          maxPositionEmbeddings,
          hiddenSize,
          embdDropout: Number(ordered.find((node) => node.type === "Dropout")?.config.dropout ?? 0.1),
          tieWordEmbeddings
        }
      : {
          family: embeddingFamily as "llama" | "phi3",
          kind: "hybrid_embeddings" as const,
          id: embeddingNode.id,
          input: "input.tokens",
          output: "hidden.embeddings",
          vocabSize,
          hiddenSize,
          tieWordEmbeddings
        };

  const inputSources = getIncoming(graph, outputNode.id);
  if (inputSources.length !== 1 || inputSources[0] !== lmHeadNode.id) {
    throw new Error("Hybrid export requires a single linear path ending at the LM head.");
  }

  return {
    family: "hybrid_decoder",
    modality: "text",
    task: "causal_lm",
    name: "Hybrid Decoder",
    source: { kind: "manual" },
    config: {
      vocabSize,
      hiddenSize,
      maxPositionEmbeddings,
      tieWordEmbeddings,
      embeddingFamily,
      finalNormFamily,
      blockFamilies: blockOps.map((block) => block.family)
    },
    tensors: {
      inputTokens: { id: "input.tokens", kind: "tokens", shape: ["batch", "seq_len"] },
      hiddenStates: [
        { id: "hidden.embeddings", kind: "sequence", shape: ["batch", "seq_len", "hidden_size"] },
        ...blockOps.map((block) => ({ id: block.output, kind: "sequence" as const, shape: ["batch", "seq_len", "hidden_size"] })),
        { id: "hidden.final_norm", kind: "sequence", shape: ["batch", "seq_len", "hidden_size"] }
      ],
      logits: { id: "logits", kind: "logits", shape: ["batch", "seq_len", "vocab_size"] }
    },
    operators: {
      embedding: embeddingOp,
      blocks: blockOps,
      finalNorm: finalNormOp,
      lmHead: lmHeadOp
    }
  };
}

export function projectLlamaIrToModelGraph(spec: LlamaArchitectureSpec): ModelGraph {
  const inputNode = {
    id: "Input-llama",
    type: "Input" as const,
    position: { x: 40, y: 92 },
    config: {
      sequenceLength: spec.config.maxPositionEmbeddings
    }
  };

  const embeddingNode = {
    id: "LlamaTokenEmbedding-llama",
    type: "LlamaTokenEmbedding" as const,
    position: { x: 300, y: 92 },
    config: {
      vocabSize: spec.config.vocabSize,
      embeddingDim: spec.config.hiddenSize
    }
  };

  const blockNodes = Array.from({ length: spec.config.numHiddenLayers }, (_, index) => ({
    id: `LlamaBlock-llama-${index}`,
    type: "LlamaBlock" as const,
    position: { x: 580 + index * 240, y: 92 + (index % 2) * 72 },
    config: {
      dModel: spec.config.hiddenSize,
      numHeads: spec.config.numAttentionHeads,
      numKeyValueHeads: spec.config.numKeyValueHeads,
      headDim: spec.config.headDim,
      ffnHidden: spec.config.intermediateSize,
      feedforwardType: "mlp",
      numExperts: 8,
      topK: 2,
      expertHidden: spec.config.intermediateSize,
      ropeTheta: spec.config.ropeTheta,
      rmsNormEpsilon: spec.config.rmsNormEpsilon,
      activation: spec.config.hiddenActivation,
      attentionBias: spec.config.attentionBias,
      dropout: spec.config.attentionDropout,
      mlpBias: spec.config.mlpBias
    }
  }));

  const finalNormNode = {
    id: "LlamaFinalRMSNorm-llama",
    type: "LlamaFinalRMSNorm" as const,
    position: { x: 580 + spec.config.numHiddenLayers * 240, y: 92 },
    config: {
      epsilon: spec.config.rmsNormEpsilon
    }
  };

  const lmHeadNode = {
    id: "LlamaLMHead-llama",
    type: "LlamaLMHead" as const,
    position: { x: 820 + spec.config.numHiddenLayers * 240, y: 92 },
    config: {
      vocabSize: spec.config.vocabSize,
      tiedWeights: spec.config.tieWordEmbeddings
    }
  };

  const outputNode = {
    id: "Output-llama",
    type: "Output" as const,
    position: { x: 1060 + spec.config.numHiddenLayers * 240, y: 92 },
    config: {
      headType: "LanguageModel"
    }
  };

  const nodes = [inputNode, embeddingNode, ...blockNodes, finalNormNode, lmHeadNode, outputNode];
  const edges = [
    { id: "edge-llama-1", source: inputNode.id, target: embeddingNode.id },
    ...blockNodes.map((node, index) => ({
      id: `edge-llama-block-${index + 1}`,
      source: index === 0 ? embeddingNode.id : blockNodes[index - 1]!.id,
      target: node.id
    })),
    {
      id: "edge-llama-final-norm",
      source: blockNodes.length > 0 ? blockNodes[blockNodes.length - 1]!.id : embeddingNode.id,
      target: finalNormNode.id
    },
    { id: "edge-llama-lm-head", source: finalNormNode.id, target: lmHeadNode.id },
    { id: "edge-llama-output", source: lmHeadNode.id, target: outputNode.id }
  ];

  return {
    nodes,
    edges,
    training: {
      optimizer: "AdamW",
      loss: "CrossEntropy",
      learningRate: 0.0003,
      activation: spec.config.hiddenActivation.toLowerCase() === "silu" ? "SiLU" : "Custom",
      optimizerCustomName: "",
      lossCustomName: "",
      activationCustomName: spec.config.hiddenActivation.toLowerCase() === "silu" ? "" : spec.config.hiddenActivation
    }
  };
}

export function projectPhi3IrToModelGraph(spec: Phi3ArchitectureSpec): ModelGraph {
  const inputNode = {
    id: "Input-phi3",
    type: "Input" as const,
    position: { x: 40, y: 92 },
    config: {
      sequenceLength: spec.config.maxPositionEmbeddings
    }
  };

  const embeddingNode = {
    id: "Phi3TokenEmbedding-phi3",
    type: "Phi3TokenEmbedding" as const,
    position: { x: 300, y: 92 },
    config: {
      vocabSize: spec.config.vocabSize,
      embeddingDim: spec.config.hiddenSize
    }
  };

  const blockNodes = Array.from({ length: spec.config.numHiddenLayers }, (_, index) => ({
    id: `Phi3Block-phi3-${index}`,
    type: "Phi3Block" as const,
    position: { x: 580 + index * 240, y: 92 + (index % 2) * 72 },
    config: {
      dModel: spec.config.hiddenSize,
      numHeads: spec.config.numAttentionHeads,
      numKeyValueHeads: spec.config.numKeyValueHeads,
      headDim: spec.config.headDim,
      ffnHidden: spec.config.intermediateSize,
      feedforwardType: "mlp",
      numExperts: 8,
      topK: 2,
      expertHidden: spec.config.intermediateSize,
      ropeTheta: spec.config.ropeTheta,
      rmsNormEpsilon: spec.config.rmsNormEpsilon,
      activation: spec.config.hiddenActivation,
      attentionBias: spec.config.attentionBias,
      dropout: spec.config.attentionDropout,
      mlpBias: spec.config.mlpBias
    }
  }));

  const finalNormNode = {
    id: "Phi3FinalRMSNorm-phi3",
    type: "Phi3FinalRMSNorm" as const,
    position: { x: 580 + spec.config.numHiddenLayers * 240, y: 92 },
    config: {
      epsilon: spec.config.rmsNormEpsilon
    }
  };

  const lmHeadNode = {
    id: "Phi3LMHead-phi3",
    type: "Phi3LMHead" as const,
    position: { x: 820 + spec.config.numHiddenLayers * 240, y: 92 },
    config: {
      vocabSize: spec.config.vocabSize,
      tiedWeights: spec.config.tieWordEmbeddings
    }
  };

  const outputNode = {
    id: "Output-phi3",
    type: "Output" as const,
    position: { x: 1060 + spec.config.numHiddenLayers * 240, y: 92 },
    config: {
      headType: "LanguageModel"
    }
  };

  const nodes = [inputNode, embeddingNode, ...blockNodes, finalNormNode, lmHeadNode, outputNode];
  const edges = [
    { id: "edge-phi3-1", source: inputNode.id, target: embeddingNode.id },
    ...blockNodes.map((node, index) => ({
      id: `edge-phi3-block-${index + 1}`,
      source: index === 0 ? embeddingNode.id : blockNodes[index - 1]!.id,
      target: node.id
    })),
    {
      id: "edge-phi3-final-norm",
      source: blockNodes.length > 0 ? blockNodes[blockNodes.length - 1]!.id : embeddingNode.id,
      target: finalNormNode.id
    },
    { id: "edge-phi3-lm-head", source: finalNormNode.id, target: lmHeadNode.id },
    { id: "edge-phi3-output", source: lmHeadNode.id, target: outputNode.id }
  ];

  return {
    nodes,
    edges,
    training: {
      optimizer: "AdamW",
      loss: "CrossEntropy",
      learningRate: 0.0003,
      activation: spec.config.hiddenActivation.toLowerCase() === "silu" ? "SiLU" : "Custom",
      optimizerCustomName: "",
      lossCustomName: "",
      activationCustomName: spec.config.hiddenActivation.toLowerCase() === "silu" ? "" : spec.config.hiddenActivation
    }
  };
}

export function projectGemma4IrToModelGraph(spec: Gemma4ArchitectureSpec): ModelGraph {
  const inputNode = {
    id: "Input-gemma4",
    type: "Input" as const,
    position: { x: 40, y: 92 },
    config: { sequenceLength: spec.config.maxPositionEmbeddings }
  };
  const embeddingNode = {
    id: "Gemma4TokenEmbedding-gemma4",
    type: "Gemma4TokenEmbedding" as const,
    position: { x: 300, y: 92 },
    config: { vocabSize: spec.config.vocabSize, embeddingDim: spec.config.hiddenSize }
  };
  const blockNodes = Array.from({ length: spec.config.numHiddenLayers }, (_, index) => ({
    id: `Gemma4Block-gemma4-${index}`,
    type: "Gemma4Block" as const,
    position: { x: 560 + index * 240, y: 92 + (index % 2) * 72 },
    config: {
      dModel: spec.config.hiddenSize,
      numHeads: spec.config.numAttentionHeads,
      numKeyValueHeads: spec.config.numKeyValueHeads,
      ffnHidden: spec.config.intermediateSize,
      feedforwardType: "mlp",
      numExperts: 8,
      topK: 2,
      expertHidden: spec.config.intermediateSize,
      layerType: spec.config.layerTypes[index] ?? (index === spec.config.numHiddenLayers - 1 ? "full_attention" : "sliding_attention"),
      slidingWindow: spec.config.slidingWindow,
      ropeTheta: spec.config.ropeTheta,
      headDim: spec.config.headDim,
      numGlobalKeyValueHeads: spec.config.numGlobalKeyValueHeads ?? 4,
      globalHeadDim: spec.config.globalHeadDim,
      rmsNormEpsilon: spec.config.rmsNormEpsilon,
      activation: spec.config.hiddenActivation,
      attentionBias: spec.config.attentionBias,
      attentionKEqV: spec.config.attentionKEqV,
      dropout: spec.config.attentionDropout,
      mlpBias: spec.config.mlpBias,
      numKvSharedLayers: spec.config.numKvSharedLayers
    }
  }));
  const finalNormNode = {
    id: "Gemma4FinalRMSNorm-gemma4",
    type: "Gemma4FinalRMSNorm" as const,
    position: { x: 560 + spec.config.numHiddenLayers * 240, y: 92 },
    config: { epsilon: spec.config.rmsNormEpsilon }
  };
  const lmHeadNode = {
    id: "Gemma4LMHead-gemma4",
    type: "Gemma4LMHead" as const,
    position: { x: 800 + spec.config.numHiddenLayers * 240, y: 92 },
    config: { vocabSize: spec.config.vocabSize, tiedWeights: spec.config.tieWordEmbeddings }
  };
  const outputNode = {
    id: "Output-gemma4",
    type: "Output" as const,
    position: { x: 1040 + spec.config.numHiddenLayers * 240, y: 92 },
    config: { headType: "LanguageModel" }
  };
  const nodes = [inputNode, embeddingNode, ...blockNodes, finalNormNode, lmHeadNode, outputNode];
  const edges = [
    { id: "edge-gemma4-1", source: inputNode.id, target: embeddingNode.id },
    ...blockNodes.map((node, index) => ({
      id: `edge-gemma4-block-${index + 1}`,
      source: index === 0 ? embeddingNode.id : blockNodes[index - 1]!.id,
      target: node.id
    })),
    { id: "edge-gemma4-final-norm", source: blockNodes.length > 0 ? blockNodes[blockNodes.length - 1]!.id : embeddingNode.id, target: finalNormNode.id },
    { id: "edge-gemma4-lm-head", source: finalNormNode.id, target: lmHeadNode.id },
    { id: "edge-gemma4-output", source: lmHeadNode.id, target: outputNode.id }
  ];
  return { nodes, edges, training: { optimizer: "AdamW", learningRate: 3e-4, loss: "CrossEntropy", activation: "GELU" } };
}

function numberField(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function stringField(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value : undefined;
}

function booleanField(value: unknown): boolean | undefined {
  return typeof value === "boolean" ? value : undefined;
}

function normalizeActivationName(value: string): string {
  switch (value) {
    case "GELU":
      return "gelu_new";
    case "ReLU":
      return "relu";
    case "SiLU":
      return "silu";
    case "Custom":
      return "custom_activation";
    default:
      return value.toLowerCase();
  }
}
