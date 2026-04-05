import type { Gemma4ArchitectureSpec } from "./types";

export type Gemma4ConfigInput = {
  name?: string;
  modelId?: string;
  vocabSize?: number;
  hiddenSize?: number;
  intermediateSize?: number;
  numHiddenLayers?: number;
  numAttentionHeads?: number;
  numKeyValueHeads?: number;
  headDim?: number;
  hiddenActivation?: string;
  maxPositionEmbeddings?: number;
  rmsNormEpsilon?: number;
  ropeTheta?: number;
  attentionBias?: boolean;
  attentionDropout?: number;
  mlpBias?: boolean;
  tieWordEmbeddings?: boolean;
  slidingWindow?: number;
  layerTypes?: string[];
  ropeParameters?: Record<string, { ropeType: string; ropeTheta: number }>;
  numGlobalKeyValueHeads?: number | null;
  globalHeadDim?: number;
  attentionKEqV?: boolean;
  numKvSharedLayers?: number;
};

export function buildGemma4ArchitectureSpec(input: Gemma4ConfigInput = {}): Gemma4ArchitectureSpec {
  const hiddenSize = input.hiddenSize ?? 5376;
  const numAttentionHeads = input.numAttentionHeads ?? 32;
  const numKeyValueHeads = input.numKeyValueHeads ?? 16;
  const numHiddenLayers = input.numHiddenLayers ?? 60;
  const headDim = input.headDim ?? Math.floor(hiddenSize / numAttentionHeads);
  const layerTypes = input.layerTypes ?? (
    numHiddenLayers === 60
      ? Array.from({ length: numHiddenLayers }, (_, index) => ((index + 1) % 6 === 0 ? "full_attention" : "sliding_attention"))
      : [
          ...Array.from({ length: Math.max(numHiddenLayers - 1, 0) }, () => "sliding_attention"),
          "full_attention"
        ]
  );
  const ropeParameters = input.ropeParameters ?? {
    sliding_attention: { ropeType: "default", ropeTheta: 10_000 },
    full_attention: { ropeType: "proportional", ropeTheta: input.ropeTheta ?? 1_000_000 }
  };
  const config = {
    vocabSize: input.vocabSize ?? 262144,
    hiddenSize,
    intermediateSize: input.intermediateSize ?? 21504,
    numHiddenLayers,
    numAttentionHeads,
    numKeyValueHeads,
    headDim,
    hiddenActivation: input.hiddenActivation ?? "gelu_pytorch_tanh",
    maxPositionEmbeddings: input.maxPositionEmbeddings ?? 262144,
    rmsNormEpsilon: input.rmsNormEpsilon ?? 1e-6,
    ropeTheta: input.ropeTheta ?? 1_000_000,
    attentionBias: input.attentionBias ?? false,
    attentionDropout: input.attentionDropout ?? 0,
    mlpBias: input.mlpBias ?? false,
    tieWordEmbeddings: input.tieWordEmbeddings ?? true,
    slidingWindow: input.slidingWindow ?? 1024,
    layerTypes,
    ropeParameters,
    numGlobalKeyValueHeads: input.numGlobalKeyValueHeads ?? 4,
    globalHeadDim: input.globalHeadDim ?? 512,
    attentionKEqV: input.attentionKEqV ?? false,
    numKvSharedLayers: input.numKvSharedLayers ?? 0
  };

  const embeddingOutput = "hidden.embeddings";
  const finalNormOutput = "hidden.final_norm";
  const logitsOutput = "logits";
  const embeddingOp = {
    kind: "llama_embeddings" as const,
    id: "gemma4_embeddings",
    input: "input.tokens",
    output: embeddingOutput,
    vocabSize: config.vocabSize,
    hiddenSize: config.hiddenSize,
    tieWordEmbeddings: config.tieWordEmbeddings
  };
  const blocks = Array.from({ length: config.numHiddenLayers }, (_, index) => ({
    kind: "llama_block" as const,
    id: `gemma4_block_${index}`,
    input: index === 0 ? embeddingOutput : `hidden.block_${index - 1}`,
    output: `hidden.block_${index}`,
    hiddenSize: config.hiddenSize,
    intermediateSize: config.intermediateSize,
    numHeads: config.numAttentionHeads,
    numKeyValueHeads: config.numKeyValueHeads,
    headDim: config.headDim,
    rmsNormEpsilon: config.rmsNormEpsilon,
    ropeTheta: config.ropeTheta,
    activation: config.hiddenActivation,
    attentionBias: config.attentionBias,
    attentionDropout: config.attentionDropout,
    mlpBias: config.mlpBias,
    expandTo: {} as never
  }));
  const finalNormOp = {
    kind: "llama_final_rms_norm" as const,
    id: "gemma4_final_norm",
    input: config.numHiddenLayers > 0 ? `hidden.block_${config.numHiddenLayers - 1}` : embeddingOutput,
    output: finalNormOutput,
    hiddenSize: config.hiddenSize,
    epsilon: config.rmsNormEpsilon
  };
  const lmHeadOp = {
    kind: "llama_lm_head" as const,
    id: "gemma4_lm_head",
    input: finalNormOutput,
    output: logitsOutput,
    hiddenSize: config.hiddenSize,
    vocabSize: config.vocabSize,
    tiedToEmbedding: embeddingOp.id
  };
  return {
    family: "gemma4",
    modality: "text",
    task: "causal_lm",
    name: input.name ?? "Gemma 4",
    source: { kind: input.modelId ? "huggingface-config" : "manual", modelId: input.modelId },
    config,
    tensors: {
      inputTokens: { id: "input.tokens", kind: "tokens", shape: ["batch", "seq_len"] },
      hiddenStates: [
        { id: embeddingOutput, kind: "sequence", shape: ["batch", "seq_len", "hidden_size"] },
        ...blocks.map((block) => ({ id: block.output, kind: "sequence" as const, shape: ["batch", "seq_len", "hidden_size"] })),
        { id: finalNormOutput, kind: "sequence", shape: ["batch", "seq_len", "hidden_size"] }
      ],
      logits: { id: logitsOutput, kind: "logits", shape: ["batch", "seq_len", "vocab_size"] }
    },
    operators: [embeddingOp, ...blocks, finalNormOp, lmHeadOp]
  };
}
