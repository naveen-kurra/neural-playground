import type {
  LlamaArchitectureSpec,
  LlamaBlockOp,
  LlamaEmbeddingOp,
  LlamaFinalNormOp,
  LlamaLmHeadOp
} from "./types";

export type LlamaConfigInput = {
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
};

export function buildLlamaArchitectureSpec(input: LlamaConfigInput = {}): LlamaArchitectureSpec {
  const hiddenSize = input.hiddenSize ?? 4096;
  const numAttentionHeads = input.numAttentionHeads ?? 32;
  const numKeyValueHeads = input.numKeyValueHeads ?? numAttentionHeads;
  const headDim = input.headDim ?? Math.floor(hiddenSize / numAttentionHeads);

  const config = {
    vocabSize: input.vocabSize ?? 32000,
    hiddenSize,
    intermediateSize: input.intermediateSize ?? 11008,
    numHiddenLayers: input.numHiddenLayers ?? 32,
    numAttentionHeads,
    numKeyValueHeads,
    headDim,
    hiddenActivation: input.hiddenActivation ?? "silu",
    maxPositionEmbeddings: input.maxPositionEmbeddings ?? 2048,
    rmsNormEpsilon: input.rmsNormEpsilon ?? 1e-6,
    ropeTheta: input.ropeTheta ?? 10000,
    attentionBias: input.attentionBias ?? false,
    attentionDropout: input.attentionDropout ?? 0,
    mlpBias: input.mlpBias ?? false,
    tieWordEmbeddings: input.tieWordEmbeddings ?? false
  };

  const embeddingOutput = "hidden.embeddings";
  const finalNormOutput = "hidden.final_norm";
  const logitsOutput = "logits";

  const embeddingOp: LlamaEmbeddingOp = {
    kind: "llama_embeddings",
    id: "llama_embeddings",
    input: "input.tokens",
    output: embeddingOutput,
    vocabSize: config.vocabSize,
    hiddenSize: config.hiddenSize,
    tieWordEmbeddings: config.tieWordEmbeddings
  };

  const blocks: LlamaBlockOp[] = Array.from({ length: config.numHiddenLayers }, (_, index) => {
    const inputRef = index === 0 ? embeddingOutput : `hidden.block_${index - 1}`;
    const outputRef = `hidden.block_${index}`;

    return {
      kind: "llama_block",
      id: `llama_block_${index}`,
      input: inputRef,
      output: outputRef,
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
      expandTo: {
        inputNorm: {
          kind: "llama_rms_norm",
          id: `llama_block_${index}.input_layernorm`,
          input: inputRef,
          output: `hidden.block_${index}.input_norm`,
          hiddenSize: config.hiddenSize,
          epsilon: config.rmsNormEpsilon
        },
        attention: {
          kind: "llama_attention",
          id: `llama_block_${index}.self_attn`,
          input: `hidden.block_${index}.input_norm`,
          output: `hidden.block_${index}.attn_out`,
          hiddenSize: config.hiddenSize,
          numHeads: config.numAttentionHeads,
          numKeyValueHeads: config.numKeyValueHeads,
          headDim: config.headDim,
          attentionBias: config.attentionBias,
          attentionDropout: config.attentionDropout,
          ropeTheta: config.ropeTheta
        },
        attentionResidual: {
          kind: "residual_add",
          id: `llama_block_${index}.attn_residual`,
          inputs: [inputRef, `hidden.block_${index}.attn_out`],
          output: `hidden.block_${index}.attn_residual`
        },
        postAttentionNorm: {
          kind: "llama_rms_norm",
          id: `llama_block_${index}.post_attention_layernorm`,
          input: `hidden.block_${index}.attn_residual`,
          output: `hidden.block_${index}.post_attention_norm`,
          hiddenSize: config.hiddenSize,
          epsilon: config.rmsNormEpsilon
        },
        mlp: {
          kind: "llama_mlp",
          id: `llama_block_${index}.mlp`,
          input: `hidden.block_${index}.post_attention_norm`,
          output: `hidden.block_${index}.mlp_out`,
          hiddenSize: config.hiddenSize,
          intermediateSize: config.intermediateSize,
          activation: config.hiddenActivation,
          mlpBias: config.mlpBias
        },
        mlpResidual: {
          kind: "residual_add",
          id: `llama_block_${index}.mlp_residual`,
          inputs: [`hidden.block_${index}.attn_residual`, `hidden.block_${index}.mlp_out`],
          output: outputRef
        }
      }
    };
  });

  const finalNormOp: LlamaFinalNormOp = {
    kind: "llama_final_rms_norm",
    id: "llama_final_norm",
    input: config.numHiddenLayers > 0 ? `hidden.block_${config.numHiddenLayers - 1}` : embeddingOutput,
    output: finalNormOutput,
    hiddenSize: config.hiddenSize,
    epsilon: config.rmsNormEpsilon
  };

  const lmHeadOp: LlamaLmHeadOp = {
    kind: "llama_lm_head",
    id: "llama_lm_head",
    input: finalNormOutput,
    output: logitsOutput,
    hiddenSize: config.hiddenSize,
    vocabSize: config.vocabSize,
    tiedToEmbedding: embeddingOp.id
  };

  return {
    family: "llama",
    modality: "text",
    task: "causal_lm",
    name: input.name ?? "LLaMA",
    source: {
      kind: input.modelId ? "huggingface-config" : "manual",
      modelId: input.modelId
    },
    config,
    tensors: {
      inputTokens: {
        id: "input.tokens",
        kind: "tokens",
        shape: ["batch", "seq_len"]
      },
      hiddenStates: [
        {
          id: embeddingOutput,
          kind: "sequence",
          shape: ["batch", "seq_len", "hidden_size"]
        },
        ...blocks.map((block) => ({
          id: block.output,
          kind: "sequence" as const,
          shape: ["batch", "seq_len", "hidden_size"]
        })),
        {
          id: finalNormOutput,
          kind: "sequence",
          shape: ["batch", "seq_len", "hidden_size"]
        }
      ],
      logits: {
        id: logitsOutput,
        kind: "logits",
        shape: ["batch", "seq_len", "vocab_size"]
      }
    },
    operators: [embeddingOp, ...blocks, finalNormOp, lmHeadOp]
  };
}
