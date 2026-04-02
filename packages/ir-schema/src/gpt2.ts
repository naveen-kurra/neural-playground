import type { GPT2ArchitectureSpec, GPT2BlockOp, GPT2EmbeddingOp, GPT2FinalNormOp, GPT2LmHeadOp } from "./types";

export type GPT2ConfigInput = {
  name?: string;
  modelId?: string;
  vocabSize?: number;
  maxPositionEmbeddings?: number;
  hiddenSize?: number;
  numHiddenLayers?: number;
  numAttentionHeads?: number;
  intermediateSize?: number;
  activationFunction?: string;
  embdDropout?: number;
  attnDropout?: number;
  residDropout?: number;
  layerNormEpsilon?: number;
  scaleAttnWeights?: boolean;
  scaleAttnByInverseLayerIdx?: boolean;
  reorderAndUpcastAttn?: boolean;
  tieWordEmbeddings?: boolean;
};

export function buildGPT2ArchitectureSpec(input: GPT2ConfigInput = {}): GPT2ArchitectureSpec {
  const config = {
    vocabSize: input.vocabSize ?? 50257,
    maxPositionEmbeddings: input.maxPositionEmbeddings ?? 1024,
    hiddenSize: input.hiddenSize ?? 768,
    numHiddenLayers: input.numHiddenLayers ?? 12,
    numAttentionHeads: input.numAttentionHeads ?? 12,
    intermediateSize: input.intermediateSize ?? (input.hiddenSize ?? 768) * 4,
    activationFunction: input.activationFunction ?? "gelu_new",
    embdDropout: input.embdDropout ?? 0.1,
    attnDropout: input.attnDropout ?? 0.1,
    residDropout: input.residDropout ?? 0.1,
    layerNormEpsilon: input.layerNormEpsilon ?? 1e-5,
    scaleAttnWeights: input.scaleAttnWeights ?? true,
    scaleAttnByInverseLayerIdx: input.scaleAttnByInverseLayerIdx ?? false,
    reorderAndUpcastAttn: input.reorderAndUpcastAttn ?? false,
    tieWordEmbeddings: input.tieWordEmbeddings ?? true
  };

  const embeddingOutput = "hidden.embeddings";
  const finalNormOutput = "hidden.final_norm";
  const logitsOutput = "logits";

  const embeddingOp: GPT2EmbeddingOp = {
    kind: "gpt2_embeddings",
    id: "gpt2_embeddings",
    input: "input.tokens",
    output: embeddingOutput,
    vocabSize: config.vocabSize,
    maxPositionEmbeddings: config.maxPositionEmbeddings,
    hiddenSize: config.hiddenSize,
    embdDropout: config.embdDropout,
    tieWordEmbeddings: config.tieWordEmbeddings
  };

  const blocks: GPT2BlockOp[] = Array.from({ length: config.numHiddenLayers }, (_, index) => {
    const inputRef = index === 0 ? embeddingOutput : `hidden.block_${index - 1}`;
    const outputRef = `hidden.block_${index}`;

    return {
      kind: "gpt2_block",
      id: `gpt2_block_${index}`,
      input: inputRef,
      output: outputRef,
      hiddenSize: config.hiddenSize,
      intermediateSize: config.intermediateSize,
      numHeads: config.numAttentionHeads,
      layerNormEpsilon: config.layerNormEpsilon,
      activation: config.activationFunction,
      attnDropout: config.attnDropout,
      residDropout: config.residDropout,
      scaleAttnWeights: config.scaleAttnWeights,
      scaleAttnByInverseLayerIdx: config.scaleAttnByInverseLayerIdx,
      reorderAndUpcastAttn: config.reorderAndUpcastAttn,
      expandTo: {
        ln1: {
          kind: "gpt2_layer_norm",
          id: `gpt2_block_${index}.ln_1`,
          input: inputRef,
          output: `hidden.block_${index}.ln_1`,
          hiddenSize: config.hiddenSize,
          epsilon: config.layerNormEpsilon
        },
        attn: {
          kind: "gpt2_attention",
          id: `gpt2_block_${index}.attn`,
          input: `hidden.block_${index}.ln_1`,
          output: `hidden.block_${index}.attn_out`,
          hiddenSize: config.hiddenSize,
          numHeads: config.numAttentionHeads,
          attnDropout: config.attnDropout,
          residDropout: config.residDropout,
          scaleAttnWeights: config.scaleAttnWeights,
          scaleAttnByInverseLayerIdx: config.scaleAttnByInverseLayerIdx,
          reorderAndUpcastAttn: config.reorderAndUpcastAttn,
          causal: true
        },
        attnResidual: {
          kind: "residual_add",
          id: `gpt2_block_${index}.attn_residual`,
          inputs: [inputRef, `hidden.block_${index}.attn_out`],
          output: `hidden.block_${index}.attn_residual`
        },
        ln2: {
          kind: "gpt2_layer_norm",
          id: `gpt2_block_${index}.ln_2`,
          input: `hidden.block_${index}.attn_residual`,
          output: `hidden.block_${index}.ln_2`,
          hiddenSize: config.hiddenSize,
          epsilon: config.layerNormEpsilon
        },
        mlp: {
          kind: "gpt2_mlp",
          id: `gpt2_block_${index}.mlp`,
          input: `hidden.block_${index}.ln_2`,
          output: `hidden.block_${index}.mlp_out`,
          hiddenSize: config.hiddenSize,
          intermediateSize: config.intermediateSize,
          activation: config.activationFunction,
          residDropout: config.residDropout
        },
        mlpResidual: {
          kind: "residual_add",
          id: `gpt2_block_${index}.mlp_residual`,
          inputs: [`hidden.block_${index}.attn_residual`, `hidden.block_${index}.mlp_out`],
          output: outputRef
        }
      }
    };
  });

  const finalNormOp: GPT2FinalNormOp = {
    kind: "gpt2_final_layer_norm",
    id: "gpt2_final_ln",
    input: config.numHiddenLayers > 0 ? `hidden.block_${config.numHiddenLayers - 1}` : embeddingOutput,
    output: finalNormOutput,
    hiddenSize: config.hiddenSize,
    epsilon: config.layerNormEpsilon
  };

  const lmHeadOp: GPT2LmHeadOp = {
    kind: "gpt2_lm_head",
    id: "gpt2_lm_head",
    input: finalNormOutput,
    output: logitsOutput,
    hiddenSize: config.hiddenSize,
    vocabSize: config.vocabSize,
    tiedToEmbedding: embeddingOp.id
  };

  return {
    family: "gpt2",
    modality: "text",
    task: "causal_lm",
    name: input.name ?? "GPT-2",
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
