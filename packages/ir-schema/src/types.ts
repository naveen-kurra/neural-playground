export type ModelFamily = "gpt2" | "llama" | "phi3" | "gemma4";

export type Modality = "text";

export type TaskType = "causal_lm";

export type TensorRef = {
  id: string;
  kind: "tokens" | "sequence" | "logits";
  shape: string[];
};

export type GPT2EmbeddingOp = {
  kind: "gpt2_embeddings";
  id: string;
  input: string;
  output: string;
  vocabSize: number;
  maxPositionEmbeddings: number;
  hiddenSize: number;
  embdDropout: number;
  tieWordEmbeddings: boolean;
};

export type GPT2AttentionOp = {
  kind: "gpt2_attention";
  id: string;
  input: string;
  output: string;
  hiddenSize: number;
  numHeads: number;
  attnDropout: number;
  residDropout: number;
  scaleAttnWeights: boolean;
  scaleAttnByInverseLayerIdx: boolean;
  reorderAndUpcastAttn: boolean;
  causal: true;
};

export type GPT2MlpOp = {
  kind: "gpt2_mlp";
  id: string;
  input: string;
  output: string;
  hiddenSize: number;
  intermediateSize: number;
  activation: string;
  residDropout: number;
};

export type GPT2LayerNormOp = {
  kind: "gpt2_layer_norm";
  id: string;
  input: string;
  output: string;
  hiddenSize: number;
  epsilon: number;
};

export type ResidualAddOp = {
  kind: "residual_add";
  id: string;
  inputs: [string, string];
  output: string;
};

export type GPT2BlockOp = {
  kind: "gpt2_block";
  id: string;
  input: string;
  output: string;
  hiddenSize: number;
  intermediateSize: number;
  numHeads: number;
  layerNormEpsilon: number;
  activation: string;
  attnDropout: number;
  residDropout: number;
  scaleAttnWeights: boolean;
  scaleAttnByInverseLayerIdx: boolean;
  reorderAndUpcastAttn: boolean;
  expandTo: {
    ln1: GPT2LayerNormOp;
    attn: GPT2AttentionOp;
    attnResidual: ResidualAddOp;
    ln2: GPT2LayerNormOp;
    mlp: GPT2MlpOp;
    mlpResidual: ResidualAddOp;
  };
};

export type GPT2FinalNormOp = {
  kind: "gpt2_final_layer_norm";
  id: string;
  input: string;
  output: string;
  hiddenSize: number;
  epsilon: number;
};

export type GPT2LmHeadOp = {
  kind: "gpt2_lm_head";
  id: string;
  input: string;
  output: string;
  hiddenSize: number;
  vocabSize: number;
  tiedToEmbedding: string;
};

export type GPT2Operator =
  | GPT2EmbeddingOp
  | GPT2BlockOp
  | GPT2FinalNormOp
  | GPT2LmHeadOp;

export type LlamaEmbeddingOp = {
  kind: "llama_embeddings";
  id: string;
  input: string;
  output: string;
  vocabSize: number;
  hiddenSize: number;
  tieWordEmbeddings: boolean;
};

export type LlamaAttentionOp = {
  kind: "llama_attention";
  id: string;
  input: string;
  output: string;
  hiddenSize: number;
  numHeads: number;
  numKeyValueHeads: number;
  headDim: number;
  attentionBias: boolean;
  attentionDropout: number;
  ropeTheta: number;
};

export type LlamaMlpOp = {
  kind: "llama_mlp";
  id: string;
  input: string;
  output: string;
  hiddenSize: number;
  intermediateSize: number;
  activation: string;
  mlpBias: boolean;
};

export type LlamaRmsNormOp = {
  kind: "llama_rms_norm";
  id: string;
  input: string;
  output: string;
  hiddenSize: number;
  epsilon: number;
};

export type LlamaBlockOp = {
  kind: "llama_block";
  id: string;
  input: string;
  output: string;
  hiddenSize: number;
  intermediateSize: number;
  numHeads: number;
  numKeyValueHeads: number;
  headDim: number;
  rmsNormEpsilon: number;
  ropeTheta: number;
  activation: string;
  attentionBias: boolean;
  attentionDropout: number;
  mlpBias: boolean;
  expandTo: {
    inputNorm: LlamaRmsNormOp;
    attention: LlamaAttentionOp;
    attentionResidual: ResidualAddOp;
    postAttentionNorm: LlamaRmsNormOp;
    mlp: LlamaMlpOp;
    mlpResidual: ResidualAddOp;
  };
};

export type LlamaFinalNormOp = {
  kind: "llama_final_rms_norm";
  id: string;
  input: string;
  output: string;
  hiddenSize: number;
  epsilon: number;
};

export type LlamaLmHeadOp = {
  kind: "llama_lm_head";
  id: string;
  input: string;
  output: string;
  hiddenSize: number;
  vocabSize: number;
  tiedToEmbedding: string;
};

export type LlamaOperator =
  | LlamaEmbeddingOp
  | LlamaBlockOp
  | LlamaFinalNormOp
  | LlamaLmHeadOp;

export type Phi3EmbeddingOp = LlamaEmbeddingOp;
export type Phi3AttentionOp = LlamaAttentionOp;
export type Phi3MlpOp = LlamaMlpOp;
export type Phi3RmsNormOp = LlamaRmsNormOp;
export type Phi3BlockOp = LlamaBlockOp;
export type Phi3FinalNormOp = LlamaFinalNormOp;
export type Phi3LmHeadOp = LlamaLmHeadOp;
export type Phi3Operator = LlamaOperator;

export type Gemma4EmbeddingOp = LlamaEmbeddingOp;
export type Gemma4AttentionOp = LlamaAttentionOp;
export type Gemma4MlpOp = LlamaMlpOp;
export type Gemma4RmsNormOp = LlamaRmsNormOp;
export type Gemma4BlockOp = LlamaBlockOp;
export type Gemma4FinalNormOp = LlamaFinalNormOp;
export type Gemma4LmHeadOp = LlamaLmHeadOp;
export type Gemma4Operator = LlamaOperator;

export type HybridEmbeddingOp =
  | {
      family: "gpt2";
      kind: "hybrid_embeddings";
      id: string;
      input: string;
      output: string;
      vocabSize: number;
      maxPositionEmbeddings: number;
      hiddenSize: number;
      embdDropout: number;
      tieWordEmbeddings: boolean;
    }
  | {
      family: "llama";
      kind: "hybrid_embeddings";
      id: string;
      input: string;
      output: string;
      vocabSize: number;
      hiddenSize: number;
      tieWordEmbeddings: boolean;
    }
  | {
      family: "phi3";
      kind: "hybrid_embeddings";
      id: string;
      input: string;
      output: string;
      vocabSize: number;
      hiddenSize: number;
      tieWordEmbeddings: boolean;
    };

export type HybridBlockOp =
  | {
      family: "gpt2";
      kind: "hybrid_block";
      id: string;
      input: string;
      output: string;
      hiddenSize: number;
      intermediateSize: number;
      numHeads: number;
      layerNormEpsilon: number;
      activation: string;
      feedforwardType: "mlp" | "moe";
      numExperts: number;
      topK: number;
      expertHidden: number;
      attnDropout: number;
      residDropout: number;
      scaleAttnWeights: boolean;
      scaleAttnByInverseLayerIdx: boolean;
      reorderAndUpcastAttn: boolean;
    }
  | {
      family: "llama";
      kind: "hybrid_block";
      id: string;
      input: string;
      output: string;
      hiddenSize: number;
      intermediateSize: number;
      numHeads: number;
      numKeyValueHeads: number;
      headDim: number;
      rmsNormEpsilon: number;
      ropeTheta: number;
      activation: string;
      feedforwardType: "mlp" | "moe";
      numExperts: number;
      topK: number;
      expertHidden: number;
      attentionBias: boolean;
      attentionDropout: number;
      mlpBias: boolean;
    }
  | {
      family: "phi3";
      kind: "hybrid_block";
      id: string;
      input: string;
      output: string;
      hiddenSize: number;
      intermediateSize: number;
      numHeads: number;
      numKeyValueHeads: number;
      headDim: number;
      rmsNormEpsilon: number;
      ropeTheta: number;
      activation: string;
      feedforwardType: "mlp" | "moe";
      numExperts: number;
      topK: number;
      expertHidden: number;
      attentionBias: boolean;
      attentionDropout: number;
      mlpBias: boolean;
    };

export type HybridFinalNormOp =
  | {
      family: "gpt2";
      kind: "hybrid_final_norm";
      id: string;
      input: string;
      output: string;
      hiddenSize: number;
      epsilon: number;
    }
  | {
      family: "llama";
      kind: "hybrid_final_norm";
      id: string;
      input: string;
      output: string;
      hiddenSize: number;
      epsilon: number;
    }
  | {
      family: "phi3";
      kind: "hybrid_final_norm";
      id: string;
      input: string;
      output: string;
      hiddenSize: number;
      epsilon: number;
    };

export type HybridLmHeadOp =
  | {
      family: "gpt2";
      kind: "hybrid_lm_head";
      id: string;
      input: string;
      output: string;
      hiddenSize: number;
      vocabSize: number;
      tiedToEmbedding: string;
    }
  | {
      family: "llama";
      kind: "hybrid_lm_head";
      id: string;
      input: string;
      output: string;
      hiddenSize: number;
      vocabSize: number;
      tiedToEmbedding: string;
    }
  | {
      family: "phi3";
      kind: "hybrid_lm_head";
      id: string;
      input: string;
      output: string;
      hiddenSize: number;
      vocabSize: number;
      tiedToEmbedding: string;
    };

export type GPT2ArchitectureSpec = {
  family: "gpt2";
  modality: "text";
  task: "causal_lm";
  name: string;
  source?: {
    kind: "manual" | "huggingface-config";
    modelId?: string;
  };
  config: {
    vocabSize: number;
    maxPositionEmbeddings: number;
    hiddenSize: number;
    numHiddenLayers: number;
    numAttentionHeads: number;
    intermediateSize: number;
    activationFunction: string;
    embdDropout: number;
    attnDropout: number;
    residDropout: number;
    layerNormEpsilon: number;
    scaleAttnWeights: boolean;
    scaleAttnByInverseLayerIdx: boolean;
    reorderAndUpcastAttn: boolean;
    tieWordEmbeddings: boolean;
  };
  tensors: {
    inputTokens: TensorRef;
    hiddenStates: TensorRef[];
    logits: TensorRef;
  };
  operators: GPT2Operator[];
};

export type LlamaArchitectureSpec = {
  family: "llama";
  modality: "text";
  task: "causal_lm";
  name: string;
  source?: {
    kind: "manual" | "huggingface-config";
    modelId?: string;
  };
  config: {
    vocabSize: number;
    hiddenSize: number;
    intermediateSize: number;
    numHiddenLayers: number;
    numAttentionHeads: number;
    numKeyValueHeads: number;
    headDim: number;
    hiddenActivation: string;
    maxPositionEmbeddings: number;
    rmsNormEpsilon: number;
    ropeTheta: number;
    attentionBias: boolean;
    attentionDropout: number;
    mlpBias: boolean;
    tieWordEmbeddings: boolean;
  };
  tensors: {
    inputTokens: TensorRef;
    hiddenStates: TensorRef[];
    logits: TensorRef;
  };
  operators: LlamaOperator[];
};

export type Phi3ArchitectureSpec = {
  family: "phi3";
  modality: "text";
  task: "causal_lm";
  name: string;
  source?: {
    kind: "manual" | "huggingface-config";
    modelId?: string;
  };
  config: {
    vocabSize: number;
    hiddenSize: number;
    intermediateSize: number;
    numHiddenLayers: number;
    numAttentionHeads: number;
    numKeyValueHeads: number;
    headDim: number;
    hiddenActivation: string;
    maxPositionEmbeddings: number;
    rmsNormEpsilon: number;
    ropeTheta: number;
    attentionBias: boolean;
    attentionDropout: number;
    mlpBias: boolean;
    tieWordEmbeddings: boolean;
  };
  tensors: {
    inputTokens: TensorRef;
    hiddenStates: TensorRef[];
    logits: TensorRef;
  };
  operators: Phi3Operator[];
};

export type Gemma4ArchitectureSpec = {
  family: "gemma4";
  modality: "text";
  task: "causal_lm";
  name: string;
  source?: {
    kind: "manual" | "huggingface-config";
    modelId?: string;
  };
  config: {
    vocabSize: number;
    hiddenSize: number;
    intermediateSize: number;
    numHiddenLayers: number;
    numAttentionHeads: number;
    numKeyValueHeads: number;
    headDim: number;
    hiddenActivation: string;
    maxPositionEmbeddings: number;
    rmsNormEpsilon: number;
    ropeTheta: number;
    attentionBias: boolean;
    attentionDropout: number;
    mlpBias: boolean;
    tieWordEmbeddings: boolean;
    slidingWindow: number;
    layerTypes: string[];
    ropeParameters: Record<string, { ropeType: string; ropeTheta: number }>;
    numGlobalKeyValueHeads: number | null;
    globalHeadDim: number;
    attentionKEqV: boolean;
    numKvSharedLayers: number;
  };
  tensors: {
    inputTokens: TensorRef;
    hiddenStates: TensorRef[];
    logits: TensorRef;
  };
  operators: Gemma4Operator[];
};

export type HybridDecoderArchitectureSpec = {
  family: "hybrid_decoder";
  modality: "text";
  task: "causal_lm";
  name: string;
  source?: {
    kind: "manual";
  };
  config: {
    vocabSize: number;
    hiddenSize: number;
    maxPositionEmbeddings: number;
    tieWordEmbeddings: boolean;
    embeddingFamily: "gpt2" | "llama" | "phi3";
    finalNormFamily: "gpt2" | "llama" | "phi3";
    blockFamilies: Array<"gpt2" | "llama" | "phi3">;
  };
  tensors: {
    inputTokens: TensorRef;
    hiddenStates: TensorRef[];
    logits: TensorRef;
  };
  operators: {
    embedding: HybridEmbeddingOp;
    blocks: HybridBlockOp[];
    finalNorm: HybridFinalNormOp;
    lmHead: HybridLmHeadOp;
  };
};

export type ArchitectureSpec = GPT2ArchitectureSpec | LlamaArchitectureSpec | Phi3ArchitectureSpec | Gemma4ArchitectureSpec | HybridDecoderArchitectureSpec;
