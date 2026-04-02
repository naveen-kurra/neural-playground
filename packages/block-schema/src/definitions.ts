import type { BlockDefinition } from "./contracts";
import type { BlockType } from "./types";

export const blockDefinitions: BlockDefinition[] = [
  {
    type: "Input",
    label: "Input",
    category: "input",
    description: "Entry point for tokenized input sequences.",
    inputs: [],
    outputs: ["tokens"],
    inputContracts: [],
    outputContracts: [
      {
        kind: "tokens",
        dims: ["seq_len"]
      }
    ],
    ruleSpecs: [
      {
        code: "invalid_sequence_length",
        severity: "error",
        description: "Sequence length must be greater than 1."
      }
    ],
    fields: [
      {
        key: "sequenceLength",
        label: "Sequence Length",
        type: "number",
        defaultValue: 1024
      }
    ]
  },
  {
    type: "Embedding",
    label: "Embedding",
    category: "embedding",
    description: "Maps token IDs into dense vector representations.",
    inputs: ["tokens"],
    outputs: ["sequence"],
    inputContracts: [
      {
        kind: "tokens",
        dims: ["seq_len"]
      }
    ],
    outputContracts: [
      {
        kind: "sequence",
        dims: ["seq_len", "d_model"]
      }
    ],
    ruleSpecs: [
      {
        code: "invalid_vocab_size",
        severity: "error",
        description: "Vocab size must be positive."
      },
      {
        code: "invalid_embedding_dim",
        severity: "error",
        description: "Embedding dimension must be positive."
      }
    ],
    fields: [
      {
        key: "vocabSize",
        label: "Vocab Size",
        type: "number",
        defaultValue: 32000
      },
      {
        key: "embeddingDim",
        label: "Embedding Dim",
        type: "number",
        defaultValue: 768
      }
    ]
  },
  {
    type: "GPT2TokenEmbedding",
    label: "GPT-2 Token Embedding",
    category: "embedding",
    description: "GPT-2 token embedding table (`wte`).",
    inputs: ["tokens"],
    outputs: ["sequence"],
    inputContracts: [
      {
        kind: "tokens",
        dims: ["seq_len"]
      }
    ],
    outputContracts: [
      {
        kind: "sequence",
        dims: ["seq_len", "d_model"]
      }
    ],
    ruleSpecs: [
      {
        code: "invalid_vocab_size",
        severity: "error",
        description: "Vocab size must be positive."
      },
      {
        code: "invalid_embedding_dim",
        severity: "error",
        description: "Embedding dimension must be positive."
      }
    ],
    fields: [
      {
        key: "vocabSize",
        label: "Vocab Size",
        type: "number",
        defaultValue: 50257
      },
      {
        key: "embeddingDim",
        label: "Embedding Dim",
        type: "number",
        defaultValue: 768
      }
    ]
  },
  {
    type: "LlamaTokenEmbedding",
    label: "LLaMA Token Embedding",
    category: "embedding",
    description: "LLaMA token embedding table (`embed_tokens`).",
    inputs: ["tokens"],
    outputs: ["sequence"],
    inputContracts: [
      {
        kind: "tokens",
        dims: ["seq_len"]
      }
    ],
    outputContracts: [
      {
        kind: "sequence",
        dims: ["seq_len", "d_model"]
      }
    ],
    ruleSpecs: [
      {
        code: "invalid_vocab_size",
        severity: "error",
        description: "Vocab size must be positive."
      },
      {
        code: "invalid_embedding_dim",
        severity: "error",
        description: "Embedding dimension must be positive."
      }
    ],
    fields: [
      {
        key: "vocabSize",
        label: "Vocab Size",
        type: "number",
        defaultValue: 32000
      },
      {
        key: "embeddingDim",
        label: "Hidden Size",
        type: "number",
        defaultValue: 4096
      }
    ]
  },
  {
    type: "GPT2PositionEmbedding",
    label: "GPT-2 Position Embedding",
    category: "embedding",
    description: "GPT-2 learned position embedding table (`wpe`).",
    inputs: ["tokens"],
    outputs: ["sequence"],
    inputContracts: [
      {
        kind: "tokens",
        dims: ["seq_len"]
      }
    ],
    outputContracts: [
      {
        kind: "sequence",
        dims: ["seq_len", "d_model"]
      }
    ],
    ruleSpecs: [
      {
        code: "invalid_sequence_length",
        severity: "error",
        description: "Maximum positions must be greater than 1."
      },
      {
        code: "invalid_embedding_dim",
        severity: "error",
        description: "Embedding dimension must be positive."
      }
    ],
    fields: [
      {
        key: "sequenceLength",
        label: "Max Positions",
        type: "number",
        defaultValue: 1024
      },
      {
        key: "embeddingDim",
        label: "Embedding Dim",
        type: "number",
        defaultValue: 768
      }
    ]
  },
  {
    type: "Add",
    label: "Add",
    category: "merge",
    description: "Merges matching sequence tensors by elementwise addition.",
    inputs: ["sequence"],
    outputs: ["sequence"],
    inputContracts: [
      {
        kind: "sequence",
        dims: ["seq_len", "d_model"]
      }
    ],
    outputContracts: [
      {
        kind: "sequence",
        dims: ["seq_len", "d_model"]
      }
    ],
    ruleSpecs: [],
    fields: []
  },
  {
    type: "Dropout",
    label: "Dropout",
    category: "regularization",
    description: "Applies dropout to sequence activations.",
    inputs: ["sequence"],
    outputs: ["sequence"],
    inputContracts: [
      {
        kind: "sequence",
        dims: ["seq_len", "d_model"]
      }
    ],
    outputContracts: [
      {
        kind: "sequence",
        dims: ["seq_len", "d_model"]
      }
    ],
    ruleSpecs: [],
    fields: [
      {
        key: "dropout",
        label: "Dropout",
        type: "number",
        defaultValue: 0.1
      }
    ]
  },
  {
    type: "TransformerBlock",
    label: "Transformer Block",
    category: "transformer",
    description: "Pre-LN causal self-attention block with residual path.",
    inputs: ["sequence"],
    outputs: ["sequence"],
    inputContracts: [
      {
        kind: "sequence",
        dims: ["seq_len", "d_model"]
      }
    ],
    outputContracts: [
      {
        kind: "sequence",
        dims: ["seq_len", "d_model"]
      }
    ],
    ruleSpecs: [
      {
        code: "invalid_d_model",
        severity: "error",
        description: "Model dimension must be positive."
      },
      {
        code: "invalid_num_heads",
        severity: "error",
        description: "Number of heads must be positive."
      },
      {
        code: "heads_dimension_mismatch",
        severity: "error",
        description: "Model dimension must be divisible by the number of heads."
      }
    ],
    fields: [
      {
        key: "dModel",
        label: "Model Dim",
        type: "number",
        defaultValue: 768
      },
      {
        key: "numHeads",
        label: "Heads",
        type: "number",
        defaultValue: 12
      },
      {
        key: "dropout",
        label: "Dropout",
        type: "number",
        defaultValue: 0.1
      },
      {
        key: "preLayerNorm",
        label: "Pre-LN",
        type: "boolean",
        defaultValue: true
      }
    ]
  },
  {
    type: "MoE",
    label: "MoE",
    category: "feedforward",
    description: "Top-k routed mixture-of-experts feedforward layer.",
    inputs: ["sequence"],
    outputs: ["sequence"],
    inputContracts: [
      {
        kind: "sequence",
        dims: ["seq_len", "d_model"]
      }
    ],
    outputContracts: [
      {
        kind: "sequence",
        dims: ["seq_len", "d_model"]
      }
    ],
    ruleSpecs: [
      {
        code: "invalid_expert_count",
        severity: "error",
        description: "Number of experts must be positive."
      },
      {
        code: "invalid_top_k",
        severity: "error",
        description: "Top-k must be positive and no greater than the number of experts."
      },
      {
        code: "invalid_expert_hidden",
        severity: "error",
        description: "Expert hidden size must be positive."
      },
      {
        code: "unknown_moe_input_dim",
        severity: "error",
        description: "MoE input dimension could not be inferred from incoming connections."
      }
    ],
    fields: [
      {
        key: "numExperts",
        label: "Experts",
        type: "number",
        defaultValue: 8
      },
      {
        key: "topK",
        label: "Top-K",
        type: "number",
        defaultValue: 2
      },
      {
        key: "expertHidden",
        label: "Expert Hidden",
        type: "number",
        defaultValue: 3072
      },
      {
        key: "activation",
        label: "Activation",
        type: "select",
        defaultValue: "gelu",
        options: ["gelu", "relu", "silu"]
      }
    ]
  },
  {
    type: "GPT2Block",
    label: "GPT-2 Block",
    category: "transformer",
    description: "GPT-2 decoder block with ln_1, masked attention, ln_2, and configurable feedforward residual path.",
    inputs: ["sequence"],
    outputs: ["sequence"],
    inputContracts: [
      {
        kind: "sequence",
        dims: ["seq_len", "d_model"]
      }
    ],
    outputContracts: [
      {
        kind: "sequence",
        dims: ["seq_len", "d_model"]
      }
    ],
    ruleSpecs: [
      {
        code: "invalid_d_model",
        severity: "error",
        description: "Model dimension must be positive."
      },
      {
        code: "invalid_num_heads",
        severity: "error",
        description: "Number of heads must be positive."
      },
      {
        code: "heads_dimension_mismatch",
        severity: "error",
        description: "Model dimension must be divisible by the number of heads."
      },
      {
        code: "invalid_ffn_hidden",
        severity: "error",
        description: "MLP hidden dimension must be positive."
      }
    ],
    fields: [
      {
        key: "dModel",
        label: "Model Dim",
        type: "number",
        defaultValue: 768
      },
      {
        key: "numHeads",
        label: "Heads",
        type: "number",
        defaultValue: 12
      },
      {
        key: "feedforwardType",
        label: "Feedforward",
        type: "select",
        defaultValue: "mlp",
        options: ["mlp", "moe"]
      },
      {
        key: "ffnHidden",
        label: "MLP Hidden",
        type: "number",
        defaultValue: 3072
      },
      {
        key: "numExperts",
        label: "Experts",
        type: "number",
        defaultValue: 8
      },
      {
        key: "topK",
        label: "Top-K",
        type: "number",
        defaultValue: 2
      },
      {
        key: "expertHidden",
        label: "Expert Hidden",
        type: "number",
        defaultValue: 3072
      },
      {
        key: "activation",
        label: "Activation",
        type: "select",
        defaultValue: "gelu_new",
        options: ["gelu_new", "gelu", "relu", "silu"]
      },
      {
        key: "layerNormEpsilon",
        label: "LayerNorm Eps",
        type: "number",
        defaultValue: 0.00001
      },
      {
        key: "dropout",
        label: "Residual Dropout",
        type: "number",
        defaultValue: 0.1
      },
      {
        key: "scaleAttnWeights",
        label: "Scale Attn",
        type: "boolean",
        defaultValue: true
      },
      {
        key: "scaleAttnByInverseLayerIdx",
        label: "Scale By Layer",
        type: "boolean",
        defaultValue: false
      },
      {
        key: "reorderAndUpcastAttn",
        label: "Reorder Upcast",
        type: "boolean",
        defaultValue: false
      }
    ]
  },
  {
    type: "LlamaBlock",
    label: "LLaMA Block",
    category: "transformer",
    description: "LLaMA decoder block with RMSNorm, RoPE attention, and configurable feedforward path.",
    inputs: ["sequence"],
    outputs: ["sequence"],
    inputContracts: [
      {
        kind: "sequence",
        dims: ["seq_len", "d_model"]
      }
    ],
    outputContracts: [
      {
        kind: "sequence",
        dims: ["seq_len", "d_model"]
      }
    ],
    ruleSpecs: [
      {
        code: "invalid_d_model",
        severity: "error",
        description: "Model dimension must be positive."
      },
      {
        code: "invalid_num_heads",
        severity: "error",
        description: "Number of heads must be positive."
      },
      {
        code: "heads_dimension_mismatch",
        severity: "error",
        description: "Model dimension must be divisible by the number of heads."
      },
      {
        code: "invalid_ffn_hidden",
        severity: "error",
        description: "MLP hidden dimension must be positive."
      }
    ],
    fields: [
      {
        key: "dModel",
        label: "Hidden Size",
        type: "number",
        defaultValue: 4096
      },
      {
        key: "numHeads",
        label: "Attention Heads",
        type: "number",
        defaultValue: 32
      },
      {
        key: "numKeyValueHeads",
        label: "KV Heads",
        type: "number",
        defaultValue: 32
      },
      {
        key: "ffnHidden",
        label: "Intermediate Size",
        type: "number",
        defaultValue: 11008
      },
      {
        key: "feedforwardType",
        label: "Feedforward",
        type: "select",
        defaultValue: "mlp",
        options: ["mlp", "moe"]
      },
      {
        key: "numExperts",
        label: "Experts",
        type: "number",
        defaultValue: 8
      },
      {
        key: "topK",
        label: "Top-K",
        type: "number",
        defaultValue: 2
      },
      {
        key: "expertHidden",
        label: "Expert Hidden",
        type: "number",
        defaultValue: 11008
      },
      {
        key: "ropeTheta",
        label: "RoPE Theta",
        type: "number",
        defaultValue: 10000
      },
      {
        key: "headDim",
        label: "Head Dim",
        type: "number",
        defaultValue: 128
      },
      {
        key: "rmsNormEpsilon",
        label: "RMSNorm Eps",
        type: "number",
        defaultValue: 0.000001
      },
      {
        key: "activation",
        label: "Activation",
        type: "select",
        defaultValue: "silu",
        options: ["silu", "relu", "gelu"]
      },
      {
        key: "attentionBias",
        label: "Attention Bias",
        type: "boolean",
        defaultValue: false
      },
      {
        key: "dropout",
        label: "Attention Dropout",
        type: "number",
        defaultValue: 0
      },
      {
        key: "mlpBias",
        label: "MLP Bias",
        type: "boolean",
        defaultValue: false
      }
    ]
  },
  {
    type: "MLP",
    label: "MLP",
    category: "feedforward",
    description: "Two-layer feedforward projection block.",
    inputs: ["sequence"],
    outputs: ["sequence"],
    inputContracts: [
      {
        kind: "sequence",
        dims: ["seq_len", "d_model"]
      }
    ],
    outputContracts: [
      {
        kind: "sequence",
        dims: ["seq_len", "d_model"]
      }
    ],
    ruleSpecs: [
      {
        code: "invalid_hidden_dim",
        severity: "error",
        description: "Hidden dimension must be positive."
      },
      {
        code: "unknown_mlp_input_dim",
        severity: "warning",
        description: "MLP input dimension should be inferable from incoming connections."
      }
    ],
    fields: [
      {
        key: "hiddenDim",
        label: "Hidden Dim",
        type: "number",
        defaultValue: 3072
      },
      {
        key: "activation",
        label: "Activation",
        type: "select",
        defaultValue: "GELU",
        options: ["GELU", "ReLU", "SiLU"]
      }
    ]
  },
  {
    type: "LayerNorm",
    label: "LayerNorm",
    category: "normalization",
    description: "Normalization over the model dimension.",
    inputs: ["sequence"],
    outputs: ["sequence"],
    inputContracts: [
      {
        kind: "sequence",
        dims: ["seq_len", "d_model"]
      }
    ],
    outputContracts: [
      {
        kind: "sequence",
        dims: ["seq_len", "d_model"]
      }
    ],
    ruleSpecs: [
      {
        code: "unknown_layernorm_dim",
        severity: "warning",
        description: "LayerNorm dimension should be inferable from incoming connections."
      }
    ],
    fields: [
      {
        key: "epsilon",
        label: "Epsilon",
        type: "number",
        defaultValue: 0.00001
      }
    ]
  },
  {
    type: "GPT2FinalLayerNorm",
    label: "GPT-2 Final LayerNorm",
    category: "normalization",
    description: "Final GPT-2 layer normalization (`ln_f`).",
    inputs: ["sequence"],
    outputs: ["sequence"],
    inputContracts: [
      {
        kind: "sequence",
        dims: ["seq_len", "d_model"]
      }
    ],
    outputContracts: [
      {
        kind: "sequence",
        dims: ["seq_len", "d_model"]
      }
    ],
    ruleSpecs: [
      {
        code: "unknown_layernorm_dim",
        severity: "warning",
        description: "LayerNorm dimension should be inferable from incoming connections."
      }
    ],
    fields: [
      {
        key: "epsilon",
        label: "Epsilon",
        type: "number",
        defaultValue: 0.00001
      }
    ]
  },
  {
    type: "LlamaFinalRMSNorm",
    label: "LLaMA Final RMSNorm",
    category: "normalization",
    description: "Final LLaMA RMSNorm (`norm`).",
    inputs: ["sequence"],
    outputs: ["sequence"],
    inputContracts: [
      {
        kind: "sequence",
        dims: ["seq_len", "d_model"]
      }
    ],
    outputContracts: [
      {
        kind: "sequence",
        dims: ["seq_len", "d_model"]
      }
    ],
    ruleSpecs: [
      {
        code: "unknown_layernorm_dim",
        severity: "warning",
        description: "Final norm dimension should be inferable from incoming connections."
      }
    ],
    fields: [
      {
        key: "epsilon",
        label: "RMSNorm Epsilon",
        type: "number",
        defaultValue: 0.000001
      }
    ]
  },
  {
    type: "Softmax",
    label: "Softmax",
    category: "activation",
    description: "Converts logits into normalized probabilities.",
    inputs: ["logits"],
    outputs: ["logits"],
    inputContracts: [
      {
        kind: "logits",
        dims: ["seq_len", "vocab_size"]
      }
    ],
    outputContracts: [
      {
        kind: "logits",
        dims: ["seq_len", "vocab_size"]
      }
    ],
    ruleSpecs: [],
    fields: [
      {
        key: "axis",
        label: "Axis",
        type: "number",
        defaultValue: -1
      }
    ]
  },
  {
    type: "GPT2LMHead",
    label: "GPT-2 LM Head",
    category: "output",
    description: "Projects GPT-2 hidden states into token logits, usually tied to `wte`.",
    inputs: ["sequence"],
    outputs: ["logits"],
    inputContracts: [
      {
        kind: "sequence",
        dims: ["seq_len", "d_model"]
      }
    ],
    outputContracts: [
      {
        kind: "logits",
        dims: ["seq_len", "vocab_size"]
      }
    ],
    ruleSpecs: [
      {
        code: "unknown_output_dim",
        severity: "warning",
        description: "LM head dimension should be inferable from incoming connections."
      }
    ],
    fields: [
      {
        key: "vocabSize",
        label: "Vocab Size",
        type: "number",
        defaultValue: 50257
      },
      {
        key: "tiedWeights",
        label: "Tie Weights",
        type: "boolean",
        defaultValue: true
      }
    ]
  },
  {
    type: "LlamaLMHead",
    label: "LLaMA LM Head",
    category: "output",
    description: "Projects LLaMA hidden states to vocabulary logits.",
    inputs: ["sequence"],
    outputs: ["logits"],
    inputContracts: [
      {
        kind: "sequence",
        dims: ["seq_len", "d_model"]
      }
    ],
    outputContracts: [
      {
        kind: "logits",
        dims: ["seq_len", "vocab_size"]
      }
    ],
    ruleSpecs: [
      {
        code: "unknown_output_dim",
        severity: "warning",
        description: "LM head dimension should be inferable from incoming connections."
      }
    ],
    fields: [
      {
        key: "vocabSize",
        label: "Vocab Size",
        type: "number",
        defaultValue: 32000
      },
      {
        key: "tiedWeights",
        label: "Tie Weights",
        type: "boolean",
        defaultValue: false
      }
    ]
  },
  {
    type: "Output",
    label: "Output",
    category: "output",
    description: "Terminal output node for the model graph.",
    inputs: ["sequence", "logits"],
    outputs: [],
    inputContracts: [
      {
        kind: "sequence",
        dims: ["seq_len", "d_model"]
      },
      {
        kind: "logits",
        dims: ["seq_len", "vocab_size"]
      }
    ],
    outputContracts: [],
    ruleSpecs: [
      {
        code: "unknown_output_dim",
        severity: "warning",
        description: "Output head dimension should be inferable from incoming connections."
      }
    ],
    fields: [
      {
        key: "headType",
        label: "Head Type",
        type: "select",
        defaultValue: "LanguageModel",
        options: ["LanguageModel", "Classifier"]
      }
    ]
  }
];

export function getBlockDefinition(type: BlockType): BlockDefinition {
  const definition = blockDefinitions.find((block) => block.type === type);
  if (!definition) {
    throw new Error(`Unknown block type: ${type}`);
  }
  return definition;
}
