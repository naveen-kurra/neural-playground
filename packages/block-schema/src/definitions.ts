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
        description: "Sequence length must be greater than 1.",
        kind: "number_gt",
        field: "sequenceLength",
        min: 1
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
    sequenceDimField: "embeddingDim",
    vocabSizeField: "vocabSize",
    ruleSpecs: [
      {
        code: "invalid_vocab_size",
        severity: "error",
        description: "Vocab size must be positive.",
        kind: "number_gt",
        field: "vocabSize",
        min: 0
      },
      {
        code: "invalid_embedding_dim",
        severity: "error",
        description: "Embedding dimension must be positive.",
        kind: "number_gt",
        field: "embeddingDim",
        min: 0
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
    sequenceDimField: "embeddingDim",
    vocabSizeField: "vocabSize",
    ruleSpecs: [
      {
        code: "invalid_vocab_size",
        severity: "error",
        description: "Vocab size must be positive.",
        kind: "number_gt",
        field: "vocabSize",
        min: 0
      },
      {
        code: "invalid_embedding_dim",
        severity: "error",
        description: "Embedding dimension must be positive.",
        kind: "number_gt",
        field: "embeddingDim",
        min: 0
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
    sequenceDimField: "embeddingDim",
    vocabSizeField: "vocabSize",
    ruleSpecs: [
      {
        code: "invalid_vocab_size",
        severity: "error",
        description: "Vocab size must be positive.",
        kind: "number_gt",
        field: "vocabSize",
        min: 0
      },
      {
        code: "invalid_embedding_dim",
        severity: "error",
        description: "Embedding dimension must be positive.",
        kind: "number_gt",
        field: "embeddingDim",
        min: 0
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
    type: "MistralTokenEmbedding",
    label: "Mistral Token Embedding",
    category: "embedding",
    description: "Mistral token embedding table (`embed_tokens`).",
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
    sequenceDimField: "embeddingDim",
    vocabSizeField: "vocabSize",
    ruleSpecs: [
      {
        code: "invalid_vocab_size",
        severity: "error",
        description: "Vocab size must be positive.",
        kind: "number_gt",
        field: "vocabSize",
        min: 0
      },
      {
        code: "invalid_embedding_dim",
        severity: "error",
        description: "Embedding dimension must be positive.",
        kind: "number_gt",
        field: "embeddingDim",
        min: 0
      }
    ],
    fields: [
      {
        key: "vocabSize",
        label: "Vocab Size",
        type: "number",
        defaultValue: 32768
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
    type: "Gemma4TokenEmbedding",
    label: "Gemma 4 Token Embedding",
    category: "embedding",
    description: "Gemma 4 text token embedding table (`embed_tokens`).",
    inputs: ["tokens"],
    outputs: ["sequence"],
    inputContracts: [{ kind: "tokens", dims: ["seq_len"] }],
    outputContracts: [{ kind: "sequence", dims: ["seq_len", "d_model"] }],
    sequenceDimField: "embeddingDim",
    vocabSizeField: "vocabSize",
    ruleSpecs: [
      { code: "invalid_vocab_size", severity: "error", description: "Vocab size must be positive.", kind: "number_gt", field: "vocabSize", min: 0 },
      { code: "invalid_embedding_dim", severity: "error", description: "Embedding dimension must be positive.", kind: "number_gt", field: "embeddingDim", min: 0 }
    ],
    fields: [
      { key: "vocabSize", label: "Vocab Size", type: "number", defaultValue: 262144 },
      { key: "embeddingDim", label: "Hidden Size", type: "number", defaultValue: 5376 }
    ]
  },
  {
    type: "Phi3TokenEmbedding",
    label: "Phi-3 Token Embedding",
    category: "embedding",
    description: "Phi-3 token embedding table (`embed_tokens`).",
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
    sequenceDimField: "embeddingDim",
    vocabSizeField: "vocabSize",
    ruleSpecs: [
      {
        code: "invalid_vocab_size",
        severity: "error",
        description: "Vocab size must be positive.",
        kind: "number_gt",
        field: "vocabSize",
        min: 0
      },
      {
        code: "invalid_embedding_dim",
        severity: "error",
        description: "Embedding dimension must be positive.",
        kind: "number_gt",
        field: "embeddingDim",
        min: 0
      }
    ],
    fields: [
      {
        key: "vocabSize",
        label: "Vocab Size",
        type: "number",
        defaultValue: 32064
      },
      {
        key: "embeddingDim",
        label: "Hidden Size",
        type: "number",
        defaultValue: 3072
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
    sequenceDimField: "embeddingDim",
    ruleSpecs: [
      {
        code: "invalid_sequence_length",
        severity: "error",
        description: "Maximum positions must be greater than 1.",
        kind: "number_gt",
        field: "sequenceLength",
        min: 1
      },
      {
        code: "invalid_embedding_dim",
        severity: "error",
        description: "Embedding dimension must be positive.",
        kind: "number_gt",
        field: "embeddingDim",
        min: 0
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
    ruleSpecs: [
      {
        code: "invalid_dropout_range",
        severity: "error",
        description: "Dropout must be between 0 and 1.",
        kind: "number_in_range",
        field: "dropout",
        min: 0,
        max: 1
      }
    ],
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
    sequenceDimField: "dModel",
    ruleSpecs: [
      {
        code: "invalid_d_model",
        severity: "error",
        description: "Model dimension must be positive.",
        kind: "number_gt",
        field: "dModel",
        min: 0
      },
      {
        code: "invalid_num_heads",
        severity: "error",
        description: "Number of heads must be positive.",
        kind: "number_gt",
        field: "numHeads",
        min: 0
      },
      {
        code: "heads_dimension_mismatch",
        severity: "error",
        description: "Model dimension must be divisible by the number of heads.",
        kind: "number_divisible",
        field: "dModel",
        otherField: "numHeads"
      },
      {
        code: "invalid_dropout_range",
        severity: "error",
        description: "Dropout must be between 0 and 1.",
        kind: "number_in_range",
        field: "dropout",
        min: 0,
        max: 1
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
        description: "Number of experts must be positive.",
        kind: "number_gt",
        field: "numExperts",
        min: 0
      },
      {
        code: "invalid_top_k",
        severity: "error",
        description: "Top-k must be positive and no greater than the number of experts.",
        kind: "number_lte_field",
        field: "topK",
        otherField: "numExperts",
        min: 0
      },
      {
        code: "invalid_expert_hidden",
        severity: "error",
        description: "Expert hidden size must be positive.",
        kind: "number_gt",
        field: "expertHidden",
        min: 0
      },
      {
        code: "unknown_moe_input_dim",
        severity: "error",
        description: "MoE input dimension could not be inferred from incoming connections.",
        kind: "sequence_dim_known"
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
    sequenceDimField: "dModel",
    ruleSpecs: [
      {
        code: "invalid_d_model",
        severity: "error",
        description: "Model dimension must be positive.",
        kind: "number_gt",
        field: "dModel",
        min: 0
      },
      {
        code: "invalid_num_heads",
        severity: "error",
        description: "Number of heads must be positive.",
        kind: "number_gt",
        field: "numHeads",
        min: 0
      },
      {
        code: "heads_dimension_mismatch",
        severity: "error",
        description: "Model dimension must be divisible by the number of heads.",
        kind: "number_divisible",
        field: "dModel",
        otherField: "numHeads"
      },
      {
        code: "invalid_ffn_hidden",
        severity: "error",
        description: "MLP hidden dimension must be positive.",
        kind: "number_gt",
        field: "ffnHidden",
        min: 0,
        when: { field: "feedforwardType", notEquals: "moe" }
      },
      {
        code: "invalid_layer_norm_epsilon",
        severity: "error",
        description: "LayerNorm epsilon must be positive.",
        kind: "number_gt",
        field: "layerNormEpsilon",
        min: 0
      },
      {
        code: "invalid_dropout_range",
        severity: "error",
        description: "Dropout must be between 0 and 1.",
        kind: "number_in_range",
        field: "dropout",
        min: 0,
        max: 1
      },
      {
        code: "invalid_expert_count",
        severity: "error",
        description: "MoE requires a positive number of experts when feedforward is set to moe.",
        kind: "number_gt",
        field: "numExperts",
        min: 0,
        when: { field: "feedforwardType", equals: "moe" }
      },
      {
        code: "invalid_top_k",
        severity: "error",
        description: "MoE top-k must be positive and no greater than the number of experts.",
        kind: "number_lte_field",
        field: "topK",
        otherField: "numExperts",
        min: 0,
        when: { field: "feedforwardType", equals: "moe" }
      },
      {
        code: "invalid_expert_hidden",
        severity: "error",
        description: "MoE expert hidden size must be positive.",
        kind: "number_gt",
        field: "expertHidden",
        min: 0,
        when: { field: "feedforwardType", equals: "moe" }
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
    sequenceDimField: "dModel",
    ruleSpecs: [
      {
        code: "invalid_d_model",
        severity: "error",
        description: "Model dimension must be positive.",
        kind: "number_gt",
        field: "dModel",
        min: 0
      },
      {
        code: "invalid_num_heads",
        severity: "error",
        description: "Number of heads must be positive.",
        kind: "number_gt",
        field: "numHeads",
        min: 0
      },
      {
        code: "heads_dimension_mismatch",
        severity: "error",
        description: "Model dimension must be divisible by the number of heads.",
        kind: "number_divisible",
        field: "dModel",
        otherField: "numHeads"
      },
      {
        code: "invalid_ffn_hidden",
        severity: "error",
        description: "MLP hidden dimension must be positive.",
        kind: "number_gt",
        field: "ffnHidden",
        min: 0,
        when: { field: "feedforwardType", notEquals: "moe" }
      },
      {
        code: "invalid_layer_norm_epsilon",
        severity: "error",
        description: "LayerNorm epsilon must be positive.",
        kind: "number_gt",
        field: "rmsNormEpsilon",
        min: 0
      },
      {
        code: "invalid_dropout_range",
        severity: "error",
        description: "Dropout must be between 0 and 1.",
        kind: "number_in_range",
        field: "dropout",
        min: 0,
        max: 1
      },
      {
        code: "invalid_expert_count",
        severity: "error",
        description: "MoE requires a positive number of experts when feedforward is set to moe.",
        kind: "number_gt",
        field: "numExperts",
        min: 0,
        when: { field: "feedforwardType", equals: "moe" }
      },
      {
        code: "invalid_top_k",
        severity: "error",
        description: "MoE top-k must be positive and no greater than the number of experts.",
        kind: "number_lte_field",
        field: "topK",
        otherField: "numExperts",
        min: 0,
        when: { field: "feedforwardType", equals: "moe" }
      },
      {
        code: "invalid_expert_hidden",
        severity: "error",
        description: "MoE expert hidden size must be positive.",
        kind: "number_gt",
        field: "expertHidden",
        min: 0,
        when: { field: "feedforwardType", equals: "moe" }
      },
      {
        code: "invalid_kv_heads",
        severity: "error",
        description: "KV heads must be positive, not exceed attention heads, and divide evenly into attention heads.",
        kind: "number_lte_and_divides_field",
        field: "numKeyValueHeads",
        otherField: "numHeads",
        min: 0
      },
      {
        code: "head_dim_mismatch",
        severity: "error",
        description: "Head dim must equal hidden size divided by attention heads.",
        kind: "number_equals_floor_div",
        field: "headDim",
        otherField: "dModel",
        divisorField: "numHeads"
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
    type: "MistralBlock",
    label: "Mistral Block",
    category: "transformer",
    description: "Mistral decoder block with RMSNorm, RoPE attention, grouped-query attention, and configurable feedforward path.",
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
    sequenceDimField: "dModel",
    ruleSpecs: [
      {
        code: "invalid_d_model",
        severity: "error",
        description: "Model dimension must be positive.",
        kind: "number_gt",
        field: "dModel",
        min: 0
      },
      {
        code: "invalid_num_heads",
        severity: "error",
        description: "Number of heads must be positive.",
        kind: "number_gt",
        field: "numHeads",
        min: 0
      },
      {
        code: "heads_dimension_mismatch",
        severity: "error",
        description: "Model dimension must be divisible by the number of heads.",
        kind: "number_divisible",
        field: "dModel",
        otherField: "numHeads"
      },
      {
        code: "invalid_ffn_hidden",
        severity: "error",
        description: "MLP hidden dimension must be positive.",
        kind: "number_gt",
        field: "ffnHidden",
        min: 0,
        when: { field: "feedforwardType", notEquals: "moe" }
      },
      {
        code: "invalid_layer_norm_epsilon",
        severity: "error",
        description: "LayerNorm epsilon must be positive.",
        kind: "number_gt",
        field: "rmsNormEpsilon",
        min: 0
      },
      {
        code: "invalid_dropout_range",
        severity: "error",
        description: "Dropout must be between 0 and 1.",
        kind: "number_in_range",
        field: "dropout",
        min: 0,
        max: 1
      },
      {
        code: "invalid_expert_count",
        severity: "error",
        description: "MoE requires a positive number of experts when feedforward is set to moe.",
        kind: "number_gt",
        field: "numExperts",
        min: 0,
        when: { field: "feedforwardType", equals: "moe" }
      },
      {
        code: "invalid_top_k",
        severity: "error",
        description: "MoE top-k must be positive and no greater than the number of experts.",
        kind: "number_lte_field",
        field: "topK",
        otherField: "numExperts",
        min: 0,
        when: { field: "feedforwardType", equals: "moe" }
      },
      {
        code: "invalid_expert_hidden",
        severity: "error",
        description: "MoE expert hidden size must be positive.",
        kind: "number_gt",
        field: "expertHidden",
        min: 0,
        when: { field: "feedforwardType", equals: "moe" }
      },
      {
        code: "invalid_kv_heads",
        severity: "error",
        description: "KV heads must be positive, not exceed attention heads, and divide evenly into attention heads.",
        kind: "number_lte_and_divides_field",
        field: "numKeyValueHeads",
        otherField: "numHeads",
        min: 0
      },
      {
        code: "head_dim_mismatch",
        severity: "error",
        description: "Head dim must equal hidden size divided by attention heads.",
        kind: "number_equals_floor_div",
        field: "headDim",
        otherField: "dModel",
        divisorField: "numHeads"
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
        defaultValue: 8
      },
      {
        key: "ffnHidden",
        label: "Intermediate Size",
        type: "number",
        defaultValue: 14336
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
        defaultValue: 14336
      },
      {
        key: "ropeTheta",
        label: "RoPE Theta",
        type: "number",
        defaultValue: 1000000
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
        defaultValue: 0.00001
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
    type: "Gemma4Block",
    label: "Gemma 4 Block",
    category: "transformer",
    description: "Gemma 4 text decoder block baseline with RMSNorm, RoPE attention, and Gemma-family defaults.",
    inputs: ["sequence"],
    outputs: ["sequence"],
    inputContracts: [{ kind: "sequence", dims: ["seq_len", "d_model"] }],
    outputContracts: [{ kind: "sequence", dims: ["seq_len", "d_model"] }],
    sequenceDimField: "dModel",
    ruleSpecs: [
      { code: "invalid_d_model", severity: "error", description: "Model dimension must be positive.", kind: "number_gt", field: "dModel", min: 0 },
      { code: "invalid_num_heads", severity: "error", description: "Number of heads must be positive.", kind: "number_gt", field: "numHeads", min: 0 },
      { code: "heads_dimension_mismatch", severity: "error", description: "Model dimension must be divisible by the number of heads.", kind: "number_divisible", field: "dModel", otherField: "numHeads" },
      { code: "invalid_ffn_hidden", severity: "error", description: "MLP hidden dimension must be positive.", kind: "number_gt", field: "ffnHidden", min: 0, when: { field: "feedforwardType", notEquals: "moe" } },
      { code: "invalid_layer_norm_epsilon", severity: "error", description: "LayerNorm epsilon must be positive.", kind: "number_gt", field: "rmsNormEpsilon", min: 0 },
      { code: "invalid_dropout_range", severity: "error", description: "Dropout must be between 0 and 1.", kind: "number_in_range", field: "dropout", min: 0, max: 1 },
      { code: "invalid_expert_count", severity: "error", description: "MoE requires a positive number of experts when feedforward is set to moe.", kind: "number_gt", field: "numExperts", min: 0, when: { field: "feedforwardType", equals: "moe" } },
      { code: "invalid_top_k", severity: "error", description: "MoE top-k must be positive and no greater than the number of experts.", kind: "number_lte_field", field: "topK", otherField: "numExperts", min: 0, when: { field: "feedforwardType", equals: "moe" } },
      { code: "invalid_expert_hidden", severity: "error", description: "MoE expert hidden size must be positive.", kind: "number_gt", field: "expertHidden", min: 0, when: { field: "feedforwardType", equals: "moe" } },
      { code: "invalid_kv_heads", severity: "error", description: "KV heads must be positive, not exceed attention heads, and divide evenly into attention heads.", kind: "number_lte_and_divides_field", field: "numKeyValueHeads", otherField: "numHeads", min: 0 },
      { code: "invalid_global_kv_heads", severity: "error", description: "Global KV heads must be positive and not exceed attention heads.", kind: "number_lte_field", field: "numGlobalKeyValueHeads", otherField: "numHeads", min: 0 },
      { code: "invalid_global_head_dim", severity: "error", description: "Global head dim must be positive.", kind: "number_gt", field: "globalHeadDim", min: 0 },
      { code: "invalid_sliding_window", severity: "error", description: "Sliding window must be positive.", kind: "number_gt", field: "slidingWindow", min: 0 },
      { code: "invalid_kv_shared_layers", severity: "error", description: "KV shared layers must be zero or positive.", kind: "number_gt", field: "numKvSharedLayers", min: -1 }
    ],
    fields: [
      { key: "dModel", label: "Hidden Size", type: "number", defaultValue: 5376 },
      { key: "numHeads", label: "Attention Heads", type: "number", defaultValue: 32 },
      { key: "numKeyValueHeads", label: "KV Heads", type: "number", defaultValue: 16 },
      { key: "numGlobalKeyValueHeads", label: "Global KV Heads", type: "number", defaultValue: 4 },
      { key: "ffnHidden", label: "Intermediate Size", type: "number", defaultValue: 21504 },
      { key: "feedforwardType", label: "Feedforward", type: "select", defaultValue: "mlp", options: ["mlp", "moe"] },
      { key: "numExperts", label: "Experts", type: "number", defaultValue: 8 },
      { key: "topK", label: "Top-K", type: "number", defaultValue: 2 },
      { key: "expertHidden", label: "Expert Hidden", type: "number", defaultValue: 21504 },
      { key: "layerType", label: "Layer Type", type: "select", defaultValue: "sliding_attention", options: ["sliding_attention", "full_attention"] },
      { key: "slidingWindow", label: "Sliding Window", type: "number", defaultValue: 1024 },
      { key: "ropeTheta", label: "RoPE Theta", type: "number", defaultValue: 1000000 },
      { key: "headDim", label: "Head Dim", type: "number", defaultValue: 256 },
      { key: "globalHeadDim", label: "Global Head Dim", type: "number", defaultValue: 512 },
      { key: "rmsNormEpsilon", label: "RMSNorm Eps", type: "number", defaultValue: 0.000001 },
      { key: "activation", label: "Activation", type: "select", defaultValue: "gelu_pytorch_tanh", options: ["silu", "relu", "gelu", "gelu_pytorch_tanh"] },
      { key: "attentionBias", label: "Attention Bias", type: "boolean", defaultValue: false },
      { key: "attentionKEqV", label: "Attention K=V", type: "boolean", defaultValue: false },
      { key: "dropout", label: "Attention Dropout", type: "number", defaultValue: 0 },
      { key: "mlpBias", label: "MLP Bias", type: "boolean", defaultValue: false },
      { key: "numKvSharedLayers", label: "KV Shared Layers", type: "number", defaultValue: 0 }
    ]
  },
  {
    type: "Phi3Block",
    label: "Phi-3 Block",
    category: "transformer",
    description: "Phi-3 decoder block with RMSNorm, RoPE attention, and configurable feedforward path.",
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
    sequenceDimField: "dModel",
    ruleSpecs: [
      {
        code: "invalid_d_model",
        severity: "error",
        description: "Model dimension must be positive.",
        kind: "number_gt",
        field: "dModel",
        min: 0
      },
      {
        code: "invalid_num_heads",
        severity: "error",
        description: "Number of heads must be positive.",
        kind: "number_gt",
        field: "numHeads",
        min: 0
      },
      {
        code: "heads_dimension_mismatch",
        severity: "error",
        description: "Model dimension must be divisible by the number of heads.",
        kind: "number_divisible",
        field: "dModel",
        otherField: "numHeads"
      },
      {
        code: "invalid_ffn_hidden",
        severity: "error",
        description: "MLP hidden dimension must be positive.",
        kind: "number_gt",
        field: "ffnHidden",
        min: 0,
        when: { field: "feedforwardType", notEquals: "moe" }
      },
      {
        code: "invalid_layer_norm_epsilon",
        severity: "error",
        description: "LayerNorm epsilon must be positive.",
        kind: "number_gt",
        field: "rmsNormEpsilon",
        min: 0
      },
      {
        code: "invalid_dropout_range",
        severity: "error",
        description: "Dropout must be between 0 and 1.",
        kind: "number_in_range",
        field: "dropout",
        min: 0,
        max: 1
      },
      {
        code: "invalid_expert_count",
        severity: "error",
        description: "MoE requires a positive number of experts when feedforward is set to moe.",
        kind: "number_gt",
        field: "numExperts",
        min: 0,
        when: { field: "feedforwardType", equals: "moe" }
      },
      {
        code: "invalid_top_k",
        severity: "error",
        description: "MoE top-k must be positive and no greater than the number of experts.",
        kind: "number_lte_field",
        field: "topK",
        otherField: "numExperts",
        min: 0,
        when: { field: "feedforwardType", equals: "moe" }
      },
      {
        code: "invalid_expert_hidden",
        severity: "error",
        description: "MoE expert hidden size must be positive.",
        kind: "number_gt",
        field: "expertHidden",
        min: 0,
        when: { field: "feedforwardType", equals: "moe" }
      },
      {
        code: "invalid_kv_heads",
        severity: "error",
        description: "KV heads must be positive, not exceed attention heads, and divide evenly into attention heads.",
        kind: "number_lte_and_divides_field",
        field: "numKeyValueHeads",
        otherField: "numHeads",
        min: 0
      },
      {
        code: "head_dim_mismatch",
        severity: "error",
        description: "Head dim must equal hidden size divided by attention heads.",
        kind: "number_equals_floor_div",
        field: "headDim",
        otherField: "dModel",
        divisorField: "numHeads"
      }
    ],
    fields: [
      {
        key: "dModel",
        label: "Hidden Size",
        type: "number",
        defaultValue: 3072
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
        defaultValue: 8192
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
        defaultValue: 8192
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
        defaultValue: 96
      },
      {
        key: "rmsNormEpsilon",
        label: "RMSNorm Eps",
        type: "number",
        defaultValue: 0.00001
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
        description: "Hidden dimension must be positive.",
        kind: "number_gt",
        field: "hiddenDim",
        min: 0
      },
      {
        code: "unknown_mlp_input_dim",
        severity: "warning",
        description: "MLP input dimension should be inferable from incoming connections.",
        kind: "sequence_dim_known"
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
        code: "invalid_layer_norm_epsilon",
        severity: "error",
        description: "LayerNorm epsilon must be positive.",
        kind: "number_gt",
        field: "epsilon",
        min: 0
      },
      {
        code: "unknown_layernorm_dim",
        severity: "warning",
        description: "LayerNorm dimension should be inferable from incoming connections.",
        kind: "sequence_dim_known"
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
        code: "invalid_layer_norm_epsilon",
        severity: "error",
        description: "LayerNorm epsilon must be positive.",
        kind: "number_gt",
        field: "epsilon",
        min: 0
      },
      {
        code: "unknown_layernorm_dim",
        severity: "warning",
        description: "LayerNorm dimension should be inferable from incoming connections.",
        kind: "sequence_dim_known"
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
        code: "invalid_layer_norm_epsilon",
        severity: "error",
        description: "LayerNorm epsilon must be positive.",
        kind: "number_gt",
        field: "epsilon",
        min: 0
      },
      {
        code: "unknown_layernorm_dim",
        severity: "warning",
        description: "Final norm dimension should be inferable from incoming connections.",
        kind: "sequence_dim_known"
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
    type: "MistralFinalRMSNorm",
    label: "Mistral Final RMSNorm",
    category: "normalization",
    description: "Final Mistral RMSNorm (`norm`).",
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
        code: "invalid_layer_norm_epsilon",
        severity: "error",
        description: "LayerNorm epsilon must be positive.",
        kind: "number_gt",
        field: "epsilon",
        min: 0
      },
      {
        code: "unknown_layernorm_dim",
        severity: "warning",
        description: "Final norm dimension should be inferable from incoming connections.",
        kind: "sequence_dim_known"
      }
    ],
    fields: [
      {
        key: "epsilon",
        label: "RMSNorm Epsilon",
        type: "number",
        defaultValue: 0.00001
      }
    ]
  },
  {
    type: "Gemma4FinalRMSNorm",
    label: "Gemma 4 Final RMSNorm",
    category: "normalization",
    description: "Final Gemma 4 text RMSNorm (`norm`).",
    inputs: ["sequence"],
    outputs: ["sequence"],
    inputContracts: [{ kind: "sequence", dims: ["seq_len", "d_model"] }],
    outputContracts: [{ kind: "sequence", dims: ["seq_len", "d_model"] }],
    ruleSpecs: [
      { code: "invalid_layer_norm_epsilon", severity: "error", description: "LayerNorm epsilon must be positive.", kind: "number_gt", field: "epsilon", min: 0 },
      { code: "unknown_layernorm_dim", severity: "warning", description: "Final norm dimension should be inferable from incoming connections.", kind: "sequence_dim_known" }
    ],
    fields: [{ key: "epsilon", label: "RMSNorm Epsilon", type: "number", defaultValue: 0.000001 }]
  },
  {
    type: "Phi3FinalRMSNorm",
    label: "Phi-3 Final RMSNorm",
    category: "normalization",
    description: "Final Phi-3 RMSNorm (`norm`).",
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
        code: "invalid_layer_norm_epsilon",
        severity: "error",
        description: "LayerNorm epsilon must be positive.",
        kind: "number_gt",
        field: "epsilon",
        min: 0
      },
      {
        code: "unknown_layernorm_dim",
        severity: "warning",
        description: "Final norm dimension should be inferable from incoming connections.",
        kind: "sequence_dim_known"
      }
    ],
    fields: [
      {
        key: "epsilon",
        label: "RMSNorm Epsilon",
        type: "number",
        defaultValue: 0.00001
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
    vocabSizeField: "vocabSize",
    ruleSpecs: [
      {
        code: "invalid_vocab_size",
        severity: "error",
        description: "Vocab size must be positive.",
        kind: "number_gt",
        field: "vocabSize",
        min: 0
      },
      {
        code: "unknown_output_dim",
        severity: "warning",
        description: "LM head dimension should be inferable from incoming connections.",
        kind: "output_dim_known"
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
    vocabSizeField: "vocabSize",
    ruleSpecs: [
      {
        code: "invalid_vocab_size",
        severity: "error",
        description: "Vocab size must be positive.",
        kind: "number_gt",
        field: "vocabSize",
        min: 0
      },
      {
        code: "unknown_output_dim",
        severity: "warning",
        description: "LM head dimension should be inferable from incoming connections.",
        kind: "output_dim_known"
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
    type: "MistralLMHead",
    label: "Mistral LM Head",
    category: "output",
    description: "Projects Mistral hidden states to vocabulary logits.",
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
    vocabSizeField: "vocabSize",
    ruleSpecs: [
      {
        code: "invalid_vocab_size",
        severity: "error",
        description: "Vocab size must be positive.",
        kind: "number_gt",
        field: "vocabSize",
        min: 0
      },
      {
        code: "unknown_output_dim",
        severity: "warning",
        description: "LM head dimension should be inferable from incoming connections.",
        kind: "output_dim_known"
      }
    ],
    fields: [
      {
        key: "vocabSize",
        label: "Vocab Size",
        type: "number",
        defaultValue: 32768
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
    type: "Gemma4LMHead",
    label: "Gemma 4 LM Head",
    category: "output",
    description: "Projects Gemma 4 text hidden states to vocabulary logits.",
    inputs: ["sequence"],
    outputs: ["logits"],
    inputContracts: [{ kind: "sequence", dims: ["seq_len", "d_model"] }],
    outputContracts: [{ kind: "logits", dims: ["seq_len", "vocab_size"] }],
    vocabSizeField: "vocabSize",
    ruleSpecs: [
      { code: "invalid_vocab_size", severity: "error", description: "Vocab size must be positive.", kind: "number_gt", field: "vocabSize", min: 0 },
      { code: "unknown_output_dim", severity: "warning", description: "LM head dimension should be inferable from incoming connections.", kind: "output_dim_known" }
    ],
    fields: [
      { key: "vocabSize", label: "Vocab Size", type: "number", defaultValue: 262144 },
      { key: "tiedWeights", label: "Tie Weights", type: "boolean", defaultValue: true }
    ]
  },
  {
    type: "Phi3LMHead",
    label: "Phi-3 LM Head",
    category: "output",
    description: "Projects Phi-3 hidden states to vocabulary logits.",
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
    vocabSizeField: "vocabSize",
    ruleSpecs: [
      {
        code: "invalid_vocab_size",
        severity: "error",
        description: "Vocab size must be positive.",
        kind: "number_gt",
        field: "vocabSize",
        min: 0
      },
      {
        code: "unknown_output_dim",
        severity: "warning",
        description: "LM head dimension should be inferable from incoming connections.",
        kind: "output_dim_known"
      }
    ],
    fields: [
      {
        key: "vocabSize",
        label: "Vocab Size",
        type: "number",
        defaultValue: 32064
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
        description: "Output head dimension should be inferable from incoming connections.",
        kind: "output_dim_known",
        when: { field: "headType", equals: "LanguageModel" }
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
