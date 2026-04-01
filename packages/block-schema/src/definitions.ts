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
    type: "TransformerBlock",
    label: "Transformer Block",
    category: "transformer",
    description: "Pre-LN self-attention and feedforward block.",
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
        description: "FFN hidden size must be positive."
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
        key: "ffnHidden",
        label: "FFN Hidden",
        type: "number",
        defaultValue: 3072
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
