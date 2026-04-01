export type BlockCategory =
  | "input"
  | "embedding"
  | "transformer"
  | "feedforward"
  | "normalization"
  | "activation"
  | "output";

export type BlockType =
  | "Input"
  | "Embedding"
  | "TransformerBlock"
  | "MLP"
  | "LayerNorm"
  | "Softmax"
  | "Output";

export type FieldType = "number" | "text" | "select" | "boolean";

export type BlockField = {
  key: string;
  label: string;
  type: FieldType;
  defaultValue: string | number | boolean;
  description?: string;
  options?: string[];
};

export type ShapeArity = "sequence" | "logits" | "tokens";

export type BlockDefinition = {
  type: BlockType;
  label: string;
  category: BlockCategory;
  description: string;
  inputs: ShapeArity[];
  outputs: ShapeArity[];
  fields: BlockField[];
};

export type BlockNode = {
  id: string;
  type: BlockType;
  position: {
    x: number;
    y: number;
  };
  config: Record<string, string | number | boolean>;
};

export type BlockEdge = {
  id: string;
  source: string;
  target: string;
};

export type TrainingConfig = {
  optimizer: "AdamW" | "SGD" | "Custom";
  optimizerCustomName?: string;
  loss: "CrossEntropy" | "Custom";
  lossCustomName?: string;
  learningRate: number;
  activation: "GELU" | "ReLU" | "SiLU" | "Custom";
  activationCustomName?: string;
};

export type ModelGraph = {
  nodes: BlockNode[];
  edges: BlockEdge[];
  training: TrainingConfig;
};

export const blockDefinitions: BlockDefinition[] = [
  {
    type: "Input",
    label: "Input",
    category: "input",
    description: "Entry point for tokenized input sequences.",
    inputs: [],
    outputs: ["tokens"],
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
