export type BlockCategory =
  | "input"
  | "embedding"
  | "transformer"
  | "feedforward"
  | "normalization"
  | "merge"
  | "regularization"
  | "activation"
  | "output";

export type BlockType =
  | "Input"
  | "Embedding"
  | "LlamaTokenEmbedding"
  | "GPT2TokenEmbedding"
  | "GPT2PositionEmbedding"
  | "Add"
  | "Dropout"
  | "TransformerBlock"
  | "LlamaBlock"
  | "GPT2Block"
  | "MoE"
  | "MLP"
  | "LayerNorm"
  | "LlamaFinalRMSNorm"
  | "GPT2FinalLayerNorm"
  | "LlamaLMHead"
  | "GPT2LMHead"
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
