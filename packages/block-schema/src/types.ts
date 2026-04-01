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

export type BlockNode = {
  id: string;
  type: BlockType;
  name?: string;
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

