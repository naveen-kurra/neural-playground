import { getBlockDefinition, type BlockNode, type BlockType, type TrainingConfig } from "@neural-playground/block-schema";

export function defaultTrainingConfig(): TrainingConfig {
  return {
    optimizer: "AdamW",
    loss: "CrossEntropy",
    learningRate: 0.0003,
    activation: "GELU",
    optimizerCustomName: "",
    lossCustomName: "",
    activationCustomName: ""
  };
}

export function createNode(type: BlockType, index: number): BlockNode {
  const definition = getBlockDefinition(type);
  const config = definition.fields.reduce<Record<string, string | number | boolean>>((acc, field) => {
    acc[field.key] = field.defaultValue;
    return acc;
  }, {});

  return {
    id: `${type}-${crypto.randomUUID().slice(0, 8)}`,
    type,
    position: {
      x: 40 + (index % 3) * 240,
      y: 36 + Math.floor(index / 3) * 148
    },
    config
  };
}
