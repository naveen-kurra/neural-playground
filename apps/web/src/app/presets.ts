import { getBlockDefinition, type BlockEdge, type BlockNode, type BlockType, type ModelGraph, type TrainingConfig } from "@neural-playground/block-schema";
import { defaultTrainingConfig } from "./defaults";

export type GraphPreset = {
  id: "small-gpt" | "decoder-block" | "text-lm-starter";
  name: string;
  description: string;
  build: () => ModelGraph;
};

function createPresetNode(
  type: BlockType,
  suffix: string,
  position: BlockNode["position"],
  overrides: Partial<BlockNode["config"]> = {}
): BlockNode {
  const definition = getBlockDefinition(type);
  const config = definition.fields.reduce<Record<string, string | number | boolean>>((acc, field) => {
    acc[field.key] = field.defaultValue;
    return acc;
  }, {});

  const sanitizedOverrides = Object.fromEntries(
    Object.entries(overrides).filter(([, value]) => value !== undefined)
  ) as Record<string, string | number | boolean>;

  return {
    id: `${type}-${suffix}`,
    type,
    position,
    config: {
      ...config,
      ...sanitizedOverrides
    }
  };
}

function createEdges(pairs: Array<[string, string]>): BlockEdge[] {
  return pairs.map(([source, target], index) => ({
    id: `edge-preset-${index + 1}`,
    source,
    target
  }));
}

function createTraining(overrides: Partial<TrainingConfig> = {}): TrainingConfig {
  return {
    ...defaultTrainingConfig(),
    ...overrides
  };
}

function buildSmallGptPreset(): ModelGraph {
  const input = createPresetNode("Input", "small-gpt-input", { x: 40, y: 92 }, { sequenceLength: 512 });
  const embedding = createPresetNode("Embedding", "small-gpt-embedding", { x: 280, y: 92 }, { vocabSize: 32000, embeddingDim: 512 });
  const block1 = createPresetNode("TransformerBlock", "small-gpt-block-1", { x: 540, y: 48 }, { dModel: 512, numHeads: 8, ffnHidden: 2048 });
  const block2 = createPresetNode("TransformerBlock", "small-gpt-block-2", { x: 800, y: 136 }, { dModel: 512, numHeads: 8, ffnHidden: 2048 });
  const output = createPresetNode("Output", "small-gpt-output", { x: 1060, y: 92 }, { headType: "LanguageModel" });

  return {
    nodes: [input, embedding, block1, block2, output],
    edges: createEdges([
      [input.id, embedding.id],
      [embedding.id, block1.id],
      [block1.id, block2.id],
      [block2.id, output.id]
    ]),
    training: createTraining({
      learningRate: 0.0002,
      activation: "GELU"
    })
  };
}

function buildDecoderBlockPreset(): ModelGraph {
  const input = createPresetNode("Input", "decoder-block-input", { x: 40, y: 92 }, { sequenceLength: 256 });
  const embedding = createPresetNode("Embedding", "decoder-block-embedding", { x: 280, y: 92 }, { vocabSize: 16000, embeddingDim: 768 });
  const layerNorm = createPresetNode("LayerNorm", "decoder-block-ln", { x: 540, y: 24 }, { epsilon: 0.00001 });
  const block = createPresetNode("TransformerBlock", "decoder-block-core", { x: 540, y: 156 }, { dModel: 768, numHeads: 12, ffnHidden: 3072 });
  const mlp = createPresetNode("MLP", "decoder-block-mlp", { x: 820, y: 92 }, { hiddenDim: 3072, activation: "GELU" });
  const output = createPresetNode("Output", "decoder-block-output", { x: 1080, y: 92 }, { headType: "LanguageModel" });

  return {
    nodes: [input, embedding, layerNorm, block, mlp, output],
    edges: createEdges([
      [input.id, embedding.id],
      [embedding.id, layerNorm.id],
      [layerNorm.id, block.id],
      [block.id, mlp.id],
      [mlp.id, output.id]
    ]),
    training: createTraining({
      learningRate: 0.0003,
      activation: "GELU"
    })
  };
}

function buildTextLmStarterPreset(): ModelGraph {
  const input = createPresetNode("Input", "text-lm-input", { x: 40, y: 92 }, { sequenceLength: 128 });
  const embedding = createPresetNode("Embedding", "text-lm-embedding", { x: 280, y: 92 }, { vocabSize: 12000, embeddingDim: 384 });
  const block = createPresetNode("TransformerBlock", "text-lm-block", { x: 540, y: 92 }, { dModel: 384, numHeads: 6, ffnHidden: 1536 });
  const output = createPresetNode("Output", "text-lm-output", { x: 800, y: 92 }, { headType: "LanguageModel" });

  return {
    nodes: [input, embedding, block, output],
    edges: createEdges([
      [input.id, embedding.id],
      [embedding.id, block.id],
      [block.id, output.id]
    ]),
    training: createTraining({
      optimizer: "AdamW",
      loss: "CrossEntropy",
      learningRate: 0.0005,
      activation: "ReLU"
    })
  };
}

export const graphPresets: GraphPreset[] = [
  {
    id: "small-gpt",
    name: "Small GPT",
    description: "Two decoder blocks with a lightweight language-model setup.",
    build: buildSmallGptPreset
  },
  {
    id: "decoder-block",
    name: "Decoder Block",
    description: "A single decoder-style block with normalization and feedforward layers.",
    build: buildDecoderBlockPreset
  },
  {
    id: "text-lm-starter",
    name: "Text LM Starter",
    description: "Minimal language-model starter graph for quick experiments.",
    build: buildTextLmStarterPreset
  }
];
