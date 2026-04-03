import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import type { ModelGraph } from "@neural-playground/block-schema";
import {
  exportGPT2IrProjectFiles,
  exportGPT2IrToPyTorch,
  exportHybridIrProjectFiles,
  exportHybridIrToPyTorch,
  exportLlamaIrProjectFiles,
  exportLlamaIrToPyTorch,
  exportProjectFiles
} from "@neural-playground/exporter-pytorch";
import {
  mapGPT2ConfigToIr,
  mapLlamaConfigToIr,
  mapModelGraphToGPT2Ir,
  mapModelGraphToHybridIr,
  mapModelGraphToLlamaIr
} from "@neural-playground/ir-schema";
import { validateGraph } from "@neural-playground/validator";

import type { HuggingFaceFetchResult } from "../../../apps/web/src/app/huggingface";
import { buildLayerRemap, getEffectiveLayerIndices } from "../../../apps/web/src/app/pruning";
import { buildPruningManifest, buildUpdatedConfig, buildWeightRemapScript } from "../../../apps/web/src/app/pruning-artifacts";
import { validateProjectDocument } from "../../../apps/web/src/app/project";
import { runLocalPrune } from "../../../apps/web/src/app/prune-service";

type TestCase = {
  name: string;
  run: () => void | Promise<void>;
};

const defaultTraining = {
  optimizer: "AdamW",
  learningRate: 3e-4,
  loss: "CrossEntropy",
  activation: "GELU"
} as const;

function buildExactGpt2Graph(): ModelGraph {
  return {
    training: { ...defaultTraining },
    nodes: [
      { id: "input", type: "Input", position: { x: 0, y: 0 }, config: { sequenceLength: 128 } },
      { id: "tok", type: "GPT2TokenEmbedding", position: { x: 100, y: 0 }, config: { vocabSize: 32000, embeddingDim: 768 } },
      { id: "pos", type: "GPT2PositionEmbedding", position: { x: 100, y: 100 }, config: { sequenceLength: 128, embeddingDim: 768 } },
      { id: "add", type: "Add", position: { x: 200, y: 50 }, config: {} },
      { id: "drop", type: "Dropout", position: { x: 300, y: 50 }, config: { dropout: 0.1 } },
      {
        id: "gpt-block-0",
        type: "GPT2Block",
        position: { x: 400, y: 50 },
        config: {
          dModel: 768,
          numHeads: 12,
          ffnHidden: 3072,
          feedforwardType: "mlp",
          numExperts: 8,
          topK: 2,
          expertHidden: 3072,
          activation: "gelu_new",
          layerNormEpsilon: 1e-5,
          dropout: 0.1,
          scaleAttnWeights: true,
          scaleAttnByInverseLayerIdx: false,
          reorderAndUpcastAttn: false
        }
      },
      { id: "final-norm", type: "GPT2FinalLayerNorm", position: { x: 500, y: 50 }, config: { epsilon: 1e-5 } },
      { id: "lm-head", type: "GPT2LMHead", position: { x: 600, y: 50 }, config: { vocabSize: 32000, tiedWeights: true } },
      { id: "output", type: "Output", position: { x: 700, y: 50 }, config: { headType: "LanguageModel" } }
    ],
    edges: [
      { id: "e1", source: "input", target: "tok" },
      { id: "e2", source: "input", target: "pos" },
      { id: "e3", source: "tok", target: "add" },
      { id: "e4", source: "pos", target: "add" },
      { id: "e5", source: "add", target: "drop" },
      { id: "e6", source: "drop", target: "gpt-block-0" },
      { id: "e7", source: "gpt-block-0", target: "final-norm" },
      { id: "e8", source: "final-norm", target: "lm-head" },
      { id: "e9", source: "lm-head", target: "output" }
    ]
  };
}

function buildHybridGraph(): ModelGraph {
  return {
    training: { ...defaultTraining },
    nodes: [
      { id: "input", type: "Input", position: { x: 0, y: 0 }, config: { sequenceLength: 128 } },
      { id: "tok", type: "GPT2TokenEmbedding", position: { x: 100, y: 0 }, config: { vocabSize: 32000, embeddingDim: 768 } },
      { id: "pos", type: "GPT2PositionEmbedding", position: { x: 100, y: 100 }, config: { sequenceLength: 128, embeddingDim: 768 } },
      { id: "add", type: "Add", position: { x: 200, y: 50 }, config: {} },
      { id: "drop", type: "Dropout", position: { x: 300, y: 50 }, config: { dropout: 0.1 } },
      {
        id: "gpt-block-0",
        type: "GPT2Block",
        position: { x: 400, y: 50 },
        config: {
          dModel: 768,
          numHeads: 12,
          ffnHidden: 3072,
          feedforwardType: "mlp",
          numExperts: 8,
          topK: 2,
          expertHidden: 3072,
          activation: "gelu_new",
          layerNormEpsilon: 1e-5,
          dropout: 0.1,
          scaleAttnWeights: true,
          scaleAttnByInverseLayerIdx: false,
          reorderAndUpcastAttn: false
        }
      },
      {
        id: "llama-block-1",
        type: "LlamaBlock",
        position: { x: 500, y: 50 },
        config: {
          dModel: 768,
          numHeads: 12,
          numKeyValueHeads: 12,
          ffnHidden: 3072,
          feedforwardType: "mlp",
          numExperts: 8,
          topK: 2,
          expertHidden: 3072,
          ropeTheta: 10000,
          headDim: 64,
          rmsNormEpsilon: 1e-6,
          activation: "silu",
          attentionBias: false,
          dropout: 0,
          mlpBias: false
        }
      },
      {
        id: "gpt-block-2",
        type: "GPT2Block",
        position: { x: 600, y: 50 },
        config: {
          dModel: 768,
          numHeads: 12,
          ffnHidden: 3072,
          feedforwardType: "mlp",
          numExperts: 8,
          topK: 2,
          expertHidden: 3072,
          activation: "gelu_new",
          layerNormEpsilon: 1e-5,
          dropout: 0.1,
          scaleAttnWeights: true,
          scaleAttnByInverseLayerIdx: false,
          reorderAndUpcastAttn: false
        }
      },
      { id: "final-norm", type: "GPT2FinalLayerNorm", position: { x: 700, y: 50 }, config: { epsilon: 1e-5 } },
      { id: "lm-head", type: "GPT2LMHead", position: { x: 800, y: 50 }, config: { vocabSize: 32000, tiedWeights: true } },
      { id: "output", type: "Output", position: { x: 900, y: 50 }, config: { headType: "LanguageModel" } }
    ],
    edges: [
      { id: "e1", source: "input", target: "tok" },
      { id: "e2", source: "input", target: "pos" },
      { id: "e3", source: "tok", target: "add" },
      { id: "e4", source: "pos", target: "add" },
      { id: "e5", source: "add", target: "drop" },
      { id: "e6", source: "drop", target: "gpt-block-0" },
      { id: "e7", source: "gpt-block-0", target: "llama-block-1" },
      { id: "e8", source: "llama-block-1", target: "gpt-block-2" },
      { id: "e9", source: "gpt-block-2", target: "final-norm" },
      { id: "e10", source: "final-norm", target: "lm-head" },
      { id: "e11", source: "lm-head", target: "output" }
    ]
  };
}

function buildInvalidLlamaGraph(): ModelGraph {
  return {
    training: { ...defaultTraining, activation: "SiLU" },
    nodes: [
      { id: "input", type: "Input", position: { x: 0, y: 0 }, config: { sequenceLength: 128 } },
      { id: "tok", type: "LlamaTokenEmbedding", position: { x: 100, y: 0 }, config: { vocabSize: 32000, embeddingDim: 768 } },
      {
        id: "llama-block-0",
        type: "LlamaBlock",
        position: { x: 200, y: 0 },
        config: {
          dModel: 768,
          numHeads: 12,
          numKeyValueHeads: 12,
          ffnHidden: 3072,
          feedforwardType: "mlp",
          numExperts: 8,
          topK: 2,
          expertHidden: 3072,
          ropeTheta: 10000,
          headDim: 128,
          rmsNormEpsilon: 1e-6,
          activation: "silu",
          attentionBias: false,
          dropout: 0,
          mlpBias: false
        }
      },
      { id: "final-norm", type: "LlamaFinalRMSNorm", position: { x: 300, y: 0 }, config: { epsilon: 1e-6 } },
      { id: "lm-head", type: "LlamaLMHead", position: { x: 400, y: 0 }, config: { vocabSize: 32000, tiedWeights: true } },
      { id: "output", type: "Output", position: { x: 500, y: 0 }, config: { headType: "LanguageModel" } }
    ],
    edges: [
      { id: "e1", source: "input", target: "tok" },
      { id: "e2", source: "tok", target: "llama-block-0" },
      { id: "e3", source: "llama-block-0", target: "final-norm" },
      { id: "e4", source: "final-norm", target: "lm-head" },
      { id: "e5", source: "lm-head", target: "output" }
    ]
  };
}

function buildExactLlamaGraph(): ModelGraph {
  return {
    training: { ...defaultTraining, activation: "SiLU" },
    nodes: [
      { id: "input", type: "Input", position: { x: 0, y: 0 }, config: { sequenceLength: 128 } },
      { id: "tok", type: "LlamaTokenEmbedding", position: { x: 100, y: 0 }, config: { vocabSize: 32000, embeddingDim: 768 } },
      {
        id: "llama-block-0",
        type: "LlamaBlock",
        position: { x: 200, y: 0 },
        config: {
          dModel: 768,
          numHeads: 12,
          numKeyValueHeads: 12,
          ffnHidden: 3072,
          feedforwardType: "mlp",
          numExperts: 8,
          topK: 2,
          expertHidden: 3072,
          ropeTheta: 10000,
          headDim: 64,
          rmsNormEpsilon: 1e-6,
          activation: "silu",
          attentionBias: false,
          dropout: 0,
          mlpBias: false
        }
      },
      { id: "final-norm", type: "LlamaFinalRMSNorm", position: { x: 300, y: 0 }, config: { epsilon: 1e-6 } },
      { id: "lm-head", type: "LlamaLMHead", position: { x: 400, y: 0 }, config: { vocabSize: 32000, tiedWeights: false } },
      { id: "output", type: "Output", position: { x: 500, y: 0 }, config: { headType: "LanguageModel" } }
    ],
    edges: [
      { id: "e1", source: "input", target: "tok" },
      { id: "e2", source: "tok", target: "llama-block-0" },
      { id: "e3", source: "llama-block-0", target: "final-norm" },
      { id: "e4", source: "final-norm", target: "lm-head" },
      { id: "e5", source: "lm-head", target: "output" }
    ]
  };
}

function buildGenericGraph(): ModelGraph {
  return {
    training: { ...defaultTraining },
    nodes: [
      { id: "input", type: "Input", position: { x: 0, y: 0 }, config: { sequenceLength: 128 } },
      { id: "embedding", type: "Embedding", position: { x: 100, y: 0 }, config: { vocabSize: 32000, embeddingDim: 384 } },
      {
        id: "block",
        type: "TransformerBlock",
        position: { x: 200, y: 0 },
        config: {
          dModel: 384,
          numHeads: 6,
          ffnHidden: 1536,
          dropout: 0.1,
          activation: "gelu"
        }
      },
      { id: "output", type: "Output", position: { x: 300, y: 0 }, config: { headType: "LanguageModel", vocabSize: 32000 } }
    ],
    edges: [
      { id: "e1", source: "input", target: "embedding" },
      { id: "e2", source: "embedding", target: "block" },
      { id: "e3", source: "block", target: "output" }
    ]
  };
}

function buildCustomizedExactGpt2Graph(): ModelGraph {
  const graph = buildExactGpt2Graph();
  const block = graph.nodes.find((node) => node.id === "gpt-block-0");
  assert.ok(block);
  block.config.feedforwardType = "moe";
  block.config.numExperts = 4;
  block.config.topK = 2;
  block.config.expertHidden = 3072;
  return graph;
}

function buildCustomizedExactLlamaGraph(): ModelGraph {
  const graph = buildExactLlamaGraph();
  const block = graph.nodes.find((node) => node.id === "llama-block-0");
  assert.ok(block);
  block.config.feedforwardType = "moe";
  block.config.numExperts = 4;
  block.config.topK = 2;
  block.config.expertHidden = 3072;
  return graph;
}

function buildInvalidHybridNoFinalNormGraph(): ModelGraph {
  const graph = buildHybridGraph();
  graph.nodes = graph.nodes.filter((node) => node.id !== "final-norm");
  graph.edges = graph.edges.filter((edge) => edge.target !== "final-norm" && edge.source !== "final-norm");
  return graph;
}

function buildInvalidHybridDualEmbeddingGraph(): ModelGraph {
  const graph = buildHybridGraph();
  graph.nodes.push({
    id: "llama-tok-extra",
    type: "LlamaTokenEmbedding",
    position: { x: 120, y: 200 },
    config: { vocabSize: 32000, embeddingDim: 768 }
  });
  return graph;
}

function buildInvalidHybridHiddenMismatchGraph(): ModelGraph {
  const graph = buildHybridGraph();
  const llamaBlock = graph.nodes.find((node) => node.id === "llama-block-1");
  assert.ok(llamaBlock);
  llamaBlock.config.dModel = 1024;
  return graph;
}

function buildInvalidHybridWrongOutputHeadGraph(): ModelGraph {
  const graph = buildHybridGraph();
  const output = graph.nodes.find((node) => node.id === "output");
  assert.ok(output);
  output.config.headType = "Classification";
  return graph;
}

function buildInvalidEmbeddingInMiddleGraph(): ModelGraph {
  return {
    training: { ...defaultTraining },
    nodes: [
      { id: "input", type: "Input", position: { x: 0, y: 0 }, config: { sequenceLength: 128 } },
      { id: "embedding-a", type: "Embedding", position: { x: 100, y: 0 }, config: { vocabSize: 32000, embeddingDim: 384 } },
      { id: "block", type: "TransformerBlock", position: { x: 200, y: 0 }, config: { dModel: 384, numHeads: 6, ffnHidden: 1536, dropout: 0.1, activation: "gelu" } },
      { id: "embedding-b", type: "Embedding", position: { x: 300, y: 0 }, config: { vocabSize: 32000, embeddingDim: 384 } },
      { id: "output", type: "Output", position: { x: 400, y: 0 }, config: { headType: "LanguageModel" } }
    ],
    edges: [
      { id: "e1", source: "input", target: "embedding-a" },
      { id: "e2", source: "embedding-a", target: "block" },
      { id: "e3", source: "input", target: "embedding-b" },
      { id: "e4", source: "embedding-b", target: "output" }
    ]
  };
}

function buildInvalidHiddenBeforeEmbeddingGraph(): ModelGraph {
  return {
    training: { ...defaultTraining },
    nodes: [
      { id: "input", type: "Input", position: { x: 0, y: 0 }, config: { sequenceLength: 128 } },
      { id: "block", type: "TransformerBlock", position: { x: 100, y: 0 }, config: { dModel: 384, numHeads: 6, ffnHidden: 1536, dropout: 0.1, activation: "gelu" } },
      { id: "embedding", type: "Embedding", position: { x: 200, y: 0 }, config: { vocabSize: 32000, embeddingDim: 384 } },
      { id: "output", type: "Output", position: { x: 300, y: 0 }, config: { headType: "LanguageModel" } }
    ],
    edges: [
      { id: "e1", source: "input", target: "embedding" },
      { id: "e2", source: "embedding", target: "output" }
    ]
  };
}

function buildInvalidHiddenAfterOutputStageGraph(): ModelGraph {
  return {
    training: { ...defaultTraining },
    nodes: [
      { id: "input", type: "Input", position: { x: 0, y: 0 }, config: { sequenceLength: 128 } },
      { id: "embedding", type: "Embedding", position: { x: 100, y: 0 }, config: { vocabSize: 32000, embeddingDim: 384 } },
      { id: "output", type: "Output", position: { x: 200, y: 0 }, config: { headType: "LanguageModel" } },
      { id: "block", type: "TransformerBlock", position: { x: 300, y: 0 }, config: { dModel: 384, numHeads: 6, ffnHidden: 1536, dropout: 0.1, activation: "gelu" } }
    ],
    edges: [
      { id: "e1", source: "input", target: "embedding" },
      { id: "e2", source: "embedding", target: "output" }
    ]
  };
}

function buildInvalidMultipleOutputsGraph(): ModelGraph {
  return {
    training: { ...defaultTraining },
    nodes: [
      { id: "input", type: "Input", position: { x: 0, y: 0 }, config: { sequenceLength: 128 } },
      { id: "embedding", type: "Embedding", position: { x: 100, y: 0 }, config: { vocabSize: 32000, embeddingDim: 384 } },
      { id: "block", type: "TransformerBlock", position: { x: 200, y: 0 }, config: { dModel: 384, numHeads: 6, ffnHidden: 1536, dropout: 0.1, activation: "gelu" } },
      { id: "output-a", type: "Output", position: { x: 300, y: 0 }, config: { headType: "LanguageModel" } },
      { id: "output-b", type: "Output", position: { x: 300, y: 100 }, config: { headType: "LanguageModel" } }
    ],
    edges: [
      { id: "e1", source: "input", target: "embedding" },
      { id: "e2", source: "embedding", target: "block" },
      { id: "e3", source: "block", target: "output-a" }
    ]
  };
}

function buildInvalidAddArityGraph(): ModelGraph {
  return {
    training: { ...defaultTraining },
    nodes: [
      { id: "input", type: "Input", position: { x: 0, y: 0 }, config: { sequenceLength: 128 } },
      { id: "tok", type: "GPT2TokenEmbedding", position: { x: 100, y: 0 }, config: { vocabSize: 32000, embeddingDim: 768 } },
      { id: "pos", type: "GPT2PositionEmbedding", position: { x: 100, y: 100 }, config: { sequenceLength: 128, embeddingDim: 768 } },
      { id: "add", type: "Add", position: { x: 200, y: 50 }, config: {} },
      { id: "output", type: "Output", position: { x: 300, y: 50 }, config: { headType: "LanguageModel" } }
    ],
    edges: [
      { id: "e1", source: "input", target: "tok" },
      { id: "e2", source: "tok", target: "add" },
      { id: "e3", source: "add", target: "output" }
    ]
  };
}

function buildInvalidSoftmaxAxisGraph(): ModelGraph {
  return {
    training: { ...defaultTraining },
    nodes: [
      { id: "input", type: "Input", position: { x: 0, y: 0 }, config: { sequenceLength: 128 } },
      { id: "embedding", type: "Embedding", position: { x: 100, y: 0 }, config: { vocabSize: 32000, embeddingDim: 384 } },
      { id: "output", type: "Output", position: { x: 200, y: 0 }, config: { headType: "LanguageModel" } },
      { id: "softmax", type: "Softmax", position: { x: 300, y: 0 }, config: { axis: 0 } }
    ],
    edges: [
      { id: "e1", source: "input", target: "embedding" }
    ]
  };
}

function buildClassifierOutputGraph(): ModelGraph {
  return {
    training: { ...defaultTraining },
    nodes: [
      { id: "input", type: "Input", position: { x: 0, y: 0 }, config: { sequenceLength: 128 } },
      { id: "embedding", type: "Embedding", position: { x: 100, y: 0 }, config: { vocabSize: 32000, embeddingDim: 384 } },
      { id: "block", type: "TransformerBlock", position: { x: 200, y: 0 }, config: { dModel: 384, numHeads: 6, ffnHidden: 1536, dropout: 0.1, activation: "gelu" } },
      { id: "output", type: "Output", position: { x: 300, y: 0 }, config: { headType: "Classifier" } }
    ],
    edges: [
      { id: "e1", source: "input", target: "embedding" },
      { id: "e2", source: "embedding", target: "block" },
      { id: "e3", source: "block", target: "output" }
    ]
  };
}

function buildInvalidTransformerMultiInputGraph(): ModelGraph {
  return {
    training: { ...defaultTraining },
    nodes: [
      { id: "input", type: "Input", position: { x: 0, y: 0 }, config: { sequenceLength: 128 } },
      { id: "tok", type: "GPT2TokenEmbedding", position: { x: 100, y: 0 }, config: { vocabSize: 32000, embeddingDim: 768 } },
      { id: "pos", type: "GPT2PositionEmbedding", position: { x: 100, y: 100 }, config: { sequenceLength: 128, embeddingDim: 768 } },
      {
        id: "block",
        type: "TransformerBlock",
        position: { x: 220, y: 50 },
        config: { dModel: 768, numHeads: 12, ffnHidden: 3072, dropout: 0.1, activation: "gelu" }
      },
      { id: "output", type: "Output", position: { x: 340, y: 50 }, config: { headType: "LanguageModel" } }
    ],
    edges: [
      { id: "e1", source: "input", target: "tok" },
      { id: "e2", source: "input", target: "pos" },
      { id: "e3", source: "tok", target: "block" },
      { id: "e4", source: "pos", target: "block" },
      { id: "e5", source: "block", target: "output" }
    ]
  };
}

function buildInvalidOutputMultiInputGraph(): ModelGraph {
  return {
    training: { ...defaultTraining },
    nodes: [
      { id: "input", type: "Input", position: { x: 0, y: 0 }, config: { sequenceLength: 128 } },
      { id: "embedding", type: "Embedding", position: { x: 100, y: 0 }, config: { vocabSize: 32000, embeddingDim: 384 } },
      { id: "block-a", type: "TransformerBlock", position: { x: 200, y: 0 }, config: { dModel: 384, numHeads: 6, ffnHidden: 1536, dropout: 0.1, activation: "gelu" } },
      { id: "block-b", type: "TransformerBlock", position: { x: 200, y: 100 }, config: { dModel: 384, numHeads: 6, ffnHidden: 1536, dropout: 0.1, activation: "gelu" } },
      { id: "output", type: "Output", position: { x: 320, y: 50 }, config: { headType: "LanguageModel" } }
    ],
    edges: [
      { id: "e1", source: "input", target: "embedding" },
      { id: "e2", source: "embedding", target: "block-a" },
      { id: "e3", source: "embedding", target: "block-b" },
      { id: "e4", source: "block-a", target: "output" },
      { id: "e5", source: "block-b", target: "output" }
    ]
  };
}

function buildInvalidTransformerBranchGraph(): ModelGraph {
  return {
    training: { ...defaultTraining },
    nodes: [
      { id: "input", type: "Input", position: { x: 0, y: 0 }, config: { sequenceLength: 128 } },
      { id: "embedding", type: "Embedding", position: { x: 100, y: 0 }, config: { vocabSize: 32000, embeddingDim: 384 } },
      { id: "block", type: "TransformerBlock", position: { x: 200, y: 0 }, config: { dModel: 384, numHeads: 6, ffnHidden: 1536, dropout: 0.1, activation: "gelu" } },
      { id: "norm", type: "LayerNorm", position: { x: 320, y: 0 }, config: { epsilon: 0.00001 } },
      { id: "mlp", type: "MLP", position: { x: 320, y: 100 }, config: { hiddenDim: 1536, activation: "GELU" } },
      { id: "output", type: "Output", position: { x: 440, y: 0 }, config: { headType: "LanguageModel" } }
    ],
    edges: [
      { id: "e1", source: "input", target: "embedding" },
      { id: "e2", source: "embedding", target: "block" },
      { id: "e3", source: "block", target: "norm" },
      { id: "e4", source: "block", target: "mlp" },
      { id: "e5", source: "norm", target: "output" }
    ]
  };
}

function buildFetchResultFixture(): HuggingFaceFetchResult {
  return {
    modelId: "microsoft/Phi-3-mini-4k-instruct",
    resolvedFamily: "unknown",
    config: {
      model_type: "phi3",
      num_hidden_layers: 32,
      vocab_size: 32064
    },
    weightIndex: {
      weight_map: {
        "model.layers.0.input_layernorm.weight": "model-00001-of-00002.safetensors"
      }
    },
    inspection: {
      layerCountHint: 32,
      layerCountKey: "num_hidden_layers",
      detectedLayerPrefix: "model.layers.",
      detectedLayerIndices: Array.from({ length: 32 }, (_, index) => index),
      broadPruningSupported: true,
      sampleLayerKeys: [
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.mlp.down_proj.weight"
      ]
    }
  };
}

function compilePythonSource(fileName: string, source: string): void {
  const dir = mkdtempSync(join(tmpdir(), "neural-playground-py-"));
  const filePath = join(dir, fileName);
  try {
    writeFileSync(filePath, source, "utf-8");
    execFileSync("/usr/bin/python3", ["-m", "py_compile", filePath], { stdio: "pipe" });
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

function testValidateProjectDocumentAcceptsRawGraphAndWrappedProject(): void {
  const graph = buildExactGpt2Graph();
  const fromRaw = validateProjectDocument(graph);
  assert.equal(fromRaw.version, 1);
  assert.deepEqual(fromRaw.graph, graph);

  const fromWrapped = validateProjectDocument({ version: 1, graph });
  assert.equal(fromWrapped.version, 1);
  assert.deepEqual(fromWrapped.graph, graph);
}

function testValidateGraphRejectsInvalidLlamaHeadDim(): void {
  const issues = validateGraph(buildInvalidLlamaGraph());
  const mismatch = issues.find((issue) => issue.code === "head_dim_mismatch");
  assert.ok(mismatch, "expected head_dim_mismatch issue");
  assert.match(mismatch.message, /Head dim must equal hidden size divided by attention heads/i);
}

function testValidateGraphRejectsEmbeddingStageInMiddle(): void {
  const issues = validateGraph(buildInvalidEmbeddingInMiddleGraph(), "pytorch-export-valid");
  const mismatch = issues.find((issue) => issue.code === "export_embedding_count");
  assert.ok(mismatch, "expected export_embedding_count issue");
  assert.match(mismatch.message, /requires exactly one Embedding node/i);
}

function testValidateGraphRejectsHiddenBeforeEmbedding(): void {
  const issues = validateGraph(buildInvalidHiddenBeforeEmbeddingGraph(), "pytorch-export-valid");
  const mismatch = issues.find((issue) => issue.code === "hidden_before_embedding");
  assert.ok(mismatch, "expected hidden_before_embedding issue");
  assert.match(mismatch.message, /must come after an embedding stage/i);
}

function testValidateGraphRejectsMultipleOutputs(): void {
  const issues = validateGraph(buildInvalidMultipleOutputsGraph(), "pytorch-export-valid");
  const mismatch = issues.find((issue) => issue.code === "export_output_count");
  assert.ok(mismatch, "expected export_output_count issue");
  assert.match(mismatch.message, /requires exactly one Output node/i);
}

function testValidateGraphRejectsInvalidAddArity(): void {
  const issues = validateGraph(buildInvalidAddArityGraph(), "pytorch-export-valid");
  const mismatch = issues.find((issue) => issue.code === "add_input_arity");
  assert.ok(mismatch, "expected add_input_arity issue");
  assert.match(mismatch.message, /Add requires exactly two incoming sequence connections/i);
}

function testValidateGraphRejectsInvalidSoftmaxAxis(): void {
  const issues = validateGraph(buildInvalidSoftmaxAxisGraph(), "pytorch-export-valid");
  const mismatch = issues.find((issue) => issue.code === "softmax_axis_invalid");
  assert.ok(mismatch, "expected softmax_axis_invalid issue");
  assert.match(mismatch.message, /Softmax must operate on the last logits dimension/i);
}

function testValidateGraphRejectsClassifierOutputExport(): void {
  const issues = validateGraph(buildClassifierOutputGraph(), "pytorch-export-valid");
  const mismatch = issues.find((issue) => issue.code === "classifier_output_unimplemented");
  assert.ok(mismatch, "expected classifier_output_unimplemented issue");
  assert.match(mismatch.message, /Classifier output mode is not implemented yet/i);
}

function testValidateGraphRejectsTransformerMultiInput(): void {
  const issues = validateGraph(buildInvalidTransformerMultiInputGraph(), "pytorch-export-valid");
  const mismatch = issues.find((issue) => issue.code === "node_input_arity");
  assert.ok(mismatch, "expected node_input_arity issue");
  assert.match(mismatch.message, /TransformerBlock allows exactly 1 incoming connection but found 2/i);
}

function testValidateGraphRejectsOutputMultiInput(): void {
  const issues = validateGraph(buildInvalidOutputMultiInputGraph(), "pytorch-export-valid");
  const mismatch = issues.find((issue) => issue.code === "node_input_arity");
  assert.ok(mismatch, "expected node_input_arity issue");
  assert.match(mismatch.message, /Output allows exactly 1 incoming connection but found 2/i);
}

function testValidateGraphRejectsTransformerBranching(): void {
  const issues = validateGraph(buildInvalidTransformerBranchGraph(), "pytorch-export-valid");
  const mismatch = issues.find((issue) => issue.code === "node_output_arity");
  assert.ok(mismatch, "expected node_output_arity issue");
  assert.match(mismatch.message, /TransformerBlock allows exactly 1 outgoing connection but found 2/i);
}

function testGpt2ProjectExportUsesSharedTrainEntrypoint(): void {
  const graph = buildExactGpt2Graph();
  const spec = mapModelGraphToGPT2Ir(graph);
  const files = exportGPT2IrProjectFiles(spec, graph.training);

  assert.ok(files["scripts/train.py"]?.includes("[train] loading configs..."));
  assert.ok(files["scripts/train.py"]?.includes("build_model(cfg, seq_len_override=args.seq_len)"));
  assert.ok(files["src/kurra_ai_cb/model.py"]?.includes("def build_model(cfg, seq_len_override: int | None = None) -> GPT2LMHeadModel:"));
  assert.ok(files["src/kurra_ai_cb/train.py"]?.includes("scaled_loss = loss / grad_accum_steps"));
}

function testHybridProjectExportUsesSharedTrainEntrypoint(): void {
  const graph = buildHybridGraph();
  const spec = mapModelGraphToHybridIr(graph);
  const files = exportHybridIrProjectFiles(spec, graph.training);

  assert.deepEqual(spec.config.blockFamilies, ["gpt2", "llama", "gpt2"]);
  assert.ok(files["scripts/train.py"]?.includes("[train] loading configs..."));
  assert.ok(files["src/kurra_ai_cb/model.py"]?.includes("def build_model(cfg, seq_len_override: int | None = None) -> HybridForCausalLM:"));
  assert.ok(files["src/kurra_ai_cb/model.py"]?.includes("allowed_keys = {"));
  assert.ok(files["src/kurra_ai_cb/model.py"]?.includes("position_embeddings"));
}

function testDirectModelExportsContainStableGoldenMarkers(): void {
  const gpt2Graph = buildExactGpt2Graph();
  const gpt2Spec = mapModelGraphToGPT2Ir(gpt2Graph);
  const gpt2Model = exportGPT2IrToPyTorch(gpt2Spec);
  assert.ok(gpt2Model.includes("class GPT2Config:"));
  assert.ok(gpt2Model.includes("class GPT2Model(nn.Module):"));
  assert.ok(gpt2Model.includes("class GPT2LMHeadModel(nn.Module):"));
  assert.ok(gpt2Model.includes("self.h = nn.ModuleList(["));
  assert.ok(gpt2Model.includes("self.lm_head.weight = self.transformer.wte.weight"));

  const llamaGraph = buildExactLlamaGraph();
  const llamaSpec = mapModelGraphToLlamaIr(llamaGraph);
  const llamaModel = exportLlamaIrToPyTorch(llamaSpec);
  assert.ok(llamaModel.includes("class LlamaConfig:"));
  assert.ok(llamaModel.includes("class LlamaModel(nn.Module):"));
  assert.ok(llamaModel.includes("class LlamaForCausalLM(nn.Module):"));
  assert.ok(llamaModel.includes("self.rotary_emb = LlamaRotaryEmbedding"));
  assert.ok(llamaModel.includes("position_embeddings = self.rotary_emb(hidden_states, position_ids)"));

  const hybridGraph = buildHybridGraph();
  const hybridSpec = mapModelGraphToHybridIr(hybridGraph);
  const hybridModel = exportHybridIrToPyTorch(hybridSpec);
  assert.ok(hybridModel.includes("class HybridConfig:"));
  assert.ok(hybridModel.includes("class HybridDecoderModel(nn.Module):"));
  assert.ok(hybridModel.includes("class HybridForCausalLM(nn.Module):"));
  assert.ok(hybridModel.includes("self.blocks = nn.ModuleList(["));
  assert.ok(hybridModel.includes("LlamaDecoderLayer("));
  assert.ok(hybridModel.includes("GPT2Block("));
}

function testConfigImportMappersProduceExpectedIr(): void {
  const gpt2 = mapGPT2ConfigToIr(
    {
      vocab_size: 32000,
      n_positions: 2048,
      n_embd: 768,
      n_layer: 16,
      n_head: 12,
      n_inner: 3072,
      activation_function: "gelu_new",
      embd_pdrop: 0.05,
      attn_pdrop: 0.1,
      resid_pdrop: 0.15,
      layer_norm_epsilon: 1e-5,
      scale_attn_weights: true,
      scale_attn_by_inverse_layer_idx: false,
      reorder_and_upcast_attn: false,
      tie_word_embeddings: true
    },
    { modelId: "openai-community/gpt2", name: "GPT2 import" }
  );
  assert.equal(gpt2.config.vocabSize, 32000);
  assert.equal(gpt2.config.maxPositionEmbeddings, 2048);
  assert.equal(gpt2.config.numHiddenLayers, 16);
  assert.equal(gpt2.config.numAttentionHeads, 12);
  assert.equal(gpt2.config.tieWordEmbeddings, true);

  const llama = mapLlamaConfigToIr(
    {
      vocab_size: 32000,
      hidden_size: 4096,
      intermediate_size: 11008,
      num_hidden_layers: 32,
      num_attention_heads: 32,
      num_key_value_heads: 8,
      hidden_act: "silu",
      max_position_embeddings: 8192,
      rms_norm_eps: 1e-6,
      rope_scaling: { rope_theta: 500000 },
      attention_bias: false,
      attention_dropout: 0,
      mlp_bias: false,
      tie_word_embeddings: false
    },
    { modelId: "meta-llama/Llama", name: "LLaMA import" }
  );
  assert.equal(llama.config.vocabSize, 32000);
  assert.equal(llama.config.numHiddenLayers, 32);
  assert.equal(llama.config.numKeyValueHeads, 8);
  assert.equal(llama.config.headDim, 128);
  assert.equal(llama.config.ropeTheta, 500000);
}

function testLlamaProjectExportUsesSharedTrainEntrypoint(): void {
  const graph = buildExactLlamaGraph();
  const spec = mapModelGraphToLlamaIr(graph);
  const files = exportLlamaIrProjectFiles(spec, graph.training);

  assert.equal(spec.config.headDim, 64);
  assert.ok(files["scripts/train.py"]?.includes("[train] loading configs..."));
  assert.ok(files["scripts/train.py"]?.includes("build_model(cfg, seq_len_override=args.seq_len)"));
  assert.ok(files["src/kurra_ai_cb/model.py"]?.includes("def build_model(cfg, seq_len_override: int | None = None) -> LlamaForCausalLM:"));
  assert.ok(files["src/kurra_ai_cb/model.py"]?.includes("head_dim=model_cfg.head_dim"));
}

function testGenericProjectExportUsesSharedTrainEntrypoint(): void {
  const graph = buildGenericGraph();
  const files = exportProjectFiles(graph);

  assert.ok(files["scripts/train.py"]?.includes("[train] loading configs..."));
  assert.ok(files["scripts/train.py"]?.includes("build_model(cfg, seq_len_override=args.seq_len)"));
  assert.ok(files["src/kurra_ai_cb/model.py"]?.includes("def build_model(cfg, seq_len_override: int | None = None) -> DecoderLM:"));
  assert.ok(files["src/kurra_ai_cb/train.py"]?.includes("raise ValueError(\"Sequence length must be at least 2\")"));
}

function testCustomizedExactGpt2GraphFallsThroughExactMapper(): void {
  const graph = buildCustomizedExactGpt2Graph();
  assert.throws(
    () => mapModelGraphToGPT2Ir(graph),
    /Customized GPT-2 block internals are exported through the hybrid path\./
  );
}

function testCustomizedExactLlamaGraphFallsThroughExactMapper(): void {
  const graph = buildCustomizedExactLlamaGraph();
  assert.throws(
    () => mapModelGraphToLlamaIr(graph),
    /Customized LLaMA block internals are exported through the hybrid path\./
  );
}

function testHybridMapperRejectsMissingFinalNormGracefully(): void {
  assert.throws(
    () => mapModelGraphToHybridIr(buildInvalidHybridNoFinalNormGraph()),
    /Hybrid export requires a final normalization stage\./
  );
}

function testHybridMapperRejectsDualEmbeddingsGracefully(): void {
  assert.throws(
    () => mapModelGraphToHybridIr(buildInvalidHybridDualEmbeddingGraph()),
    /Hybrid export requires exactly one exact embedding stage\./
  );
}

function testHybridMapperRejectsHiddenMismatchGracefully(): void {
  assert.throws(
    () => mapModelGraphToHybridIr(buildInvalidHybridHiddenMismatchGraph()),
    /Hybrid export requires consistent hidden size across blocks\./
  );
}

function testHybridMapperRejectsWrongOutputHeadGracefully(): void {
  assert.throws(
    () => mapModelGraphToHybridIr(buildInvalidHybridWrongOutputHeadGraph()),
    /Hybrid export requires Output head type LanguageModel\./
  );
}

function testPruningHelpersBuildStableRemapAndConfig(): void {
  const result = buildFetchResultFixture();
  const effective = getEffectiveLayerIndices(result.inspection.detectedLayerIndices, result.inspection.layerCountHint);
  assert.equal(effective.length, 32);

  const kept = effective.filter((index) => ![4, 7, 10, 31].includes(index));
  const dropped = effective.filter((index) => !kept.includes(index));
  const remap = buildLayerRemap(kept);

  assert.deepEqual(remap.slice(0, 5), [
    { newIndex: 0, oldIndex: 0 },
    { newIndex: 1, oldIndex: 1 },
    { newIndex: 2, oldIndex: 2 },
    { newIndex: 3, oldIndex: 3 },
    { newIndex: 4, oldIndex: 5 }
  ]);

  const updatedConfig = buildUpdatedConfig(result, remap);
  assert.equal(updatedConfig.num_hidden_layers, 28);

  const manifest = buildPruningManifest(result, kept, dropped, remap);
  assert.equal(manifest.originalLayerCount, 32);
  assert.equal(manifest.detectedLayerPrefix, "model.layers.");
  assert.deepEqual(manifest.droppedLayerIndices, [4, 7, 10, 31]);

  const script = buildWeightRemapScript(manifest);
  assert.ok(script.includes('LAYER_PREFIX = "model.layers."'));
  assert.ok(script.includes('LAYER_COUNT_KEY = "num_hidden_layers"'));
  assert.ok(script.includes("config[LAYER_COUNT_KEY] = len(REMAP)"));
}

function testPruningHelpersHandleInspectionEdgeCases(): void {
  assert.deepEqual(getEffectiveLayerIndices([], 4), [0, 1, 2, 3]);
  assert.deepEqual(getEffectiveLayerIndices([], null), []);

  const result: HuggingFaceFetchResult = {
    modelId: "unknown/model",
    resolvedFamily: "unknown",
    config: { hidden_size: 768 },
    weightIndex: null,
    inspection: {
      layerCountHint: null,
      layerCountKey: null,
      detectedLayerPrefix: null,
      detectedLayerIndices: [],
      broadPruningSupported: false,
      sampleLayerKeys: []
    }
  };

  const remap = buildLayerRemap([0, 2, 4]);
  const updatedConfig = buildUpdatedConfig(result, remap);
  assert.equal(updatedConfig.hidden_size, 768);
  assert.equal("num_hidden_layers" in updatedConfig, false);

  const manifest = buildPruningManifest(result, [0, 2, 4], [1, 3], remap);
  assert.equal(manifest.originalLayerCount, 0);
  assert.equal(manifest.detectedLayerPrefix, null);

  const script = buildWeightRemapScript(manifest);
  assert.ok(script.includes('LAYER_PREFIX = ""'));
  assert.ok(script.includes('LAYER_COUNT_KEY = ""'));
}

async function testPruneServiceThrowsOnNdjsonErrorEvent(): Promise<void> {
  const ndjson =
    JSON.stringify({ type: "log", stage: "download", message: "Downloading model snapshot from Hugging Face." }) +
    "\n" +
    JSON.stringify({ type: "error", error: "Cannot access gated repo for model google/gemma-3-4b-it." }) +
    "\n";

  const originalFetch = globalThis.fetch;
  (globalThis as Record<string, unknown>).fetch = async () => ({
    ok: true,
    body: new ReadableStream({
      start(controller: ReadableStreamDefaultController) {
        controller.enqueue(new TextEncoder().encode(ndjson));
        controller.close();
      }
    })
  });

  try {
    let thrownError: Error | null = null;
    try {
      await runLocalPrune({ outputDir: "/tmp/out", modelId: "google/gemma-3-4b-it", droppedLayerIndices: [1, 3] });
    } catch (err) {
      thrownError = err instanceof Error ? err : new Error(String(err));
    }
    assert.ok(thrownError !== null, "runLocalPrune should throw when service returns an error event");
    assert.equal(thrownError!.message, "Cannot access gated repo for model google/gemma-3-4b-it.");
  } finally {
    (globalThis as Record<string, unknown>).fetch = originalFetch;
  }
}

function testGeneratedPythonCompilesForSupportedExports(): void {
  const gpt2Graph = buildExactGpt2Graph();
  const gpt2Spec = mapModelGraphToGPT2Ir(gpt2Graph);
  const gpt2Files = exportGPT2IrProjectFiles(gpt2Spec, gpt2Graph.training);
  compilePythonSource("gpt2_model.py", exportGPT2IrToPyTorch(gpt2Spec));
  compilePythonSource("gpt2_train_script.py", gpt2Files["scripts/train.py"]);

  const llamaGraph = buildExactLlamaGraph();
  const llamaSpec = mapModelGraphToLlamaIr(llamaGraph);
  const llamaFiles = exportLlamaIrProjectFiles(llamaSpec, llamaGraph.training);
  compilePythonSource("llama_model.py", exportLlamaIrToPyTorch(llamaSpec));
  compilePythonSource("llama_train_script.py", llamaFiles["scripts/train.py"]);

  const hybridGraph = buildHybridGraph();
  const hybridSpec = mapModelGraphToHybridIr(hybridGraph);
  const hybridFiles = exportHybridIrProjectFiles(hybridSpec, hybridGraph.training);
  compilePythonSource("hybrid_model.py", exportHybridIrToPyTorch(hybridSpec));
  compilePythonSource("hybrid_train_script.py", hybridFiles["scripts/train.py"]);
}

const tests: TestCase[] = [
  {
    name: "validateProjectDocument accepts raw and wrapped project JSON",
    run: testValidateProjectDocumentAcceptsRawGraphAndWrappedProject
  },
  {
    name: "validateGraph reports friendly LLaMA head-dim mismatch",
    run: testValidateGraphRejectsInvalidLlamaHeadDim
  },
  {
    name: "validateGraph rejects embedding stages in the middle of the model",
    run: testValidateGraphRejectsEmbeddingStageInMiddle
  },
  {
    name: "validateGraph rejects hidden stages before embeddings",
    run: testValidateGraphRejectsHiddenBeforeEmbedding
  },
  {
    name: "validateGraph rejects multiple output nodes for export",
    run: testValidateGraphRejectsMultipleOutputs
  },
  {
    name: "validateGraph rejects invalid Add arity",
    run: testValidateGraphRejectsInvalidAddArity
  },
  {
    name: "validateGraph rejects invalid Softmax axis",
    run: testValidateGraphRejectsInvalidSoftmaxAxis
  },
  {
    name: "validateGraph rejects classifier output export",
    run: testValidateGraphRejectsClassifierOutputExport
  },
  {
    name: "validateGraph rejects multiple direct inputs into transformer blocks",
    run: testValidateGraphRejectsTransformerMultiInput
  },
  {
    name: "validateGraph rejects multiple direct inputs into output nodes",
    run: testValidateGraphRejectsOutputMultiInput
  },
  {
    name: "validateGraph rejects branching direct outputs from transformer blocks",
    run: testValidateGraphRejectsTransformerBranching
  },
  {
    name: "GPT-2 project export uses the shared train entrypoint",
    run: testGpt2ProjectExportUsesSharedTrainEntrypoint
  },
  {
    name: "hybrid project export uses the shared train entrypoint",
    run: testHybridProjectExportUsesSharedTrainEntrypoint
  },
  {
    name: "direct model exports contain stable golden markers",
    run: testDirectModelExportsContainStableGoldenMarkers
  },
  {
    name: "config import mappers produce expected IR",
    run: testConfigImportMappersProduceExpectedIr
  },
  {
    name: "LLaMA project export uses the shared train entrypoint",
    run: testLlamaProjectExportUsesSharedTrainEntrypoint
  },
  {
    name: "generic project export uses the shared train entrypoint",
    run: testGenericProjectExportUsesSharedTrainEntrypoint
  },
  {
    name: "customized exact GPT-2 graphs are rejected by the exact mapper",
    run: testCustomizedExactGpt2GraphFallsThroughExactMapper
  },
  {
    name: "customized exact LLaMA graphs are rejected by the exact mapper",
    run: testCustomizedExactLlamaGraphFallsThroughExactMapper
  },
  {
    name: "hybrid mapper reports missing final norm clearly",
    run: testHybridMapperRejectsMissingFinalNormGracefully
  },
  {
    name: "hybrid mapper reports dual embeddings clearly",
    run: testHybridMapperRejectsDualEmbeddingsGracefully
  },
  {
    name: "hybrid mapper reports hidden-size mismatch clearly",
    run: testHybridMapperRejectsHiddenMismatchGracefully
  },
  {
    name: "hybrid mapper reports wrong output head clearly",
    run: testHybridMapperRejectsWrongOutputHeadGracefully
  },
  {
    name: "pruning helpers build stable remap, config, and script artifacts",
    run: testPruningHelpersBuildStableRemapAndConfig
  },
  {
    name: "pruning helpers handle inspection edge cases",
    run: testPruningHelpersHandleInspectionEdgeCases
  },
  {
    name: "prune service throws on NDJSON error event",
    run: testPruneServiceThrowsOnNdjsonErrorEvent
  },
  {
    name: "generated Python compiles for supported exports",
    run: testGeneratedPythonCompilesForSupportedExports
  }
];

let failures = 0;

for (const test of tests) {
  try {
    await Promise.resolve(test.run());
    console.log(`PASS ${test.name}`);
  } catch (error) {
    failures += 1;
    console.error(`FAIL ${test.name}`);
    console.error(error);
  }
}

if (failures > 0) {
  console.error(`test-suite failed: ${failures} test(s) failed`);
  process.exit(1);
}

console.log(`test-suite passed: ${tests.length} test(s)`);
