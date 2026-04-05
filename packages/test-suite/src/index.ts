import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import type { ModelGraph } from "@neural-playground/block-schema";
import {
  exportGemma4IrProjectFiles,
  exportGemma4IrToPyTorch,
  exportGPT2IrProjectFiles,
  exportGPT2IrToPyTorch,
  exportHybridIrProjectFiles,
  exportHybridIrToPyTorch,
  exportLlamaIrProjectFiles,
  exportLlamaIrToPyTorch,
  exportPhi3IrProjectFiles,
  exportPhi3IrToPyTorch,
  exportProjectFiles
} from "@neural-playground/exporter-pytorch";
import {
  buildGemma4ArchitectureSpec,
  mapGPT2ConfigToIr,
  mapGemma4ConfigToIr,
  mapLlamaConfigToIr,
  mapPhi3ConfigToIr,
  buildPhi3ArchitectureSpec,
  mapModelGraphToGemma4Ir,
  mapModelGraphToPhi3Ir,
  projectGemma4IrToModelGraph,
  projectPhi3IrToModelGraph,
  mapModelGraphToGPT2Ir,
  mapModelGraphToHybridIr,
  mapModelGraphToLlamaIr
} from "@neural-playground/ir-schema";
import { validateGraph } from "@neural-playground/validator";

import { resolveTemplate } from "../../../apps/web/src/app/model-templates";
import { fetchHuggingFaceModel, type HuggingFaceFetchResult } from "../../../apps/web/src/app/huggingface";
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

function buildHybridPhi3Graph(): ModelGraph {
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
        id: "phi3-block-1",
        type: "Phi3Block",
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
          rmsNormEpsilon: 1e-5,
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
      { id: "final-norm", type: "Phi3FinalRMSNorm", position: { x: 700, y: 50 }, config: { epsilon: 1e-5 } },
      { id: "lm-head", type: "Phi3LMHead", position: { x: 800, y: 50 }, config: { vocabSize: 32000, tiedWeights: false } },
      { id: "output", type: "Output", position: { x: 900, y: 50 }, config: { headType: "LanguageModel" } }
    ],
    edges: [
      { id: "e1", source: "input", target: "tok" },
      { id: "e2", source: "input", target: "pos" },
      { id: "e3", source: "tok", target: "add" },
      { id: "e4", source: "pos", target: "add" },
      { id: "e5", source: "add", target: "drop" },
      { id: "e6", source: "drop", target: "gpt-block-0" },
      { id: "e7", source: "gpt-block-0", target: "phi3-block-1" },
      { id: "e8", source: "phi3-block-1", target: "gpt-block-2" },
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

function buildCustomizedExactPhi3Graph(): ModelGraph {
  const graph = projectPhi3IrToModelGraph(buildPhi3ArchitectureSpec({ numHiddenLayers: 1 }));
  const block = graph.nodes.find((node) => node.type === "Phi3Block");
  assert.ok(block);
  block.config.feedforwardType = "moe";
  block.config.numExperts = 4;
  block.config.topK = 2;
  block.config.expertHidden = 8192;
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
    resolvedFamily: "phi3",
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

function testHybridPhi3ProjectExportUsesSharedTrainEntrypoint(): void {
  const graph = buildHybridPhi3Graph();
  const spec = mapModelGraphToHybridIr(graph);
  const files = exportHybridIrProjectFiles(spec, graph.training);

  assert.deepEqual(spec.config.blockFamilies, ["gpt2", "phi3", "gpt2"]);
  assert.equal(spec.config.finalNormFamily, "phi3");
  assert.ok(files["scripts/train.py"]?.includes("[train] loading configs..."));
  assert.ok(files["configs/model.yaml"]?.includes("embedding_family: gpt2"));
  assert.ok(files["configs/model.yaml"]?.includes("final_norm_family: phi3"));
  assert.ok(files["configs/model.yaml"]?.includes("family: phi3"));
  assert.ok(files["src/kurra_ai_cb/model.py"]?.includes("Phi3DecoderLayer("));
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

  const hybridPhi3Graph = buildHybridPhi3Graph();
  const hybridPhi3Spec = mapModelGraphToHybridIr(hybridPhi3Graph);
  const hybridPhi3Model = exportHybridIrToPyTorch(hybridPhi3Spec);
  assert.ok(hybridPhi3Model.includes("class Phi3DecoderLayer(nn.Module):"));
  assert.ok(hybridPhi3Model.includes("class Phi3RMSNorm(nn.Module):"));
  assert.ok(hybridPhi3Model.includes("Phi3DecoderLayer("));
  assert.ok(hybridPhi3Model.includes("GPT2Block("));
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

  const phi3 = mapPhi3ConfigToIr(
    {
      model_type: "phi3",
      vocab_size: 32064,
      hidden_size: 3072,
      intermediate_size: 8192,
      num_hidden_layers: 32,
      num_attention_heads: 32,
      num_key_value_heads: 32,
      hidden_act: "silu",
      max_position_embeddings: 4096,
      rms_norm_eps: 1e-5,
      rope_theta: 10000,
      attention_bias: false,
      attention_dropout: 0,
      mlp_bias: false,
      tie_word_embeddings: false
    },
    { modelId: "microsoft/Phi-3-mini-4k-instruct", name: "Phi-3 import" }
  );
  assert.equal(phi3.family, "phi3");
  assert.equal(phi3.config.vocabSize, 32064);
  assert.equal(phi3.config.hiddenSize, 3072);
  assert.equal(phi3.config.numHiddenLayers, 32);
  assert.equal(phi3.config.headDim, 96);
  assert.equal(phi3.config.rmsNormEpsilon, 1e-5);
}

async function testFetchHuggingFaceModelInfersPhi3FamilyAndNestedLayerCount(): Promise<void> {
  const originalFetch = globalThis.fetch;
  const responses = new Map<string, Record<string, unknown>>([
    [
      "https://huggingface.co/microsoft/Phi-3-vision/resolve/main/config.json",
      {
        model_type: "phi3",
        architectures: ["Phi3ForCausalLM"],
        text_config: {
          num_hidden_layers: 40
        }
      }
    ],
    [
      "https://huggingface.co/microsoft/Phi-3-vision/resolve/main/model.safetensors.index.json",
      {
        weight_map: {
          "model.layers.0.input_layernorm.weight": "model-00001-of-00002.safetensors",
          "model.layers.1.input_layernorm.weight": "model-00001-of-00002.safetensors"
        }
      }
    ]
  ]);

  (globalThis as Record<string, unknown>).fetch = async (input: string | URL) => {
    const url = String(input);
    const body = responses.get(url);
    if (!body) {
      return new Response("not found", { status: 404 });
    }
    return new Response(JSON.stringify(body), {
      status: 200,
      headers: { "Content-Type": "application/json" }
    });
  };

  try {
    const result = await fetchHuggingFaceModel("microsoft/Phi-3-vision");
    assert.equal(result.resolvedFamily, "phi3");
    assert.equal(result.inspection.layerCountHint, 40);
    assert.equal(result.inspection.layerCountKey, "text_config.num_hidden_layers");
    assert.equal(result.inspection.detectedLayerPrefix, "model.layers.");
  } finally {
    (globalThis as Record<string, unknown>).fetch = originalFetch;
  }
}

function testPhi3TemplateUsesDedicatedFamilyDefaults(): void {
  const template = resolveTemplate("Phi-3 Mini");
  assert.ok(template, "expected Phi-3 template to resolve");
  assert.equal(template.family, "phi3");

  const spec = buildPhi3ArchitectureSpec({
    modelId: template.modelId,
    numHiddenLayers: template.defaultBlockCount,
    vocabSize: template.overrides?.vocabSize,
    hiddenSize: template.overrides?.hiddenSize,
    intermediateSize: template.overrides?.intermediateSize,
    numAttentionHeads: template.overrides?.numAttentionHeads,
    numKeyValueHeads: template.overrides?.numKeyValueHeads,
    headDim: template.overrides?.headDim,
    maxPositionEmbeddings: template.overrides?.maxPositionEmbeddings,
    rmsNormEpsilon: template.overrides?.rmsNormEpsilon,
    ropeTheta: template.overrides?.ropeTheta,
    tieWordEmbeddings: template.overrides?.tieWordEmbeddings
  });
  assert.equal(spec.family, "phi3");
  assert.equal(spec.config.hiddenSize, 3072);
  assert.equal(spec.config.intermediateSize, 8192);
  assert.equal(spec.config.numHiddenLayers, 32);
  assert.equal(spec.config.vocabSize, 32064);

  const graph = projectPhi3IrToModelGraph(spec);
  assert.equal(graph.nodes.filter((node) => node.type === "Phi3Block").length, 32);
  assert.equal(graph.nodes.find((node) => node.type === "Phi3TokenEmbedding")?.config.embeddingDim, 3072);
}

function testPhi3MediumTemplateUsesDedicatedFamilyDefaults(): void {
  const template = resolveTemplate("Phi-3 Medium");
  assert.ok(template, "expected Phi-3 Medium template to resolve");
  assert.equal(template.family, "phi3");

  const spec = buildPhi3ArchitectureSpec({
    modelId: template.modelId,
    numHiddenLayers: template.defaultBlockCount,
    vocabSize: template.overrides?.vocabSize,
    hiddenSize: template.overrides?.hiddenSize,
    intermediateSize: template.overrides?.intermediateSize,
    numAttentionHeads: template.overrides?.numAttentionHeads,
    numKeyValueHeads: template.overrides?.numKeyValueHeads,
    headDim: template.overrides?.headDim,
    maxPositionEmbeddings: template.overrides?.maxPositionEmbeddings,
    rmsNormEpsilon: template.overrides?.rmsNormEpsilon,
    ropeTheta: template.overrides?.ropeTheta,
    tieWordEmbeddings: template.overrides?.tieWordEmbeddings
  });
  assert.equal(spec.family, "phi3");
  assert.equal(spec.config.hiddenSize, 5120);
  assert.equal(spec.config.intermediateSize, 17920);
  assert.equal(spec.config.numHiddenLayers, 40);
  assert.equal(spec.config.numAttentionHeads, 40);
  assert.equal(spec.config.numKeyValueHeads, 10);
  assert.equal(spec.config.headDim, 128);
  assert.equal(spec.config.vocabSize, 32064);

  const graph = projectPhi3IrToModelGraph(spec);
  assert.equal(graph.nodes.filter((node) => node.type === "Phi3Block").length, 40);
  assert.equal(graph.nodes.find((node) => node.type === "Phi3TokenEmbedding")?.config.embeddingDim, 5120);
}

function testPhi35MiniTemplateUsesDedicatedFamilyDefaults(): void {
  const template = resolveTemplate("Phi-3.5 Mini");
  assert.ok(template, "expected Phi-3.5 Mini template to resolve");
  assert.equal(template.family, "phi3");

  const spec = buildPhi3ArchitectureSpec({
    modelId: template.modelId,
    numHiddenLayers: template.defaultBlockCount,
    vocabSize: template.overrides?.vocabSize,
    hiddenSize: template.overrides?.hiddenSize,
    intermediateSize: template.overrides?.intermediateSize,
    numAttentionHeads: template.overrides?.numAttentionHeads,
    numKeyValueHeads: template.overrides?.numKeyValueHeads,
    headDim: template.overrides?.headDim,
    maxPositionEmbeddings: template.overrides?.maxPositionEmbeddings,
    rmsNormEpsilon: template.overrides?.rmsNormEpsilon,
    ropeTheta: template.overrides?.ropeTheta,
    tieWordEmbeddings: template.overrides?.tieWordEmbeddings
  });
  assert.equal(spec.family, "phi3");
  assert.equal(spec.config.hiddenSize, 3072);
  assert.equal(spec.config.intermediateSize, 8192);
  assert.equal(spec.config.numHiddenLayers, 32);
  assert.equal(spec.config.maxPositionEmbeddings, 131072);
  assert.equal(spec.config.headDim, 96);
  assert.equal(spec.config.vocabSize, 32064);

  const graph = projectPhi3IrToModelGraph(spec);
  assert.equal(graph.nodes.filter((node) => node.type === "Phi3Block").length, 32);
  assert.equal(graph.nodes.find((node) => node.type === "Phi3TokenEmbedding")?.config.embeddingDim, 3072);
}

function testGemma4TemplateUsesDedicatedFamilyDefaults(): void {
  const template = resolveTemplate("Gemma 4 31B");
  assert.ok(template, "expected Gemma 4 template to resolve");
  assert.equal(template.family, "gemma4");

  const spec = buildGemma4ArchitectureSpec({
    modelId: template.modelId,
    numHiddenLayers: template.defaultBlockCount,
    vocabSize: template.overrides?.vocabSize,
    hiddenSize: template.overrides?.hiddenSize,
    intermediateSize: template.overrides?.intermediateSize,
    numAttentionHeads: template.overrides?.numAttentionHeads,
    numKeyValueHeads: template.overrides?.numKeyValueHeads,
    headDim: template.overrides?.headDim,
    maxPositionEmbeddings: template.overrides?.maxPositionEmbeddings,
    rmsNormEpsilon: template.overrides?.rmsNormEpsilon,
    ropeTheta: template.overrides?.ropeTheta,
    tieWordEmbeddings: template.overrides?.tieWordEmbeddings
  });
  assert.equal(spec.family, "gemma4");
  assert.equal(spec.config.hiddenSize, 5376);
  assert.equal(spec.config.intermediateSize, 21504);
  assert.equal(spec.config.numHiddenLayers, 60);
  assert.equal(spec.config.numAttentionHeads, 32);
  assert.equal(spec.config.numKeyValueHeads, 16);
  assert.equal(spec.config.headDim, 256);
  assert.equal(spec.config.vocabSize, 262144);
  assert.equal(spec.config.slidingWindow, 1024);
  assert.equal(spec.config.numGlobalKeyValueHeads, 4);
  assert.equal(spec.config.globalHeadDim, 512);
  assert.equal(spec.config.attentionKEqV, false);
  assert.equal(spec.config.numKvSharedLayers, 0);
  assert.equal(spec.config.layerTypes.length, 60);
  assert.equal(spec.config.layerTypes[0], "sliding_attention");
  assert.equal(spec.config.layerTypes[5], "full_attention");
  assert.equal(spec.config.layerTypes[11], "full_attention");
  assert.equal(spec.config.layerTypes.at(-1), "full_attention");
  assert.equal(spec.config.ropeParameters.sliding_attention?.ropeType, "default");
  assert.equal(spec.config.ropeParameters.sliding_attention?.ropeTheta, 10000);
  assert.equal(spec.config.ropeParameters.full_attention?.ropeType, "proportional");
  assert.equal(spec.config.ropeParameters.full_attention?.ropeTheta, 1000000);

  const graph = projectGemma4IrToModelGraph(spec);
  assert.equal(graph.nodes.filter((node) => node.type === "Gemma4Block").length, 60);
  assert.equal(graph.nodes.find((node) => node.type === "Gemma4TokenEmbedding")?.config.embeddingDim, 5376);
  const gemmaBlocks = graph.nodes.filter((node) => node.type === "Gemma4Block");
  assert.equal(gemmaBlocks[0]?.config.layerType, "sliding_attention");
  assert.equal(gemmaBlocks[5]?.config.layerType, "full_attention");
  assert.equal(gemmaBlocks.at(-1)?.config.layerType, "full_attention");
  assert.equal(gemmaBlocks[0]?.config.slidingWindow, 1024);
  assert.equal(gemmaBlocks[0]?.config.numGlobalKeyValueHeads, 4);
  assert.equal(gemmaBlocks[0]?.config.globalHeadDim, 512);
  assert.equal(gemmaBlocks[0]?.config.attentionKEqV, false);
  assert.equal(gemmaBlocks[0]?.config.numKvSharedLayers, 0);
}

function testGemma4ConfigImportProducesExpectedIr(): void {
  const gemma = mapGemma4ConfigToIr(
    {
      model_type: "gemma4",
      text_config: {
        vocab_size: 262144,
        hidden_size: 5376,
        intermediate_size: 21504,
        num_hidden_layers: 60,
        num_attention_heads: 32,
        num_key_value_heads: 16,
        head_dim: 256,
        hidden_activation: "gelu_pytorch_tanh",
        max_position_embeddings: 262144,
        rms_norm_eps: 1e-6,
        rope_theta: 1000000,
        attention_bias: false,
        attention_dropout: 0,
        mlp_bias: false,
        tie_word_embeddings: true,
        sliding_window: 1024,
        layer_types: ["sliding_attention", "full_attention"],
        rope_parameters: {
          sliding_attention: { rope_type: "default", rope_theta: 1000000 },
          full_attention: { rope_type: "default", rope_theta: 1000000 }
        },
        num_global_key_value_heads: 4,
        global_head_dim: 512,
        attention_k_eq_v: false,
        num_kv_shared_layers: 0
      }
    },
    { modelId: "google/gemma-4-31B-it", name: "Gemma 4 import" }
  );
  assert.equal(gemma.family, "gemma4");
  assert.equal(gemma.config.hiddenSize, 5376);
  assert.equal(gemma.config.intermediateSize, 21504);
  assert.equal(gemma.config.numHiddenLayers, 60);
  assert.equal(gemma.config.numAttentionHeads, 32);
  assert.equal(gemma.config.numKeyValueHeads, 16);
  assert.equal(gemma.config.headDim, 256);
  assert.equal(gemma.config.slidingWindow, 1024);
  assert.equal(gemma.config.layerTypes.length, 2);
  assert.equal(gemma.config.layerTypes[1], "full_attention");
  assert.equal(gemma.config.numGlobalKeyValueHeads, 4);
  assert.equal(gemma.config.globalHeadDim, 512);
}

function testGemma4ProjectExportUsesDedicatedFamilyPath(): void {
  const spec = buildGemma4ArchitectureSpec({ numHiddenLayers: 2 });
  const modelPy = exportGemma4IrToPyTorch(spec);
  const files = exportGemma4IrProjectFiles(spec, { ...defaultTraining });
  assert.ok(modelPy.includes("class Gemma4Config:"));
  assert.ok(modelPy.includes("class Gemma4TextModel(nn.Module):"));
  assert.ok(modelPy.includes("class Gemma4ForCausalLM(nn.Module):"));
  assert.ok(modelPy.includes("self.q_proj = nn.Linear"));
  assert.ok(modelPy.includes("self.k_proj = nn.Linear"));
  assert.ok(modelPy.includes("self.v_proj = nn.Linear"));
  assert.ok(modelPy.includes("self.gate_proj = nn.Linear"));
  assert.ok(modelPy.includes("self.up_proj = nn.Linear"));
  assert.ok(modelPy.includes("self.layers = nn.ModuleList("));
  assert.ok(modelPy.includes("[Gemma4TextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]"));
  assert.ok(files["configs/model.yaml"]?.includes("model_family: gemma4"));
  assert.ok(files["src/kurra_ai_cb/model.py"]?.includes("def build_model(cfg, seq_len_override: int | None = None) -> Gemma4ForCausalLM:"));
}

function testExactGemma4GraphMapsToDedicatedFamily(): void {
  const spec = buildGemma4ArchitectureSpec({ numHiddenLayers: 2, headDim: 256 });
  const graph = projectGemma4IrToModelGraph(spec);
  const remapped = mapModelGraphToGemma4Ir(graph);
  assert.equal(remapped.family, "gemma4");
  assert.equal(remapped.config.numHiddenLayers, 2);
  assert.equal(remapped.config.headDim, 256);
  assert.deepEqual(remapped.config.layerTypes, ["sliding_attention", "full_attention"]);
  assert.equal(remapped.config.slidingWindow, 1024);
  assert.equal(remapped.config.numGlobalKeyValueHeads, 4);
  assert.equal(remapped.config.globalHeadDim, 512);
}

function testPhi3ProjectExportUsesDedicatedFamilyPath(): void {
  const spec = buildPhi3ArchitectureSpec();
  const modelPy = exportPhi3IrToPyTorch(spec);
  const files = exportPhi3IrProjectFiles(spec, { ...defaultTraining });
  assert.ok(modelPy.includes("class Phi3Config:"));
  assert.ok(modelPy.includes("class Phi3Model(nn.Module):"));
  assert.ok(modelPy.includes("class Phi3ForCausalLM(nn.Module):"));
  assert.ok(modelPy.includes("self.qkv_proj = nn.Linear"));
  assert.ok(modelPy.includes("self.gate_up_proj = nn.Linear"));
  assert.ok(modelPy.includes("self.resid_attn_dropout = nn.Dropout"));
  assert.ok(modelPy.includes("position_ids=position_ids"));
  assert.ok(modelPy.includes("position_embeddings=position_embeddings"));
  assert.ok(files["configs/model.yaml"]?.includes("model_family: phi3"));
  assert.ok(files["src/kurra_ai_cb/model.py"]?.includes("def build_model(cfg, seq_len_override: int | None = None) -> Phi3ForCausalLM:"));
}

function testExactPhi3GraphMapsToDedicatedFamily(): void {
  const spec = buildPhi3ArchitectureSpec({ numHiddenLayers: 2 });
  const graph = projectPhi3IrToModelGraph(spec);
  const remapped = mapModelGraphToPhi3Ir(graph);
  assert.equal(remapped.family, "phi3");
  assert.equal(remapped.config.numHiddenLayers, 2);
}

function testCustomizedExactPhi3GraphFallsThroughExactMapper(): void {
  const graph = buildCustomizedExactPhi3Graph();
  assert.throws(
    () => mapModelGraphToPhi3Ir(graph),
    /Customized Phi-3 block internals are not exported through the exact Phi-3 path yet\./
  );
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

  const hybridPhi3Graph = buildHybridPhi3Graph();
  const hybridPhi3Spec = mapModelGraphToHybridIr(hybridPhi3Graph);
  const hybridPhi3Files = exportHybridIrProjectFiles(hybridPhi3Spec, hybridPhi3Graph.training);
  compilePythonSource("hybrid_phi3_model.py", exportHybridIrToPyTorch(hybridPhi3Spec));
  compilePythonSource("hybrid_phi3_train_script.py", hybridPhi3Files["scripts/train.py"]);

  const phi3Spec = buildPhi3ArchitectureSpec({ numHiddenLayers: 2 });
  const phi3Files = exportPhi3IrProjectFiles(phi3Spec, { ...defaultTraining });
  compilePythonSource("phi3_model.py", exportPhi3IrToPyTorch(phi3Spec));
  compilePythonSource("phi3_train_script.py", phi3Files["scripts/train.py"]);

  const gemma4Spec = buildGemma4ArchitectureSpec({ numHiddenLayers: 2, headDim: 256 });
  const gemma4Files = exportGemma4IrProjectFiles(gemma4Spec, { ...defaultTraining });
  compilePythonSource("gemma4_model.py", exportGemma4IrToPyTorch(gemma4Spec));
  compilePythonSource("gemma4_train_script.py", gemma4Files["scripts/train.py"]);
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
    name: "hybrid Phi-3 project export uses the shared train entrypoint",
    run: testHybridPhi3ProjectExportUsesSharedTrainEntrypoint
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
    name: "Hugging Face model inspection infers Phi-3 family and nested layer count",
    run: testFetchHuggingFaceModelInfersPhi3FamilyAndNestedLayerCount
  },
  {
    name: "Phi-3 template uses the dedicated Phi-3 family defaults",
    run: testPhi3TemplateUsesDedicatedFamilyDefaults
  },
  {
    name: "Phi-3 Medium template uses the dedicated Phi-3 family defaults",
    run: testPhi3MediumTemplateUsesDedicatedFamilyDefaults
  },
  {
    name: "Phi-3.5 Mini template uses the dedicated Phi-3 family defaults",
    run: testPhi35MiniTemplateUsesDedicatedFamilyDefaults
  },
  {
    name: "Gemma 4 template uses the dedicated Gemma 4 family defaults",
    run: testGemma4TemplateUsesDedicatedFamilyDefaults
  },
  {
    name: "Phi-3 project export uses the dedicated family path",
    run: testPhi3ProjectExportUsesDedicatedFamilyPath
  },
  {
    name: "Gemma 4 config import produces the expected IR",
    run: testGemma4ConfigImportProducesExpectedIr
  },
  {
    name: "Gemma 4 project export uses the dedicated family path",
    run: testGemma4ProjectExportUsesDedicatedFamilyPath
  },
  {
    name: "exact Phi-3 graphs map to the dedicated family",
    run: testExactPhi3GraphMapsToDedicatedFamily
  },
  {
    name: "exact Gemma 4 graphs map to the dedicated family",
    run: testExactGemma4GraphMapsToDedicatedFamily
  },
  {
    name: "customized exact Phi-3 graphs are rejected by the exact mapper",
    run: testCustomizedExactPhi3GraphFallsThroughExactMapper
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
