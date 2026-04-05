export type ModelTemplate = {
  id: string;
  label: string;
  family: "gpt2" | "llama" | "phi3" | "gemma4";
  modelId: string;
  description: string;
  defaultBlockCount: number;
  overrides?: {
    vocabSize?: number;
    maxPositionEmbeddings?: number;
    hiddenSize?: number;
    intermediateSize?: number;
    numAttentionHeads?: number;
    numKeyValueHeads?: number;
    headDim?: number;
    rmsNormEpsilon?: number;
    ropeTheta?: number;
    tieWordEmbeddings?: boolean;
    slidingWindow?: number;
    layerTypes?: string[];
    ropeParameters?: Record<string, { ropeType: string; ropeTheta: number }>;
    numGlobalKeyValueHeads?: number | null;
    globalHeadDim?: number;
    attentionKEqV?: boolean;
    numKvSharedLayers?: number;
  };
};

export const modelTemplates: ModelTemplate[] = [
  {
    id: "gpt2",
    label: "GPT-2",
    family: "gpt2",
    modelId: "gpt2",
    description: "Baseline GPT-2 decoder stack.",
    defaultBlockCount: 12
  },
  {
    id: "llama",
    label: "LLaMA",
    family: "llama",
    modelId: "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    description: "Public LLaMA-family decoder stack.",
    defaultBlockCount: 22
  },
  {
    id: "gemma4-31b",
    label: "Gemma 4 31B",
    family: "gemma4",
    modelId: "google/gemma-4-31B-it",
    description: "Gemma 4 text-only baseline preset on a separate Gemma 4 family path.",
    defaultBlockCount: 60,
    overrides: {
      vocabSize: 262144,
      maxPositionEmbeddings: 262144,
      hiddenSize: 5376,
      intermediateSize: 21504,
      numAttentionHeads: 32,
      numKeyValueHeads: 16,
      headDim: 256,
      rmsNormEpsilon: 1e-6,
      ropeTheta: 1000000,
      tieWordEmbeddings: true,
      slidingWindow: 1024,
      layerTypes: Array.from({ length: 60 }, (_, index) => ((index + 1) % 6 === 0 ? "full_attention" : "sliding_attention")),
      ropeParameters: {
        sliding_attention: { ropeType: "default", ropeTheta: 10000 },
        full_attention: { ropeType: "proportional", ropeTheta: 1000000 }
      },
      numGlobalKeyValueHeads: 4,
      globalHeadDim: 512,
      attentionKEqV: false,
      numKvSharedLayers: 0
    }
  },
  {
    id: "phi3",
    label: "Phi-3 Mini",
    family: "phi3",
    modelId: "microsoft/Phi-3-mini-4k-instruct",
    description: "Phi-3 Mini decoder starter with dedicated Phi-3 family defaults.",
    defaultBlockCount: 32,
    overrides: {
      vocabSize: 32064,
      maxPositionEmbeddings: 4096,
      hiddenSize: 3072,
      intermediateSize: 8192,
      numAttentionHeads: 32,
      numKeyValueHeads: 32,
      headDim: 96,
      rmsNormEpsilon: 1e-5,
      ropeTheta: 10000,
      tieWordEmbeddings: false
    }
  },
  {
    id: "phi3-medium",
    label: "Phi-3 Medium",
    family: "phi3",
    modelId: "microsoft/Phi-3-medium-4k-instruct",
    description: "Phi-3 Medium exact-family preset using the dedicated Phi-3 exporter/runtime path.",
    defaultBlockCount: 40,
    overrides: {
      vocabSize: 32064,
      maxPositionEmbeddings: 4096,
      hiddenSize: 5120,
      intermediateSize: 17920,
      numAttentionHeads: 40,
      numKeyValueHeads: 10,
      headDim: 128,
      rmsNormEpsilon: 1e-5,
      ropeTheta: 10000,
      tieWordEmbeddings: false
    }
  },
  {
    id: "phi3.5-mini",
    label: "Phi-3.5 Mini",
    family: "phi3",
    modelId: "microsoft/Phi-3.5-mini-instruct",
    description: "Phi-3.5 Mini long-context preset on the Phi-3 family path.",
    defaultBlockCount: 32,
    overrides: {
      vocabSize: 32064,
      maxPositionEmbeddings: 131072,
      hiddenSize: 3072,
      intermediateSize: 8192,
      numAttentionHeads: 32,
      numKeyValueHeads: 32,
      headDim: 96,
      rmsNormEpsilon: 1e-5,
      ropeTheta: 10000,
      tieWordEmbeddings: false
    }
  }
];

export function resolveTemplate(selection: string): ModelTemplate | null {
  const normalized = selection.trim().toLowerCase();
  if (!normalized) {
    return null;
  }

  const match = modelTemplates.find(
    (template) =>
      template.id.toLowerCase() === normalized ||
      template.label.toLowerCase() === normalized ||
      template.modelId.toLowerCase() === normalized
  );

  return match ?? null;
}
