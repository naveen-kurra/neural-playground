import type { BlockNode } from "@neural-playground/block-schema";

export function isCustomizedGpt2Block(block: BlockNode): boolean {
  const feedforwardType = String(block.config.feedforwardType ?? "mlp");
  const activation = String(block.config.activation ?? "gelu_new");
  const layerNormEpsilon = Number(block.config.layerNormEpsilon ?? 1e-5);
  const scaleAttnWeights = Boolean(block.config.scaleAttnWeights ?? true);
  const scaleByLayer = Boolean(block.config.scaleAttnByInverseLayerIdx ?? false);
  const reorder = Boolean(block.config.reorderAndUpcastAttn ?? false);
  return (
    feedforwardType !== "mlp" ||
    activation !== "gelu_new" ||
    layerNormEpsilon !== 1e-5 ||
    scaleAttnWeights !== true ||
    scaleByLayer !== false ||
    reorder !== false
  );
}

export function isCustomizedLlamaBlock(block: BlockNode): boolean {
  const feedforwardType = String(block.config.feedforwardType ?? "mlp");
  const activation = String(block.config.activation ?? "silu");
  const rmsNormEpsilon = Number(block.config.rmsNormEpsilon ?? 1e-6);
  const attentionBias = Boolean(block.config.attentionBias ?? false);
  const dropout = Number(block.config.dropout ?? 0);
  const mlpBias = Boolean(block.config.mlpBias ?? false);
  return (
    feedforwardType !== "mlp" ||
    activation !== "silu" ||
    rmsNormEpsilon !== 1e-6 ||
    attentionBias !== false ||
    dropout !== 0 ||
    mlpBias !== false
  );
}

export function isCustomizedMistralBlock(block: BlockNode): boolean {
  const feedforwardType = String(block.config.feedforwardType ?? "mlp");
  const activation = String(block.config.activation ?? "silu");
  const rmsNormEpsilon = Number(block.config.rmsNormEpsilon ?? 1e-5);
  const attentionBias = Boolean(block.config.attentionBias ?? false);
  const dropout = Number(block.config.dropout ?? 0);
  const mlpBias = Boolean(block.config.mlpBias ?? false);
  return (
    feedforwardType !== "mlp" ||
    activation !== "silu" ||
    rmsNormEpsilon !== 1e-5 ||
    attentionBias !== false ||
    dropout !== 0 ||
    mlpBias !== false
  );
}

export function isCustomizedPhi3Block(block: BlockNode): boolean {
  const feedforwardType = String(block.config.feedforwardType ?? "mlp");
  const activation = String(block.config.activation ?? "silu");
  const rmsNormEpsilon = Number(block.config.rmsNormEpsilon ?? 1e-5);
  const attentionBias = Boolean(block.config.attentionBias ?? false);
  const dropout = Number(block.config.dropout ?? 0);
  const mlpBias = Boolean(block.config.mlpBias ?? false);
  return (
    feedforwardType !== "mlp" ||
    activation !== "silu" ||
    rmsNormEpsilon !== 1e-5 ||
    attentionBias !== false ||
    dropout !== 0 ||
    mlpBias !== false
  );
}
