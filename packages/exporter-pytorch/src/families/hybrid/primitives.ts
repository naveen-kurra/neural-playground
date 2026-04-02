import type { HybridDecoderArchitectureSpec } from "@neural-playground/ir-schema";
import {
  renderActivationRegistry,
  renderConv1D,
  renderGpt2Attention,
  renderGpt2Block,
  renderGpt2MLP,
  renderGpt2MoE,
  renderLlamaAttention,
  renderLlamaDecoderLayer,
  renderLlamaMLP,
  renderLlamaMoE,
  renderLlamaNormAndRotaryHelpers,
  renderNewGeluActivation
} from "../../shared/python-snippets";

function needsGpt2(spec: HybridDecoderArchitectureSpec): boolean {
  return spec.operators.embedding.family === "gpt2" || spec.operators.blocks.some((block) => block.family === "gpt2");
}

function needsGpt2Moe(spec: HybridDecoderArchitectureSpec): boolean {
  return spec.operators.blocks.some((block) => block.family === "gpt2" && block.feedforwardType === "moe");
}

function needsLlama(spec: HybridDecoderArchitectureSpec): boolean {
  return (
    spec.operators.embedding.family === "llama" ||
    spec.operators.finalNorm.family === "llama" ||
    spec.operators.blocks.some((block) => block.family === "llama")
  );
}

function needsLlamaMoe(spec: HybridDecoderArchitectureSpec): boolean {
  return spec.operators.blocks.some((block) => block.family === "llama" && block.feedforwardType === "moe");
}

function renderActivationHelpers(spec: HybridDecoderArchitectureSpec): string {
  const sections: string[] = [];

  if (needsGpt2(spec)) {
    sections.push(renderNewGeluActivation(), renderActivationRegistry("gpt2", "get_gpt2_activation"));
  }

  if (needsLlama(spec)) {
    sections.push(renderActivationRegistry("llama", "get_llama_activation"));
  }

  return sections.join("\n");
}

function renderGpt2Primitives(spec: HybridDecoderArchitectureSpec): string {
  if (!needsGpt2(spec)) {
    return "";
  }

  return [
    renderConv1D(),
    renderGpt2Attention("HybridGPT2BlockConfig"),
    renderGpt2MLP("HybridGPT2BlockConfig", "get_gpt2_activation"),
    needsGpt2Moe(spec) ? renderGpt2MoE("HybridGPT2BlockConfig") : "",
    renderGpt2Block("HybridGPT2BlockConfig", { hybrid: true })
  ]
    .filter(Boolean)
    .join("\n");
}

function renderLlamaPrimitives(spec: HybridDecoderArchitectureSpec): string {
  if (!needsLlama(spec)) {
    return "";
  }

  return [
    renderLlamaNormAndRotaryHelpers(),
    renderLlamaAttention("HybridLlamaBlockConfig"),
    renderLlamaMLP("HybridLlamaBlockConfig", "get_llama_activation"),
    needsLlamaMoe(spec) ? renderLlamaMoE("HybridLlamaBlockConfig") : "",
    renderLlamaDecoderLayer("HybridLlamaBlockConfig", { hybrid: true })
  ]
    .filter(Boolean)
    .join("\n");
}

export function renderPrimitiveSections(spec: HybridDecoderArchitectureSpec): string {
  return [renderActivationHelpers(spec), renderGpt2Primitives(spec), renderLlamaPrimitives(spec)].filter(Boolean).join("\n");
}
