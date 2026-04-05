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
import { renderPhi3Primitives } from "../phi3/primitives";

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

function needsPhi3(spec: HybridDecoderArchitectureSpec): boolean {
  return (
    spec.operators.embedding.family === "phi3" ||
    spec.operators.finalNorm.family === "phi3" ||
    spec.operators.blocks.some((block) => block.family === "phi3")
  );
}

function needsPhi3Moe(spec: HybridDecoderArchitectureSpec): boolean {
  return spec.operators.blocks.some((block) => block.family === "phi3" && block.feedforwardType === "moe");
}

function renderActivationHelpers(spec: HybridDecoderArchitectureSpec): string {
  const sections: string[] = [];

  if (needsGpt2(spec)) {
    sections.push(renderNewGeluActivation(), renderActivationRegistry("gpt2", "get_gpt2_activation"));
  }

  if (needsLlama(spec)) {
    sections.push(renderActivationRegistry("llama", "get_llama_activation"));
  }

  if (needsPhi3(spec)) {
    sections.push(renderActivationRegistry("phi3", "get_phi3_activation"));
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

function renderPhi3HybridPrimitives(spec: HybridDecoderArchitectureSpec): string {
  if (!needsPhi3(spec)) {
    return "";
  }

  const renamed = renderPhi3Primitives()
    .replaceAll("get_activation", "get_phi3_activation")
    .replaceAll("Phi3Config", "HybridLlamaBlockConfig");

  if (needsPhi3Moe(spec)) {
    return renamed;
  }

  return renamed.replace(/class Phi3MoE[\s\S]*?(?=\nclass Phi3DecoderLayer)/, "");
}

export function renderPrimitiveSections(spec: HybridDecoderArchitectureSpec): string {
  return [renderActivationHelpers(spec), renderGpt2Primitives(spec), renderLlamaPrimitives(spec), renderPhi3HybridPrimitives(spec)]
    .filter(Boolean)
    .join("\n");
}
