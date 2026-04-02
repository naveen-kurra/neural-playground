import type { GPT2ArchitectureSpec } from "@neural-playground/ir-schema";
import { renderGpt2Primitives } from "./primitives";
import { renderGpt2Model } from "./recipes";

export function exportGPT2IrToPyTorch(spec: GPT2ArchitectureSpec): string {
  return renderGpt2Model(spec, renderGpt2Primitives());
}
