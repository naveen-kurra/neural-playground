import type { MistralArchitectureSpec } from "@neural-playground/ir-schema";
import { renderMistralPrimitives } from "./primitives";
import { renderMistralModel } from "./recipes";

export function exportMistralIrToPyTorch(spec: MistralArchitectureSpec): string {
  return renderMistralModel(spec, renderMistralPrimitives());
}
