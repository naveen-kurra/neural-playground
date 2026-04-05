import type { Gemma4ArchitectureSpec } from "@neural-playground/ir-schema";
import { renderGemma4Primitives } from "./primitives";
import { renderGemma4Model } from "./recipes";

export function exportGemma4IrToPyTorch(spec: Gemma4ArchitectureSpec): string {
  return renderGemma4Model(spec, renderGemma4Primitives());
}
