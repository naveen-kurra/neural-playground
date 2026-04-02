import type { HybridDecoderArchitectureSpec } from "@neural-playground/ir-schema";
import { renderPrimitiveSections } from "./primitives";
import { renderHybridModel } from "./recipes";

export function exportHybridIrToPyTorch(spec: HybridDecoderArchitectureSpec): string {
  return renderHybridModel(spec, renderPrimitiveSections(spec));
}

