import type { Phi3ArchitectureSpec } from "@neural-playground/ir-schema";
import { renderPhi3Primitives } from "./primitives";
import { renderPhi3Model } from "./recipes";

export function exportPhi3IrToPyTorch(spec: Phi3ArchitectureSpec): string {
  return renderPhi3Model(spec, renderPhi3Primitives());
}
