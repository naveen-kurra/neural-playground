import type { LlamaArchitectureSpec } from "@neural-playground/ir-schema";
import { renderLlamaPrimitives } from "./primitives";
import { renderLlamaModel } from "./recipes";

export function exportLlamaIrToPyTorch(spec: LlamaArchitectureSpec): string {
  return renderLlamaModel(spec, renderLlamaPrimitives());
}
