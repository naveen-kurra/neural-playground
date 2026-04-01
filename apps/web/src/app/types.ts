import type { ModelGraph } from "@neural-playground/block-schema";

export type SelectedState =
  | { kind: "node"; nodeId: string }
  | { kind: "training" }
  | null;

export type ExportPreview = "json" | "pytorch" | null;
export type CopyStatus = "idle" | "json-copied" | "pytorch-copied" | "project-downloaded" | "copy-failed";
export type SafeExport<T> = { ok: true; value: T } | { ok: false; error: string };

export type ProjectDocument = {
  version: 1;
  graph: ModelGraph;
};
