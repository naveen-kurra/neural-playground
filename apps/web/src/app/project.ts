import type { ChangeEvent } from "react";
import type { ModelGraph } from "@neural-playground/block-schema";
import type { ProjectDocument } from "./types";

export function isGraphLike(value: unknown): value is ModelGraph {
  if (!value || typeof value !== "object") {
    return false;
  }

  const graphValue = value as Partial<ModelGraph>;
  return Array.isArray(graphValue.nodes) && Array.isArray(graphValue.edges) && !!graphValue.training;
}

export function validateProjectDocument(value: unknown): ProjectDocument {
  if (!value || typeof value !== "object") {
    throw new Error("Project file must be a JSON object.");
  }

  if (isGraphLike(value)) {
    return {
      version: 1,
      graph: value
    };
  }

  const doc = value as Partial<ProjectDocument>;
  if (doc.version !== 1) {
    throw new Error("Unsupported project version.");
  }
  if (!isGraphLike(doc.graph)) {
    throw new Error("Project graph is incomplete.");
  }

  return {
    version: 1,
    graph: doc.graph
  };
}

export async function readProjectFromInput(event: ChangeEvent<HTMLInputElement>): Promise<{ document: ProjectDocument; fileName: string } | null> {
  const file = event.target.files?.[0];
  if (!file) {
    return null;
  }

  const text = await file.text();
  const parsed = JSON.parse(text);
  return {
    document: validateProjectDocument(parsed),
    fileName: file.name
  };
}
