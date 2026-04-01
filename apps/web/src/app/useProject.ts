import { useRef, useState, type ChangeEvent } from "react";
import type { ModelGraph } from "@neural-playground/block-schema";
import { downloadTextFile } from "./file-utils";
import { readProjectFromInput } from "./project";
import type { ProjectDocument } from "./types";

type GraphSnapshot = Pick<ModelGraph, "nodes" | "edges" | "training">;

export function useProject({ graph }: { graph: ModelGraph }) {
  const [projectStatus, setProjectStatus] = useState("");
  const loadInputRef = useRef<HTMLInputElement | null>(null);

  function saveProject() {
    const document: ProjectDocument = { version: 1, graph };
    downloadTextFile("neural-playground.project.json", JSON.stringify(document, null, 2));
    setProjectStatus("Project saved.");
  }

  async function loadFromFile(event: ChangeEvent<HTMLInputElement>): Promise<GraphSnapshot | null> {
    try {
      const result = await readProjectFromInput(event);
      if (!result) return null;
      setProjectStatus(`Loaded ${result.fileName}.`);
      return {
        nodes: result.document.graph.nodes,
        edges: result.document.graph.edges,
        training: result.document.graph.training
      };
    } catch (error) {
      setProjectStatus(error instanceof Error ? error.message : "Failed to load project.");
      return null;
    } finally {
      event.target.value = "";
    }
  }

  return { projectStatus, setProjectStatus, loadInputRef, saveProject, loadFromFile };
}
