import { useMemo, useState } from "react";
import { exportModelGraphToPyTorch, exportProjectFiles } from "@neural-playground/exporter-pytorch";
import JSZip from "jszip";
import type { ModelGraph } from "@neural-playground/block-schema";
import { downloadBlobFile, downloadTextFile } from "./file-utils";
import type { CopyStatus, ExportPreview, SafeExport } from "./types";

export function useExport({ graph }: { graph: ModelGraph }) {
  const [copyStatus, setCopyStatus] = useState<CopyStatus>("idle");
  const [exportPreview, setExportPreview] = useState<ExportPreview>(null);

  const trainingWarnings = useMemo(() => {
    const { training } = graph;
    const warnings: string[] = [];
    if (training.optimizer === "Custom" && !(training.optimizerCustomName ?? "").trim()) {
      warnings.push("Custom optimizer selected but no optimizer name is provided.");
    }
    if (training.loss === "Custom" && !(training.lossCustomName ?? "").trim()) {
      warnings.push("Custom loss selected but no loss name is provided.");
    }
    if (training.activation === "Custom" && !(training.activationCustomName ?? "").trim()) {
      warnings.push("Custom activation selected but no activation name is provided.");
    }
    return warnings;
  }, [graph]);

  const exportedPyTorch = useMemo(() => {
    try {
      return exportModelGraphToPyTorch(graph);
    } catch (error) {
      return `# Export failed\n# ${error instanceof Error ? error.message : "Unknown export error"}`;
    }
  }, [graph]);

  const exportedJson = useMemo(() => JSON.stringify(graph, null, 2), [graph]);

  const exportedProject = useMemo<SafeExport<ReturnType<typeof exportProjectFiles>>>(() => {
    if (trainingWarnings.length > 0) {
      return { ok: false, error: trainingWarnings[0]! };
    }
    try {
      return { ok: true, value: exportProjectFiles(graph) };
    } catch (error) {
      return { ok: false, error: error instanceof Error ? error.message : "Unknown project export error" };
    }
  }, [graph, trainingWarnings]);

  async function copyText(contents: string, artifact: "json" | "pytorch") {
    try {
      await navigator.clipboard.writeText(contents);
      setCopyStatus(artifact === "json" ? "json-copied" : "pytorch-copied");
    } catch {
      setCopyStatus("copy-failed");
    }
  }

  async function downloadProjectArchive() {
    if (!exportedProject.ok) {
      setCopyStatus("copy-failed");
      return;
    }
    try {
      const zip = new JSZip();
      for (const [path, contents] of Object.entries(exportedProject.value)) {
        zip.file(path, contents);
      }
      const blob = await zip.generateAsync({ type: "blob" });
      await downloadBlobFile("neural-playground-export.zip", blob);
      setCopyStatus("project-downloaded");
    } catch {
      setCopyStatus("copy-failed");
    }
  }

  return {
    exportedJson,
    exportedPyTorch,
    exportedProject,
    trainingWarnings,
    copyStatus,
    exportPreview,
    setExportPreview,
    copyText,
    downloadProjectArchive: downloadProjectArchive,
    downloadTextFile
  };
}
