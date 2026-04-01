import { useEffect, useMemo, useRef, useState, type ChangeEvent, type PointerEvent as ReactPointerEvent } from "react";
import { getBlockDefinition, type BlockEdge, type BlockField, type BlockNode, type ModelGraph } from "@neural-playground/block-schema";
import { exportModelGraphToPyTorch, exportProjectFiles } from "@neural-playground/exporter-pytorch";
import JSZip from "jszip";
import { validateGraph } from "@neural-playground/validator";
import { createNode, defaultTrainingConfig } from "./app/defaults";
import { downloadBlobFile, downloadTextFile } from "./app/file-utils";
import { readProjectFromInput } from "./app/project";
import type { CopyStatus, ExportPreview, ProjectDocument, SafeExport, SelectedState } from "./app/types";
import { CanvasPanel } from "./components/CanvasPanel";
import { ConnectionsPanel } from "./components/ConnectionsPanel";
import { ExportPanel } from "./components/ExportPanel";
import { NodeInspector } from "./components/NodeInspector";
import { PaletteSidebar } from "./components/PaletteSidebar";
import { TrainingInspector } from "./components/TrainingInspector";

export function App() {
  const initialNodes = useMemo(
    () => [createNode("Input", 0), createNode("Embedding", 1), createNode("TransformerBlock", 2), createNode("Output", 3)],
    []
  );
  const [nodes, setNodes] = useState<BlockNode[]>(initialNodes);
  const [edges, setEdges] = useState<BlockEdge[]>([
    { id: "edge-1", source: initialNodes[0]!.id, target: initialNodes[1]!.id },
    { id: "edge-2", source: initialNodes[1]!.id, target: initialNodes[2]!.id },
    { id: "edge-3", source: initialNodes[2]!.id, target: initialNodes[3]!.id }
  ]);
  const [training, setTraining] = useState(defaultTrainingConfig);
  const [selected, setSelected] = useState<SelectedState>({ kind: "training" });
  const [draggingNodeId, setDraggingNodeId] = useState<string | null>(null);
  const [pendingConnectionSourceId, setPendingConnectionSourceId] = useState<string | null>(null);
  const [connectionError, setConnectionError] = useState("");
  const [exportPreview, setExportPreview] = useState<ExportPreview>(null);
  const [copyStatus, setCopyStatus] = useState<CopyStatus>("idle");
  const [projectStatus, setProjectStatus] = useState("");
  const dragStateRef = useRef<{ nodeId: string; pointerOffsetX: number; pointerOffsetY: number } | null>(null);
  const canvasRef = useRef<HTMLDivElement | null>(null);
  const loadInputRef = useRef<HTMLInputElement | null>(null);

  const graph: ModelGraph = useMemo(() => ({ nodes, edges, training }), [nodes, edges, training]);
  const issues = useMemo(() => validateGraph(graph), [graph]);
  const selectedNode = selected?.kind === "node" ? nodes.find((node) => node.id === selected.nodeId) ?? null : null;

  const trainingWarnings = useMemo(() => {
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
  }, [training]);

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

  useEffect(() => {
    function handlePointerMove(event: PointerEvent) {
      const dragState = dragStateRef.current;
      const canvas = canvasRef.current;
      if (!dragState || !canvas) {
        return;
      }

      const bounds = canvas.getBoundingClientRect();
      const nextX = event.clientX - bounds.left + canvas.scrollLeft - dragState.pointerOffsetX;
      const nextY = event.clientY - bounds.top + canvas.scrollTop - dragState.pointerOffsetY;

      setNodes((current) =>
        current.map((node) =>
          node.id === dragState.nodeId
            ? {
                ...node,
                position: {
                  x: Math.max(16, nextX),
                  y: Math.max(16, nextY)
                }
              }
            : node
        )
      );
    }

    function handlePointerUp() {
      dragStateRef.current = null;
      setDraggingNodeId(null);
    }

    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", handlePointerUp);
    return () => {
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", handlePointerUp);
    };
  }, []);

  function addNode(type: BlockNode["type"]) {
    const nextNode = createNode(type, nodes.length);
    setNodes((current) => [...current, nextNode]);
    setSelected({ kind: "node", nodeId: nextNode.id });
    setConnectionError("");
    setProjectStatus("");
  }

  function updateNodeConfig(nodeId: string, field: BlockField, rawValue: string) {
    setNodes((current) =>
      current.map((node) => {
        if (node.id !== nodeId) {
          return node;
        }

        let nextValue: string | number | boolean = rawValue;
        if (field.type === "number") {
          nextValue = Number(rawValue);
        } else if (field.type === "boolean") {
          nextValue = rawValue === "true";
        }

        return {
          ...node,
          config: {
            ...node.config,
            [field.key]: nextValue
          }
        };
      })
    );
  }

  function removeNode(nodeId: string) {
    setNodes((current) => current.filter((node) => node.id !== nodeId));
    setEdges((current) => current.filter((edge) => edge.source !== nodeId && edge.target !== nodeId));
    setSelected({ kind: "training" });
    setProjectStatus("");
  }

  function removeEdge(edgeId: string) {
    setEdges((current) => current.filter((edge) => edge.id !== edgeId));
    setProjectStatus("");
  }

  function beginConnection(nodeId: string) {
    setConnectionError("");
    setPendingConnectionSourceId(nodeId);
  }

  function completeConnection(targetNodeId: string) {
    if (!pendingConnectionSourceId) {
      return;
    }

    if (pendingConnectionSourceId === targetNodeId) {
      setPendingConnectionSourceId(null);
      setConnectionError("Source and target must be different blocks.");
      return;
    }

    const sourceNode = nodes.find((node) => node.id === pendingConnectionSourceId);
    const targetNode = nodes.find((node) => node.id === targetNodeId);
    if (!sourceNode || !targetNode) {
      setPendingConnectionSourceId(null);
      setConnectionError("Connection failed because one of the blocks no longer exists.");
      return;
    }

    const sourceDefinition = getBlockDefinition(sourceNode.type);
    const targetDefinition = getBlockDefinition(targetNode.type);
    const compatible = sourceDefinition.outputs.some((shape) => targetDefinition.inputs.includes(shape));
    if (!compatible) {
      setConnectionError(`${sourceDefinition.label} cannot connect to ${targetDefinition.label}.`);
      setPendingConnectionSourceId(null);
      return;
    }

    setEdges((current) => {
      const alreadyExists = current.some((edge) => edge.source === pendingConnectionSourceId && edge.target === targetNodeId);
      if (alreadyExists) {
        setConnectionError("That connection already exists.");
        return current;
      }

      return [
        ...current,
        {
          id: `edge-${crypto.randomUUID().slice(0, 8)}`,
          source: pendingConnectionSourceId,
          target: targetNodeId
        }
      ];
    });
    setConnectionError("");
    setPendingConnectionSourceId(null);
  }

  function cancelPendingConnection() {
    setPendingConnectionSourceId(null);
    setConnectionError("");
  }

  function preventFocusScroll(event: ReactPointerEvent<HTMLElement>) {
    event.preventDefault();
    event.stopPropagation();
  }

  function saveProject() {
    const document: ProjectDocument = {
      version: 1,
      graph
    };
    downloadTextFile("neural-playground.project.json", JSON.stringify(document, null, 2));
    setProjectStatus("Project saved.");
  }

  async function loadProjectFromFile(event: ChangeEvent<HTMLInputElement>) {
    try {
      const result = await readProjectFromInput(event);
      if (!result) {
        return;
      }

      setNodes(result.document.graph.nodes);
      setEdges(result.document.graph.edges);
      setTraining(result.document.graph.training);
      setSelected({ kind: "training" });
      setPendingConnectionSourceId(null);
      setConnectionError("");
      setExportPreview(null);
      setProjectStatus(`Loaded ${result.fileName}.`);
    } catch (error) {
      setProjectStatus(error instanceof Error ? error.message : "Failed to load project.");
    } finally {
      event.target.value = "";
    }
  }

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

  function startDraggingNode(event: ReactPointerEvent<HTMLButtonElement>, node: BlockNode) {
    if (!canvasRef.current) {
      return;
    }

    const nodeBounds = event.currentTarget.getBoundingClientRect();
    dragStateRef.current = {
      nodeId: node.id,
      pointerOffsetX: event.clientX - nodeBounds.left,
      pointerOffsetY: event.clientY - nodeBounds.top
    };
    setDraggingNodeId(node.id);
    setSelected({ kind: "node", nodeId: node.id });
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  return (
    <div className="app-shell">
      <PaletteSidebar
        projectStatus={projectStatus}
        loadInputRef={loadInputRef}
        onAddNode={addNode}
        onSave={saveProject}
        onLoadClick={() => loadInputRef.current?.click()}
        onLoadFile={loadProjectFromFile}
      />

      <main className="workspace">
        <CanvasPanel
          canvasRef={canvasRef}
          nodes={nodes}
          edges={edges}
          selected={selected}
          draggingNodeId={draggingNodeId}
          pendingConnectionSourceId={pendingConnectionSourceId}
          connectionError={connectionError}
          onCancelConnection={cancelPendingConnection}
          onShowTraining={() => setSelected({ kind: "training" })}
          onSelectNode={(nodeId) => setSelected({ kind: "node", nodeId })}
          onStartDrag={startDraggingNode}
          onBeginConnection={beginConnection}
          onCompleteConnection={completeConnection}
          onPreventHandleFocus={preventFocusScroll}
        />

        <ConnectionsPanel edges={edges} onRemoveEdge={removeEdge} />

        <ExportPanel
          exportedJson={exportedJson}
          exportedPyTorch={exportedPyTorch}
          exportedProject={exportedProject}
          exportPreview={exportPreview}
          copyStatus={copyStatus}
          onOpenPreview={setExportPreview}
          onCopy={copyText}
          onDownloadText={downloadTextFile}
          onDownloadProject={downloadProjectArchive}
        />
      </main>

      <aside className="sidebar inspector">
        {selectedNode ? (
          <NodeInspector
            node={selectedNode}
            definition={getBlockDefinition(selectedNode.type)}
            onChange={updateNodeConfig}
            onDelete={removeNode}
          />
        ) : (
          <TrainingInspector training={training} warnings={trainingWarnings} onChange={setTraining} />
        )}

        <section className="issues-panel">
          <div className="panel-header">
            <p className="eyebrow">Validation</p>
            <h2>Issues</h2>
          </div>
          {issues.length === 0 ? <p className="success-copy">Graph looks valid at this layer.</p> : null}
          {issues.map((issue) => (
            <div key={`${issue.code}-${issue.message}-${issue.nodeId ?? ""}-${issue.edgeId ?? ""}`} className="issue-card">
              <strong>{issue.code}</strong>
              <span>{issue.message}</span>
            </div>
          ))}
        </section>
      </aside>
    </div>
  );
}
