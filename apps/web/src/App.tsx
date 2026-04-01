import { useMemo, useState, type PointerEvent as ReactPointerEvent } from "react";
import { getBlockDefinition, type BlockEdge, type BlockField, type BlockNode, type ModelGraph } from "@neural-playground/block-schema";
import { validateGraph } from "@neural-playground/validator";
import { createNode, defaultTrainingConfig } from "./app/defaults";
import { useCanvasDrag } from "./app/useCanvasDrag";
import { useConnections } from "./app/useConnections";
import { useExport } from "./app/useExport";
import { useHistory } from "./app/useHistory";
import { useProject } from "./app/useProject";
import type { SelectedState } from "./app/types";
import { CanvasPanel } from "./components/CanvasPanel";
import { ConnectionsPanel } from "./components/ConnectionsPanel";
import { ExportPanel } from "./components/ExportPanel";
import { NodeInspector } from "./components/NodeInspector";
import { PaletteSidebar } from "./components/PaletteSidebar";
import { TrainingInspector } from "./components/TrainingInspector";

export function App() {
  const initialSnapshot = useMemo(() => {
    const initialNodes = [createNode("Input", 0), createNode("Embedding", 1), createNode("TransformerBlock", 2), createNode("Output", 3)];
    return {
      nodes: initialNodes,
      edges: [
        { id: "edge-1", source: initialNodes[0]!.id, target: initialNodes[1]!.id },
        { id: "edge-2", source: initialNodes[1]!.id, target: initialNodes[2]!.id },
        { id: "edge-3", source: initialNodes[2]!.id, target: initialNodes[3]!.id }
      ] as BlockEdge[],
      training: defaultTrainingConfig()
    };
  }, []);

  const { state, push, patch, commitDrag, reset, undo, redo, canUndo, canRedo } = useHistory(initialSnapshot);
  const { nodes, edges, training } = state;
  const [selected, setSelected] = useState<SelectedState>({ kind: "training" });

  const graph: ModelGraph = useMemo(() => ({ nodes, edges, training }), [nodes, edges, training]);
  const issues = useMemo(() => validateGraph(graph), [graph]);
  const selectedNode = selected?.kind === "node" ? nodes.find((node) => node.id === selected.nodeId) ?? null : null;

  const { canvasRef, draggingNodeId, startDraggingNode } = useCanvasDrag({
    state,
    patch,
    commitDrag,
    onNodeSelect: (nodeId) => setSelected({ kind: "node", nodeId })
  });

  const { pendingConnectionSourceId, connectionError, beginConnection, completeConnection, cancelConnection, resetConnections } = useConnections({ state, push });

  const { exportedJson, exportedPyTorch, exportedProject, trainingWarnings, copyStatus, exportPreview, setExportPreview, copyText, downloadProjectArchive, downloadTextFile } = useExport({ graph });

  const { projectStatus, setProjectStatus, loadInputRef, saveProject, loadFromFile } = useProject({ graph });

  function addNode(type: BlockNode["type"]) {
    const nextNode = createNode(type, nodes.length);
    push({ ...state, nodes: [...nodes, nextNode] });
    setSelected({ kind: "node", nodeId: nextNode.id });
    setProjectStatus("");
  }

  function updateNodeConfig(nodeId: string, field: BlockField, rawValue: string) {
    let nextValue: string | number | boolean = rawValue;
    if (field.type === "number") nextValue = Number(rawValue);
    else if (field.type === "boolean") nextValue = rawValue === "true";
    push({ ...state, nodes: nodes.map((node) => node.id !== nodeId ? node : { ...node, config: { ...node.config, [field.key]: nextValue } }) });
  }

  function renameNode(nodeId: string, name: string) {
    push({ ...state, nodes: nodes.map((n) => n.id !== nodeId ? n : { ...n, name: name || undefined }) });
  }

  function removeNode(nodeId: string) {
    push({ ...state, nodes: nodes.filter((n) => n.id !== nodeId), edges: edges.filter((e) => e.source !== nodeId && e.target !== nodeId) });
    setSelected({ kind: "training" });
    setProjectStatus("");
  }

  function removeEdge(edgeId: string) {
    push({ ...state, edges: edges.filter((e) => e.id !== edgeId) });
    setProjectStatus("");
  }

  async function handleLoadFile(event: React.ChangeEvent<HTMLInputElement>) {
    const snapshot = await loadFromFile(event);
    if (snapshot) {
      reset(snapshot);
      setSelected({ kind: "training" });
      resetConnections();
      setExportPreview(null);
    }
  }

  function preventFocusScroll(event: ReactPointerEvent<HTMLElement>) {
    event.preventDefault();
    event.stopPropagation();
  }

  return (
    <div className="app-shell">
      <PaletteSidebar
        projectStatus={projectStatus}
        loadInputRef={loadInputRef}
        onAddNode={addNode}
        onSave={saveProject}
        onLoadClick={() => loadInputRef.current?.click()}
        onLoadFile={handleLoadFile}
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
          canUndo={canUndo}
          canRedo={canRedo}
          onUndo={undo}
          onRedo={redo}
          onCancelConnection={cancelConnection}
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
            onRename={renameNode}
            onDelete={removeNode}
          />
        ) : (
          <TrainingInspector training={training} warnings={trainingWarnings} onChange={(next) => push({ ...state, training: next })} />
        )}

        <section className="issues-panel">
          <div className="panel-header">
            <p className="eyebrow">Validation</p>
            <h2>Issues</h2>
          </div>
          {issues.length === 0 ? <p className="success-copy">Graph looks valid at this layer.</p> : null}
          {issues.map((issue) => (
            <div key={`${issue.code}-${issue.message}-${issue.nodeId ?? ""}-${issue.edgeId ?? ""}`} className={`issue-card issue-card--${issue.severity ?? "error"}`}>
              <span className="issue-badge">{issue.severity ?? "error"}</span>
              <strong>{issue.code}</strong>
              <span>{issue.message}</span>
            </div>
          ))}
        </section>
      </aside>
    </div>
  );
}
