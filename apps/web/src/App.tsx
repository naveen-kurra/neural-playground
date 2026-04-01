import { useEffect, useMemo, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";
import {
  blockDefinitions,
  getBlockDefinition,
  type BlockDefinition,
  type BlockEdge,
  type BlockField,
  type BlockNode,
  type BlockType,
  type ModelGraph,
  type TrainingConfig
} from "@neural-playground/block-schema";
import { exportModelGraphToPyTorch, exportProjectFiles } from "@neural-playground/exporter-pytorch";
import JSZip from "jszip";
import { validateGraph } from "@neural-playground/validator";

type SelectedState =
  | { kind: "node"; nodeId: string }
  | { kind: "training" }
  | null;

type ExportPreview = "json" | "pytorch" | null;
type CopyStatus = "idle" | "json-copied" | "pytorch-copied" | "project-downloaded" | "copy-failed";
type SafeExport<T> = { ok: true; value: T } | { ok: false; error: string };

function defaultTrainingConfig(): TrainingConfig {
  return {
    optimizer: "AdamW",
    loss: "CrossEntropy",
    learningRate: 0.0003,
    activation: "GELU",
    optimizerCustomName: "",
    lossCustomName: "",
    activationCustomName: ""
  };
}

function createNode(type: BlockType, index: number): BlockNode {
  const definition = getBlockDefinition(type);
  const config = definition.fields.reduce<Record<string, string | number | boolean>>((acc, field) => {
    acc[field.key] = field.defaultValue;
    return acc;
  }, {});

  return {
    id: `${type}-${crypto.randomUUID().slice(0, 8)}`,
    type,
    position: {
      x: 40 + (index % 3) * 240,
      y: 36 + Math.floor(index / 3) * 148
    },
    config
  };
}

function fieldValueAsString(value: string | number | boolean): string {
  if (typeof value === "boolean") {
    return value ? "true" : "false";
  }
  return String(value);
}

function getNodeAnchor(node: BlockNode) {
  return {
    left: node.position.x,
    top: node.position.y,
    right: node.position.x + 200,
    centerY: node.position.y + 32
  };
}

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
  const [training, setTraining] = useState<TrainingConfig>(defaultTrainingConfig);
  const [selected, setSelected] = useState<SelectedState>({ kind: "training" });
  const [draggingNodeId, setDraggingNodeId] = useState<string | null>(null);
  const [pendingConnectionSourceId, setPendingConnectionSourceId] = useState<string | null>(null);
  const [connectionError, setConnectionError] = useState<string>("");
  const [exportPreview, setExportPreview] = useState<ExportPreview>(null);
  const [copyStatus, setCopyStatus] = useState<CopyStatus>("idle");
  const dragStateRef = useRef<{
    nodeId: string;
    pointerOffsetX: number;
    pointerOffsetY: number;
  } | null>(null);
  const canvasRef = useRef<HTMLDivElement | null>(null);

  const graph: ModelGraph = useMemo(
    () => ({
      nodes,
      edges,
      training
    }),
    [nodes, edges, training]
  );

  const issues = useMemo(() => validateGraph(graph), [graph]);
  const selectedNode = selected?.kind === "node" ? nodes.find((node) => node.id === selected.nodeId) ?? null : null;
  const exportedPyTorch = useMemo(() => {
    try {
      return exportModelGraphToPyTorch(graph);
    } catch (error) {
      return `# Export failed\n# ${error instanceof Error ? error.message : "Unknown export error"}`;
    }
  }, [graph]);

  const exportedJson = useMemo(() => JSON.stringify(graph, null, 2), [graph]);
  const exportedProject = useMemo<SafeExport<ReturnType<typeof exportProjectFiles>>>(() => {
    try {
      return { ok: true, value: exportProjectFiles(graph) };
    } catch (error) {
      return {
        ok: false,
        error: error instanceof Error ? error.message : "Unknown project export error"
      };
    }
  }, [graph]);

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

  function addNode(type: BlockType) {
    const nextNode = createNode(type, nodes.length);
    setNodes((current) => [...current, nextNode]);
    setSelected({ kind: "node", nodeId: nextNode.id });
    setConnectionError("");
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
  }

  function removeEdge(edgeId: string) {
    setEdges((current) => current.filter((edge) => edge.id !== edgeId));
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
      const alreadyExists = current.some(
        (edge) => edge.source === pendingConnectionSourceId && edge.target === targetNodeId
      );
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

  function downloadTextFile(filename: string, contents: string) {
    const blob = new Blob([contents], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
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
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "neural-playground-export.zip";
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
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
      <aside className="sidebar palette">
        <div className="panel-header">
          <p className="eyebrow">Neural Playground</p>
          <h1>Block Library</h1>
        </div>
        <p className="panel-copy">Add architecture pieces to the canvas and wire them into a graph.</p>
        <div className="block-grid">
          {blockDefinitions.map((definition) => (
            <button type="button" key={definition.type} className="block-card" onClick={() => addNode(definition.type)}>
              <span className="block-category">{definition.category}</span>
              <strong>{definition.label}</strong>
              <span>{definition.description}</span>
            </button>
          ))}
        </div>
      </aside>

      <main className="workspace">
        <section className="canvas-panel">
          <div className="panel-header row">
            <div>
              <p className="eyebrow">Builder</p>
              <h2>Model Canvas</h2>
            </div>
            <div className="canvas-actions">
              {pendingConnectionSourceId ? (
                <button type="button" className="ghost-button" onClick={cancelPendingConnection}>
                  Cancel Connect
                </button>
              ) : null}
              <button type="button" className="ghost-button" onClick={() => setSelected({ kind: "training" })}>
                Training Config
              </button>
            </div>
          </div>

          {pendingConnectionSourceId ? (
            <p className="connection-hint">
              Connecting from <code>{pendingConnectionSourceId}</code>. Click an input handle on another block.
            </p>
          ) : (
            <p className="connection-hint">Use the right handle to start a connection and the left handle to finish it.</p>
          )}
          {connectionError ? <p className="connection-error">{connectionError}</p> : null}

          <div ref={canvasRef} className="canvas-grid">
            {nodes.length === 0 ? <p className="empty-canvas">Add blocks from the left panel to start building.</p> : null}
            <svg className="connection-overlay" aria-hidden="true">
              <defs>
                <marker
                  id="canvas-arrowhead"
                  markerWidth="10"
                  markerHeight="8"
                  refX="9"
                  refY="4"
                  orient="auto"
                  markerUnits="strokeWidth"
                >
                  <path d="M 0 0 L 10 4 L 0 8 z" fill="#76c1ff" />
                </marker>
              </defs>
              {edges.map((edge) => {
                const sourceNode = nodes.find((node) => node.id === edge.source);
                const targetNode = nodes.find((node) => node.id === edge.target);
                if (!sourceNode || !targetNode) {
                  return null;
                }

                const source = getNodeAnchor(sourceNode);
                const target = getNodeAnchor(targetNode);
                const startX = source.right;
                const startY = source.centerY;
                const endX = target.left;
                const endY = target.centerY;
                const controlOffset = Math.max(60, Math.abs(endX - startX) * 0.45);
                const path = `M ${startX} ${startY} C ${startX + controlOffset} ${startY}, ${endX - controlOffset} ${endY}, ${endX} ${endY}`;

                return (
                  <path
                    key={edge.id}
                    d={path}
                    className="connection-path"
                    markerEnd="url(#canvas-arrowhead)"
                  />
                );
              })}
            </svg>
            {nodes.map((node) => {
              const definition = getBlockDefinition(node.type);
              const isSelected = selected?.kind === "node" && selected.nodeId === node.id;
              const isDragging = draggingNodeId === node.id;

              return (
                <button
                  type="button"
                  key={node.id}
                  className={`canvas-node${isSelected ? " selected" : ""}${isDragging ? " dragging" : ""}`}
                  onClick={() => setSelected({ kind: "node", nodeId: node.id })}
                  onPointerDown={(event) => startDraggingNode(event, node)}
                  style={{
                    left: node.position.x,
                    top: node.position.y
                  }}
                >
                  <span
                    className={`node-handle node-handle-in${
                      pendingConnectionSourceId && pendingConnectionSourceId !== node.id ? " active" : ""
                    }`}
                    onPointerDown={preventFocusScroll}
                    onClick={(event) => {
                      event.preventDefault();
                      event.stopPropagation();
                      completeConnection(node.id);
                    }}
                  />
                  <span
                    className={`node-handle node-handle-out${pendingConnectionSourceId === node.id ? " active" : ""}`}
                    onPointerDown={preventFocusScroll}
                    onClick={(event) => {
                      event.preventDefault();
                      event.stopPropagation();
                      beginConnection(node.id);
                    }}
                  />
                  <span className="block-category">{definition.category}</span>
                  <strong>{definition.label}</strong>
                  <span className="node-meta">{node.id}</span>
                </button>
              );
            })}
          </div>
        </section>

        <section className="connections-panel">
          <div className="panel-header row">
            <div>
              <p className="eyebrow">Graph</p>
              <h2>Connections</h2>
            </div>
          </div>

          <p className="panel-copy">Connections are now created directly on the canvas using block handles.</p>

          <div className="edge-list">
            {edges.length === 0 ? <p className="muted">No connections yet.</p> : null}
            {edges.map((edge) => (
              <div key={edge.id} className="edge-row">
                <span>
                  {edge.source} → {edge.target}
                </span>
                <button type="button" className="inline-button" onClick={() => removeEdge(edge.id)}>
                  Remove
                </button>
              </div>
            ))}
          </div>
        </section>

        <section className="export-panel">
          <div className="panel-header row">
            <div>
              <p className="eyebrow">Export</p>
              <h2>Artifacts</h2>
            </div>
          </div>
          <div className="export-actions">
            <div className="export-card">
              <strong>Graph JSON</strong>
              <span>Serialized graph, nodes, edges, and training configuration.</span>
              <div className="export-buttons">
                <button type="button" className="ghost-button" onClick={() => setExportPreview("json")}>
                  View
                </button>
                <button type="button" className="ghost-button" onClick={() => copyText(exportedJson, "json")}>
                  Copy
                </button>
                <button type="button" className="ghost-button" onClick={() => downloadTextFile("model-graph.json", exportedJson)}>
                  Download
                </button>
              </div>
            </div>

            <div className="export-card">
              <strong>PyTorch Model</strong>
              <span>Generated `model.py` style export aligned to the current graph.</span>
              <div className="export-buttons">
                <button type="button" className="ghost-button" onClick={() => setExportPreview("pytorch")}>
                  View
                </button>
                <button type="button" className="ghost-button" onClick={() => copyText(exportedPyTorch, "pytorch")}>
                  Copy
                </button>
                <button type="button" className="ghost-button" onClick={() => downloadTextFile("model.py", exportedPyTorch)}>
                  Download
                </button>
              </div>
            </div>

            <div className="export-card">
              <strong>Full Project</strong>
              <span>Downloads a reusable training project scaffold with generated `model.py` and config files.</span>
              {!exportedProject.ok ? <span className="export-warning">Unavailable: {exportedProject.error}</span> : null}
              <div className="export-buttons">
                <button type="button" className="ghost-button" onClick={downloadProjectArchive} disabled={!exportedProject.ok}>
                  Download Project
                </button>
              </div>
            </div>
          </div>
          {copyStatus !== "idle" ? (
            <p className={`copy-status${copyStatus === "copy-failed" ? " error" : ""}`}>
              {copyStatus === "json-copied" ? "Graph JSON copied." : null}
              {copyStatus === "pytorch-copied" ? "PyTorch model copied." : null}
              {copyStatus === "project-downloaded" ? "Project archive downloaded." : null}
              {copyStatus === "copy-failed" ? "Copy failed in this browser." : null}
            </p>
          ) : null}
          {exportPreview ? (
            <div className="export-preview">
              <div className="panel-header row export-subheader">
                <div>
                  <p className="eyebrow">Preview</p>
                  <h2>{exportPreview === "json" ? "Graph JSON" : "PyTorch Model"}</h2>
                </div>
                <button type="button" className="ghost-button" onClick={() => setExportPreview(null)}>
                  Close
                </button>
              </div>
              <pre>{exportPreview === "json" ? exportedJson : exportedPyTorch}</pre>
            </div>
          ) : null}
        </section>
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
          <TrainingInspector training={training} onChange={setTraining} />
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

function NodeInspector(props: {
  node: BlockNode;
  definition: BlockDefinition;
  onChange: (nodeId: string, field: BlockField, rawValue: string) => void;
  onDelete: (nodeId: string) => void;
}) {
  const { node, definition, onChange, onDelete } = props;

  return (
    <section className="inspector-panel">
      <div className="panel-header">
        <p className="eyebrow">Inspector</p>
        <h2>{definition.label}</h2>
      </div>
      <p className="panel-copy">{definition.description}</p>

      <div className="form-stack">
        {definition.fields.map((field) => {
          const value = node.config[field.key] ?? field.defaultValue;
          return (
            <label key={field.key} className="field">
              <span>{field.label}</span>
              {field.type === "select" ? (
                <select value={fieldValueAsString(value)} onChange={(event) => onChange(node.id, field, event.target.value)}>
                  {field.options?.map((option) => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))}
                </select>
              ) : field.type === "boolean" ? (
                <select value={fieldValueAsString(value)} onChange={(event) => onChange(node.id, field, event.target.value)}>
                  <option value="true">True</option>
                  <option value="false">False</option>
                </select>
              ) : (
                <input
                  type={field.type === "number" ? "number" : "text"}
                  value={fieldValueAsString(value)}
                  onChange={(event) => onChange(node.id, field, event.target.value)}
                />
              )}
            </label>
          );
        })}
      </div>

      <div className="node-summary">
        <strong>Inputs</strong>
        <span>{definition.inputs.join(", ") || "none"}</span>
        <strong>Outputs</strong>
        <span>{definition.outputs.join(", ") || "none"}</span>
      </div>

      <button type="button" className="danger-button" onClick={() => onDelete(node.id)}>
        Delete Node
      </button>
    </section>
  );
}

function TrainingInspector(props: {
  training: TrainingConfig;
  onChange: (training: TrainingConfig) => void;
}) {
  const { training, onChange } = props;

  return (
    <section className="inspector-panel">
      <div className="panel-header">
        <p className="eyebrow">Inspector</p>
        <h2>Training Config</h2>
      </div>
      <p className="panel-copy">These settings describe how the graph would be trained or exported later.</p>

      <div className="form-stack">
        <label className="field">
          <span>Optimizer</span>
          <select value={training.optimizer} onChange={(event) => onChange({ ...training, optimizer: event.target.value as TrainingConfig["optimizer"] })}>
            <option value="AdamW">AdamW</option>
            <option value="SGD">SGD</option>
            <option value="Custom">Custom</option>
          </select>
        </label>
        {training.optimizer === "Custom" ? (
          <label className="field">
            <span>Custom Optimizer Name</span>
            <input
              type="text"
              value={training.optimizerCustomName ?? ""}
              placeholder="my_optimizer"
              onChange={(event) => onChange({ ...training, optimizerCustomName: event.target.value })}
            />
          </label>
        ) : null}

        <label className="field">
          <span>Loss</span>
          <select value={training.loss} onChange={(event) => onChange({ ...training, loss: event.target.value as TrainingConfig["loss"] })}>
            <option value="CrossEntropy">CrossEntropy</option>
            <option value="Custom">Custom</option>
          </select>
        </label>
        {training.loss === "Custom" ? (
          <label className="field">
            <span>Custom Loss Name</span>
            <input
              type="text"
              value={training.lossCustomName ?? ""}
              placeholder="my_loss"
              onChange={(event) => onChange({ ...training, lossCustomName: event.target.value })}
            />
          </label>
        ) : null}

        <label className="field">
          <span>Activation</span>
          <select
            value={training.activation}
            onChange={(event) => onChange({ ...training, activation: event.target.value as TrainingConfig["activation"] })}
          >
            <option value="GELU">GELU</option>
            <option value="ReLU">ReLU</option>
            <option value="SiLU">SiLU</option>
            <option value="Custom">Custom</option>
          </select>
        </label>
        {training.activation === "Custom" ? (
          <label className="field">
            <span>Custom Activation Name</span>
            <input
              type="text"
              value={training.activationCustomName ?? ""}
              placeholder="my_activation"
              onChange={(event) => onChange({ ...training, activationCustomName: event.target.value })}
            />
          </label>
        ) : null}

        <label className="field">
          <span>Learning Rate</span>
          <input
            type="number"
            step="0.0001"
            value={training.learningRate}
            onChange={(event) => onChange({ ...training, learningRate: Number(event.target.value) })}
          />
        </label>
      </div>
    </section>
  );
}
