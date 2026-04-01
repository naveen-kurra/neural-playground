import { useMemo, useState } from "react";
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
import { validateGraph } from "@neural-playground/validator";

type SelectedState =
  | { kind: "node"; nodeId: string }
  | { kind: "training" }
  | null;

function defaultTrainingConfig(): TrainingConfig {
  return {
    optimizer: "AdamW",
    loss: "CrossEntropy",
    learningRate: 0.0003,
    activation: "GELU"
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

function connectionOptions(nodes: BlockNode[], currentNodeId: string): BlockNode[] {
  return nodes.filter((node) => node.id !== currentNodeId);
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
  const [pendingSource, setPendingSource] = useState<string>("");
  const [pendingTarget, setPendingTarget] = useState<string>("");

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

  function addNode(type: BlockType) {
    const nextNode = createNode(type, nodes.length);
    setNodes((current) => [...current, nextNode]);
    setSelected({ kind: "node", nodeId: nextNode.id });
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

  function addConnectionFromForm() {
    if (!pendingSource || !pendingTarget || pendingSource === pendingTarget) {
      return;
    }

    setEdges((current) => [
      ...current,
      {
        id: `edge-${crypto.randomUUID().slice(0, 8)}`,
        source: pendingSource,
        target: pendingTarget
      }
    ]);
    setPendingTarget("");
  }

  function removeEdge(edgeId: string) {
    setEdges((current) => current.filter((edge) => edge.id !== edgeId));
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
            <button key={definition.type} className="block-card" onClick={() => addNode(definition.type)}>
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
            <button className="ghost-button" onClick={() => setSelected({ kind: "training" })}>
              Training Config
            </button>
          </div>

          <div className="canvas-grid">
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

              return (
                <button
                  key={node.id}
                  className={`canvas-node${isSelected ? " selected" : ""}`}
                  onClick={() => setSelected({ kind: "node", nodeId: node.id })}
                  style={{
                    left: node.position.x,
                    top: node.position.y
                  }}
                >
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

          <div className="connection-form">
            <select value={pendingSource} onChange={(event) => setPendingSource(event.target.value)}>
              <option value="">Source node</option>
              {nodes.map((node) => (
                <option key={node.id} value={node.id}>
                  {node.type} ({node.id})
                </option>
              ))}
            </select>

            <select value={pendingTarget} onChange={(event) => setPendingTarget(event.target.value)}>
              <option value="">Target node</option>
              {connectionOptions(nodes, pendingSource).map((node) => (
                <option key={node.id} value={node.id}>
                  {node.type} ({node.id})
                </option>
              ))}
            </select>

            <button onClick={addConnectionFromForm}>Add Connection</button>
          </div>

          <div className="edge-list">
            {edges.length === 0 ? <p className="muted">No connections yet.</p> : null}
            {edges.map((edge) => (
              <div key={edge.id} className="edge-row">
                <span>
                  {edge.source} → {edge.target}
                </span>
                <button className="inline-button" onClick={() => removeEdge(edge.id)}>
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
              <h2>Graph JSON</h2>
            </div>
          </div>
          <pre>{JSON.stringify(graph, null, 2)}</pre>
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

      <button className="danger-button" onClick={() => onDelete(node.id)}>
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
          </select>
        </label>

        <label className="field">
          <span>Loss</span>
          <select value={training.loss} onChange={(event) => onChange({ ...training, loss: event.target.value as TrainingConfig["loss"] })}>
            <option value="CrossEntropy">CrossEntropy</option>
          </select>
        </label>

        <label className="field">
          <span>Activation</span>
          <select
            value={training.activation}
            onChange={(event) => onChange({ ...training, activation: event.target.value as TrainingConfig["activation"] })}
          >
            <option value="GELU">GELU</option>
            <option value="ReLU">ReLU</option>
            <option value="SiLU">SiLU</option>
          </select>
        </label>

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
