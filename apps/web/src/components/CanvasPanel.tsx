import type { PointerEvent as ReactPointerEvent, RefObject } from "react";
import { getBlockDefinition, type BlockEdge, type BlockNode } from "@neural-playground/block-schema";
import { getNodeAnchor } from "../app/canvas";
import type { SelectedState } from "../app/types";

type CanvasPanelProps = {
  canvasRef: RefObject<HTMLDivElement | null>;
  nodes: BlockNode[];
  edges: BlockEdge[];
  selected: SelectedState;
  draggingNodeId: string | null;
  pendingConnectionSourceId: string | null;
  connectionError: string;
  onCancelConnection: () => void;
  onShowTraining: () => void;
  onSelectNode: (nodeId: string) => void;
  onStartDrag: (event: ReactPointerEvent<HTMLButtonElement>, node: BlockNode) => void;
  onBeginConnection: (nodeId: string) => void;
  onCompleteConnection: (nodeId: string) => void;
  onPreventHandleFocus: (event: ReactPointerEvent<HTMLElement>) => void;
};

export function CanvasPanel(props: CanvasPanelProps) {
  const {
    canvasRef,
    nodes,
    edges,
    selected,
    draggingNodeId,
    pendingConnectionSourceId,
    connectionError,
    onCancelConnection,
    onShowTraining,
    onSelectNode,
    onStartDrag,
    onBeginConnection,
    onCompleteConnection,
    onPreventHandleFocus
  } = props;

  return (
    <section className="canvas-panel">
      <div className="panel-header row">
        <div>
          <p className="eyebrow">Builder</p>
          <h2>Model Canvas</h2>
        </div>
        <div className="canvas-actions">
          {pendingConnectionSourceId ? (
            <button type="button" className="ghost-button" onClick={onCancelConnection}>
              Cancel Connect
            </button>
          ) : null}
          <button type="button" className="ghost-button" onClick={onShowTraining}>
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
            <marker id="canvas-arrowhead" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto" markerUnits="strokeWidth">
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

            return <path key={edge.id} d={path} className="connection-path" markerEnd="url(#canvas-arrowhead)" />;
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
              onClick={() => onSelectNode(node.id)}
              onPointerDown={(event) => onStartDrag(event, node)}
              style={{ left: node.position.x, top: node.position.y }}
            >
              <span
                className={`node-handle node-handle-in${pendingConnectionSourceId && pendingConnectionSourceId !== node.id ? " active" : ""}`}
                onPointerDown={onPreventHandleFocus}
                onClick={(event) => {
                  event.preventDefault();
                  event.stopPropagation();
                  onCompleteConnection(node.id);
                }}
              />
              <span
                className={`node-handle node-handle-out${pendingConnectionSourceId === node.id ? " active" : ""}`}
                onPointerDown={onPreventHandleFocus}
                onClick={(event) => {
                  event.preventDefault();
                  event.stopPropagation();
                  onBeginConnection(node.id);
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
  );
}
