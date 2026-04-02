import { useEffect, useRef, useState, type PointerEvent as ReactPointerEvent, type RefObject } from "react";
import { getBlockDefinition, type BlockEdge, type BlockNode } from "@neural-playground/block-schema";
import { getNodeAnchor } from "../app/canvas";
import type { SelectedState } from "../app/types";

type CanvasPanelProps = {
  canvasRef: RefObject<HTMLDivElement | null>;
  parameterSummary: string;
  nodes: BlockNode[];
  edges: BlockEdge[];
  selected: SelectedState;
  draggingNodeId: string | null;
  pendingConnectionSourceId: string | null;
  connectionError: string;
  selectedEdgeId: string | null;
  zoom: number;
  pan: { x: number; y: number };
  onZoomChange: (zoom: number) => void;
  onPanChange: (pan: { x: number; y: number }) => void;
  canUndo: boolean;
  canRedo: boolean;
  onUndo: () => void;
  onRedo: () => void;
  onFitToScreen: () => void;
  onCancelConnection: () => void;
  onShowTraining: () => void;
  onSelectNode: (nodeId: string) => void;
  onSelectEdge: (edgeId: string | null) => void;
  onRemoveEdge: (edgeId: string) => void;
  onStartDrag: (event: ReactPointerEvent<HTMLButtonElement>, node: BlockNode) => void;
  onBeginConnection: (nodeId: string) => void;
  onCompleteConnection: (nodeId: string) => void;
  onPreventHandleFocus: (event: ReactPointerEvent<HTMLElement>) => void;
};

const MIN_ZOOM = 0.15;
const MAX_ZOOM = 3;

export function CanvasPanel(props: CanvasPanelProps) {
  const {
    canvasRef,
    parameterSummary,
    nodes,
    edges,
    selected,
    draggingNodeId,
    pendingConnectionSourceId,
    connectionError,
    selectedEdgeId,
    zoom,
    pan,
    onZoomChange,
    onPanChange,
    canUndo,
    canRedo,
    onUndo,
    onRedo,
    onFitToScreen,
    onCancelConnection,
    onShowTraining,
    onSelectNode,
    onSelectEdge,
    onRemoveEdge,
    onStartDrag,
    onBeginConnection,
    onCompleteConnection,
    onPreventHandleFocus
  } = props;

  const [isPanning, setIsPanning] = useState(false);
  const panDragRef = useRef<{ startX: number; startY: number; startPanX: number; startPanY: number } | null>(null);
  const zoomRef = useRef(zoom);
  const panRef = useRef(pan);
  useEffect(() => { zoomRef.current = zoom; }, [zoom]);
  useEffect(() => { panRef.current = pan; }, [pan]);

  // Wheel-to-zoom (non-passive so we can preventDefault)
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    function handleWheel(event: WheelEvent) {
      event.preventDefault();
      const bounds = canvas!.getBoundingClientRect();
      const cursorX = event.clientX - bounds.left;
      const cursorY = event.clientY - bounds.top;
      const currentZoom = zoomRef.current;
      const currentPan = panRef.current;
      // Normalize to ±1 so trackpad inertia & mouse wheel feel the same
      const direction = Math.sign(event.deltaY);
      const factor = direction < 0 ? 1.07 : 1 / 1.07;
      const newZoom = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, currentZoom * factor));
      const logX = (cursorX - currentPan.x) / currentZoom;
      const logY = (cursorY - currentPan.y) / currentZoom;
      onZoomChange(newZoom);
      onPanChange({ x: cursorX - logX * newZoom, y: cursorY - logY * newZoom });
    }

    canvas.addEventListener("wheel", handleWheel, { passive: false });
    return () => canvas.removeEventListener("wheel", handleWheel);
  }, [canvasRef, onZoomChange, onPanChange]);

  function handleCanvasPointerDown(event: ReactPointerEvent<HTMLDivElement>) {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const target = event.target as Element;
    const isBackground = target === canvas || target.classList.contains("canvas-viewport");
    if (!isBackground) return;
    panDragRef.current = {
      startX: event.clientX,
      startY: event.clientY,
      startPanX: panRef.current.x,
      startPanY: panRef.current.y
    };
    canvas.setPointerCapture(event.pointerId);
    setIsPanning(true);
  }

  function handleCanvasPointerMove(event: ReactPointerEvent<HTMLDivElement>) {
    const pd = panDragRef.current;
    if (!pd) return;
    onPanChange({
      x: pd.startPanX + event.clientX - pd.startX,
      y: pd.startPanY + event.clientY - pd.startY
    });
  }

  function handleCanvasPointerUp() {
    panDragRef.current = null;
    setIsPanning(false);
  }

  function stepZoom(direction: 1 | -1) {
    const factor = direction > 0 ? 1.25 : 1 / 1.25;
    onZoomChange(Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, zoom * factor)));
  }

  function resetView() {
    onZoomChange(1);
    onPanChange({ x: 0, y: 0 });
  }

  return (
    <section className="canvas-panel">
      <div className="panel-header row">
        <div>
          <p className="eyebrow">Builder</p>
          <h2>Model Canvas</h2>
        </div>
        <div className="canvas-actions">
          <div className="metric-pill">{parameterSummary}</div>
          <div className="zoom-controls">
            <button type="button" className="zoom-btn" onClick={() => stepZoom(-1)} title="Zoom out">−</button>
            <button type="button" className="zoom-level-btn" onClick={resetView} title="Reset view">{Math.round(zoom * 100)}%</button>
            <button type="button" className="zoom-btn" onClick={() => stepZoom(1)} title="Zoom in">+</button>
            <button type="button" className="zoom-btn zoom-fit-btn" onClick={onFitToScreen} title="Fit all nodes to screen">⊡</button>
          </div>
          <div className="zoom-controls">
            <button type="button" className="zoom-btn" onClick={onUndo} disabled={!canUndo} title="Undo (Cmd+Z)">↩</button>
            <button type="button" className="zoom-btn" onClick={onRedo} disabled={!canRedo} title="Redo (Cmd+Shift+Z)">↪</button>
          </div>
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
          Connecting from <code>{pendingConnectionSourceId}</code> — click an input handle to complete.
        </p>
      ) : (
        <p className="connection-hint">Scroll to zoom · Drag canvas to pan · Click an edge to delete it.</p>
      )}
      {connectionError ? <p className="connection-error">{connectionError}</p> : null}

      <div
        ref={canvasRef}
        className={`canvas-grid${isPanning ? " panning" : ""}`}
        onClick={() => onSelectEdge(null)}
        onPointerDown={handleCanvasPointerDown}
        onPointerMove={handleCanvasPointerMove}
        onPointerUp={handleCanvasPointerUp}
      >
        {nodes.length === 0 ? <p className="empty-canvas">Add blocks from the left panel to start building.</p> : null}
        <div
          className="canvas-viewport"
          style={{ transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})` }}
        >
          <svg className="connection-overlay" aria-hidden="false">
            <defs>
              <marker id="canvas-arrowhead" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto" markerUnits="strokeWidth">
                <path d="M 0 0 L 10 4 L 0 8 z" fill="#76c1ff" />
              </marker>
            </defs>
            {edges.map((edge) => {
              const sourceNode = nodes.find((node) => node.id === edge.source);
              const targetNode = nodes.find((node) => node.id === edge.target);
              if (!sourceNode || !targetNode) return null;

              const source = getNodeAnchor(sourceNode);
              const target = getNodeAnchor(targetNode);
              const startX = source.right;
              const startY = source.centerY;
              const endX = target.left;
              const endY = target.centerY;
              const controlOffset = Math.max(60, Math.abs(endX - startX) * 0.45);
              const path = `M ${startX} ${startY} C ${startX + controlOffset} ${startY}, ${endX - controlOffset} ${endY}, ${endX} ${endY}`;
              const midX = (startX + endX) / 2;
              const midY = (startY + endY) / 2;
              const isEdgeSelected = selectedEdgeId === edge.id;

              return (
                <g key={edge.id}>
                  <path
                    d={path}
                    className="connection-path-hitarea"
                    onPointerDown={(e) => e.stopPropagation()}
                    onClick={(e) => { e.stopPropagation(); onSelectEdge(isEdgeSelected ? null : edge.id); }}
                  />
                  <path
                    d={path}
                    className={`connection-path${isEdgeSelected ? " selected" : ""}`}
                    markerEnd="url(#canvas-arrowhead)"
                  />
                  {isEdgeSelected && (
                    <g
                      className="edge-delete-btn"
                      transform={`translate(${midX}, ${midY})`}
                      onPointerDown={(e) => e.stopPropagation()}
                      onClick={(e) => { e.stopPropagation(); onRemoveEdge(edge.id); }}
                    >
                      <circle r="11" className="edge-delete-circle" />
                      <text x="0" y="1" textAnchor="middle" dominantBaseline="middle" className="edge-delete-x">×</text>
                    </g>
                  )}
                </g>
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
                onClick={(e) => { e.stopPropagation(); onSelectNode(node.id); }}
                onPointerDown={(event) => { event.stopPropagation(); onStartDrag(event, node); }}
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
      </div>
    </section>
  );
}
