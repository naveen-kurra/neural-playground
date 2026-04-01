import { useEffect, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";
import type { BlockNode } from "@neural-playground/block-schema";
import type { useHistory } from "./useHistory";

type GraphState = ReturnType<typeof useHistory>["state"];

type UseCanvasDragOptions = {
  state: GraphState;
  patch: (next: GraphState) => void;
  commitDrag: (preDragSnapshot: GraphState) => void;
  onNodeSelect: (nodeId: string) => void;
};

export function useCanvasDrag({ state, patch, commitDrag, onNodeSelect }: UseCanvasDragOptions) {
  const canvasRef = useRef<HTMLDivElement | null>(null);
  const dragStateRef = useRef<{ nodeId: string; pointerOffsetX: number; pointerOffsetY: number } | null>(null);
  const dragStartSnapshotRef = useRef<GraphState | null>(null);
  const stateRef = useRef(state);
  stateRef.current = state;
  const patchRef = useRef(patch);
  patchRef.current = patch;
  const commitDragRef = useRef(commitDrag);
  commitDragRef.current = commitDrag;
  const [draggingNodeId, setDraggingNodeId] = useState<string | null>(null);

  useEffect(() => {
    function handlePointerMove(event: PointerEvent) {
      const dragState = dragStateRef.current;
      const canvas = canvasRef.current;
      if (!dragState || !canvas) return;

      const bounds = canvas.getBoundingClientRect();
      const nextX = event.clientX - bounds.left + canvas.scrollLeft - dragState.pointerOffsetX;
      const nextY = event.clientY - bounds.top + canvas.scrollTop - dragState.pointerOffsetY;

      const currentState = stateRef.current;
      patchRef.current({
        ...currentState,
        nodes: currentState.nodes.map((node) =>
          node.id === dragState.nodeId
            ? { ...node, position: { x: Math.max(16, nextX), y: Math.max(16, nextY) } }
            : node
        )
      });
    }

    function handlePointerUp() {
      if (dragStateRef.current && dragStartSnapshotRef.current) {
        commitDragRef.current(dragStartSnapshotRef.current);
        dragStartSnapshotRef.current = null;
      }
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

  function startDraggingNode(event: ReactPointerEvent<HTMLButtonElement>, node: BlockNode) {
    if (!canvasRef.current) return;

    const nodeBounds = event.currentTarget.getBoundingClientRect();
    dragStartSnapshotRef.current = stateRef.current;
    dragStateRef.current = {
      nodeId: node.id,
      pointerOffsetX: event.clientX - nodeBounds.left,
      pointerOffsetY: event.clientY - nodeBounds.top
    };
    setDraggingNodeId(node.id);
    onNodeSelect(node.id);
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  return { canvasRef, draggingNodeId, startDraggingNode };
}
