import { useEffect, useRef, useState } from "react";
import type { ModelGraph } from "@neural-playground/block-schema";

type GraphSnapshot = Pick<ModelGraph, "nodes" | "edges" | "training">;

const MAX_HISTORY = 50;

export function useHistory(initial: GraphSnapshot) {
  const [past, setPast] = useState<GraphSnapshot[]>([]);
  const [present, setPresent] = useState<GraphSnapshot>(initial);
  const [future, setFuture] = useState<GraphSnapshot[]>([]);

  const undoRef = useRef<() => void>(() => {});
  const redoRef = useRef<() => void>(() => {});

  function push(next: GraphSnapshot) {
    setPast((p) => [...p.slice(-(MAX_HISTORY - 1)), present]);
    setPresent(next);
    setFuture([]);
  }

  function patch(next: GraphSnapshot) {
    setPresent(next);
  }

  function commitDrag(preDragSnapshot: GraphSnapshot) {
    setPast((p) => [...p.slice(-(MAX_HISTORY - 1)), preDragSnapshot]);
    setFuture([]);
  }

  function reset(next: GraphSnapshot) {
    setPast([]);
    setPresent(next);
    setFuture([]);
  }

  function undo() {
    if (past.length === 0) return;
    const previous = past[past.length - 1]!;
    setPast((p) => p.slice(0, -1));
    setFuture((f) => [present, ...f]);
    setPresent(previous);
  }

  function redo() {
    if (future.length === 0) return;
    const next = future[0]!;
    setPast((p) => [...p, present]);
    setFuture((f) => f.slice(1));
    setPresent(next);
  }

  undoRef.current = undo;
  redoRef.current = redo;

  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      const modifier = event.metaKey || event.ctrlKey;
      if (!modifier) return;

      if (event.key === "z" && !event.shiftKey) {
        event.preventDefault();
        undoRef.current();
      } else if ((event.key === "z" && event.shiftKey) || event.key === "y") {
        event.preventDefault();
        redoRef.current();
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  return {
    state: present,
    push,
    patch,
    commitDrag,
    reset,
    undo,
    redo,
    canUndo: past.length > 0,
    canRedo: future.length > 0,
  };
}
