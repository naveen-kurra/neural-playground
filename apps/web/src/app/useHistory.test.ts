import { act, renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { useHistory } from "./useHistory";
import type { BlockEdge, BlockNode, TrainingConfig } from "@neural-playground/block-schema";

// ─── Helpers ────────────────────────────────────────────────────────────────

function makeNode(id: string): BlockNode {
  return { id, type: "Input", position: { x: 0, y: 0 }, config: {} };
}

function makeTraining(): TrainingConfig {
  return { optimizer: "AdamW", loss: "CrossEntropy", learningRate: 0.0003, activation: "GELU" };
}

function makeSnapshot(nodeIds: string[] = ["a"]) {
  return {
    nodes: nodeIds.map(makeNode),
    edges: [] as BlockEdge[],
    training: makeTraining()
  };
}

// ─── Initial state ───────────────────────────────────────────────────────────

describe("useHistory — initial state", () => {
  it("returns the initial snapshot as state", () => {
    const initial = makeSnapshot(["a", "b"]);
    const { result } = renderHook(() => useHistory(initial));
    expect(result.current.state).toEqual(initial);
  });

  it("canUndo is false initially", () => {
    const { result } = renderHook(() => useHistory(makeSnapshot()));
    expect(result.current.canUndo).toBe(false);
  });

  it("canRedo is false initially", () => {
    const { result } = renderHook(() => useHistory(makeSnapshot()));
    expect(result.current.canRedo).toBe(false);
  });
});

// ─── push ────────────────────────────────────────────────────────────────────

describe("useHistory — push", () => {
  it("updates state after push", () => {
    const { result } = renderHook(() => useHistory(makeSnapshot(["a"])));
    const next = makeSnapshot(["a", "b"]);
    act(() => result.current.push(next));
    expect(result.current.state).toEqual(next);
  });

  it("canUndo is true after push", () => {
    const { result } = renderHook(() => useHistory(makeSnapshot()));
    act(() => result.current.push(makeSnapshot(["x"])));
    expect(result.current.canUndo).toBe(true);
  });

  it("clears redo history on push", () => {
    const { result } = renderHook(() => useHistory(makeSnapshot(["a"])));
    act(() => result.current.push(makeSnapshot(["b"])));
    act(() => result.current.undo());
    expect(result.current.canRedo).toBe(true);
    act(() => result.current.push(makeSnapshot(["c"])));
    expect(result.current.canRedo).toBe(false);
  });
});

// ─── undo ────────────────────────────────────────────────────────────────────

describe("useHistory — undo", () => {
  it("reverts to previous state on undo", () => {
    const initial = makeSnapshot(["a"]);
    const { result } = renderHook(() => useHistory(initial));
    act(() => result.current.push(makeSnapshot(["a", "b"])));
    act(() => result.current.undo());
    expect(result.current.state).toEqual(initial);
  });

  it("canUndo is false after undoing all history", () => {
    const { result } = renderHook(() => useHistory(makeSnapshot()));
    act(() => result.current.push(makeSnapshot(["x"])));
    act(() => result.current.undo());
    expect(result.current.canUndo).toBe(false);
  });

  it("canRedo is true after undo", () => {
    const { result } = renderHook(() => useHistory(makeSnapshot()));
    act(() => result.current.push(makeSnapshot(["x"])));
    act(() => result.current.undo());
    expect(result.current.canRedo).toBe(true);
  });

  it("does nothing when there is no history to undo", () => {
    const initial = makeSnapshot(["a"]);
    const { result } = renderHook(() => useHistory(initial));
    act(() => result.current.undo());
    expect(result.current.state).toEqual(initial);
  });

  it("can undo multiple steps", () => {
    const s0 = makeSnapshot(["a"]);
    const s1 = makeSnapshot(["a", "b"]);
    const s2 = makeSnapshot(["a", "b", "c"]);
    const { result } = renderHook(() => useHistory(s0));
    act(() => result.current.push(s1));
    act(() => result.current.push(s2));
    act(() => result.current.undo());
    expect(result.current.state).toEqual(s1);
    act(() => result.current.undo());
    expect(result.current.state).toEqual(s0);
  });
});

// ─── redo ────────────────────────────────────────────────────────────────────

describe("useHistory — redo", () => {
  it("re-applies undone state on redo", () => {
    const s1 = makeSnapshot(["a", "b"]);
    const { result } = renderHook(() => useHistory(makeSnapshot(["a"])));
    act(() => result.current.push(s1));
    act(() => result.current.undo());
    act(() => result.current.redo());
    expect(result.current.state).toEqual(s1);
  });

  it("canRedo is false after redo", () => {
    const { result } = renderHook(() => useHistory(makeSnapshot()));
    act(() => result.current.push(makeSnapshot(["x"])));
    act(() => result.current.undo());
    act(() => result.current.redo());
    expect(result.current.canRedo).toBe(false);
  });

  it("does nothing when there is nothing to redo", () => {
    const initial = makeSnapshot(["a"]);
    const { result } = renderHook(() => useHistory(initial));
    act(() => result.current.redo());
    expect(result.current.state).toEqual(initial);
  });

  it("can redo multiple steps", () => {
    const s0 = makeSnapshot(["a"]);
    const s1 = makeSnapshot(["a", "b"]);
    const s2 = makeSnapshot(["a", "b", "c"]);
    const { result } = renderHook(() => useHistory(s0));
    act(() => result.current.push(s1));
    act(() => result.current.push(s2));
    act(() => result.current.undo());
    act(() => result.current.undo());
    act(() => result.current.redo());
    expect(result.current.state).toEqual(s1);
    act(() => result.current.redo());
    expect(result.current.state).toEqual(s2);
  });
});

// ─── patch ───────────────────────────────────────────────────────────────────

describe("useHistory — patch", () => {
  it("updates state without adding to history", () => {
    const { result } = renderHook(() => useHistory(makeSnapshot(["a"])));
    act(() => result.current.patch(makeSnapshot(["a", "b"])));
    expect(result.current.canUndo).toBe(false);
  });

  it("reflects patched state correctly", () => {
    const next = makeSnapshot(["a", "b"]);
    const { result } = renderHook(() => useHistory(makeSnapshot(["a"])));
    act(() => result.current.patch(next));
    expect(result.current.state).toEqual(next);
  });
});

// ─── commitDrag ──────────────────────────────────────────────────────────────

describe("useHistory — commitDrag", () => {
  it("commits pre-drag snapshot to history without changing present", () => {
    const initial = makeSnapshot(["a"]);
    const dragged = makeSnapshot(["a"]);
    dragged.nodes[0]!.position = { x: 100, y: 200 };
    const { result } = renderHook(() => useHistory(initial));
    act(() => result.current.patch(dragged));
    act(() => result.current.commitDrag(initial));
    expect(result.current.state).toEqual(dragged);
    expect(result.current.canUndo).toBe(true);
    act(() => result.current.undo());
    expect(result.current.state).toEqual(initial);
  });
});

// ─── reset ───────────────────────────────────────────────────────────────────

describe("useHistory — reset", () => {
  it("clears all history and sets new state", () => {
    const { result } = renderHook(() => useHistory(makeSnapshot(["a"])));
    act(() => result.current.push(makeSnapshot(["b"])));
    act(() => result.current.push(makeSnapshot(["c"])));
    const fresh = makeSnapshot(["x"]);
    act(() => result.current.reset(fresh));
    expect(result.current.state).toEqual(fresh);
    expect(result.current.canUndo).toBe(false);
    expect(result.current.canRedo).toBe(false);
  });
});

// ─── History limit ───────────────────────────────────────────────────────────

describe("useHistory — history limit", () => {
  it("caps history at 50 entries", () => {
    const { result } = renderHook(() => useHistory(makeSnapshot(["0"])));
    act(() => {
      for (let i = 1; i <= 55; i++) {
        result.current.push(makeSnapshot([String(i)]));
      }
    });
    // undo 50 times should hit the limit
    act(() => {
      for (let i = 0; i < 50; i++) {
        result.current.undo();
      }
    });
    expect(result.current.canUndo).toBe(false);
  });
});
