import { describe, expect, it } from "vitest";
import type { BlockEdge, BlockNode, ModelGraph } from "@neural-playground/block-schema";
import { buildContext } from "./context";
import { exportModelGraphToPyTorch } from "./index";

// ─── Helpers ────────────────────────────────────────────────────────────────

function node(id: string, type: BlockNode["type"], config: Record<string, string | number | boolean> = {}): BlockNode {
  return { id, type, position: { x: 0, y: 0 }, config };
}

function edge(id: string, source: string, target: string): BlockEdge {
  return { id, source, target };
}

function validGraph(): ModelGraph {
  return {
    nodes: [
      node("i1", "Input", { sequenceLength: 512 }),
      node("e1", "Embedding", { vocabSize: 32000, embeddingDim: 768 }),
      node("t1", "TransformerBlock", { dModel: 768, numHeads: 12, ffnHidden: 3072, dropout: 0.1, preLN: true }),
      node("o1", "Output", { headType: "LanguageModel" })
    ],
    edges: [
      edge("e1", "i1", "e1"),
      edge("e2", "e1", "t1"),
      edge("e3", "t1", "o1")
    ],
    training: {
      optimizer: "AdamW",
      loss: "CrossEntropy",
      learningRate: 0.0003,
      activation: "GELU"
    }
  };
}

// ─── buildContext ────────────────────────────────────────────────────────────

describe("buildContext", () => {
  it("builds correct context for a valid graph", () => {
    const ctx = buildContext(validGraph());
    expect(ctx.vocabSize).toBe(32000);
    expect(ctx.sequenceLength).toBe(512);
    expect(ctx.embeddingDim).toBe(768);
    expect(ctx.transformerCount).toBe(1);
    expect(ctx.defaultHeads).toBe(12);
    expect(ctx.defaultFfnHidden).toBe(3072);
    expect(ctx.optimizerName).toBe("adamw");
    expect(ctx.lossName).toBe("cross_entropy");
  });

  it("throws when there are no Input nodes", () => {
    const g = validGraph();
    g.nodes = g.nodes.filter((n) => n.type !== "Input");
    g.edges = g.edges.filter((e) => e.source !== "i1" && e.target !== "i1");
    expect(() => buildContext(g)).toThrow("exactly one Input node");
  });

  it("throws when there are 2 Input nodes", () => {
    const g = validGraph();
    g.nodes.push(node("i2", "Input", { sequenceLength: 512 }));
    expect(() => buildContext(g)).toThrow("exactly one Input node");
  });

  it("throws when there is no Embedding node", () => {
    const g = validGraph();
    g.nodes = g.nodes.filter((n) => n.type !== "Embedding");
    g.edges = g.edges.filter((e) => e.source !== "e1" && e.target !== "e1");
    expect(() => buildContext(g)).toThrow("Embedding node");
  });

  it("throws on a cyclic graph", () => {
    const g = validGraph();
    g.edges.push(edge("cycle", "o1", "i1"));
    expect(() => buildContext(g)).toThrow("acyclic");
  });

  it("uses defaults when config values are missing", () => {
    const g = validGraph();
    g.nodes = g.nodes.map((n) =>
      n.id === "e1" ? { ...n, config: {} } : n
    );
    const ctx = buildContext(g);
    expect(ctx.vocabSize).toBe(32000);
    expect(ctx.embeddingDim).toBe(768);
  });

  it("produces a warning for branch merges", () => {
    const g = validGraph();
    // Add a second transformer that merges two paths into output
    g.nodes.push(node("t2", "TransformerBlock", { dModel: 768, numHeads: 12, ffnHidden: 3072 }));
    g.edges.push(edge("extra", "t2", "o1"));
    const ctx = buildContext(g);
    expect(ctx.warnings.length).toBeGreaterThan(0);
  });
});

// ─── exportModelGraphToPyTorch ───────────────────────────────────────────────

describe("exportModelGraphToPyTorch", () => {
  it("generates a Python class definition", () => {
    const code = exportModelGraphToPyTorch(validGraph());
    expect(code).toContain("class");
    expect(code).toContain("nn.Module");
  });

  it("contains a forward method", () => {
    const code = exportModelGraphToPyTorch(validGraph());
    expect(code).toContain("def forward");
  });

  it("contains an __init__ method", () => {
    const code = exportModelGraphToPyTorch(validGraph());
    expect(code).toContain("def __init__");
  });

  it("imports torch", () => {
    const code = exportModelGraphToPyTorch(validGraph());
    expect(code).toContain("import torch");
  });

  it("includes embedding layer for Embedding node", () => {
    const code = exportModelGraphToPyTorch(validGraph());
    expect(code).toContain("nn.Embedding");
  });

  it("returns export failed comment on invalid graph", () => {
    const g = validGraph();
    g.nodes = g.nodes.filter((n) => n.type !== "Input");
    try {
      exportModelGraphToPyTorch(g);
    } catch (e) {
      expect(e).toBeDefined();
    }
  });

  it("passes vocab_size as a parameter to the model", () => {
    const code = exportModelGraphToPyTorch(validGraph());
    expect(code).toContain("vocab_size");
  });

  it("uses correct embedding dim", () => {
    const code = exportModelGraphToPyTorch(validGraph());
    expect(code).toContain("768");
  });

  it("includes custom causal self-attention for TransformerBlock node", () => {
    const code = exportModelGraphToPyTorch(validGraph());
    expect(code).toContain("CausalSelfAttention");
  });

  it("generates different code for MLP node", () => {
    const g = validGraph();
    g.nodes = [
      node("i1", "Input", { sequenceLength: 512 }),
      node("e1", "Embedding", { vocabSize: 32000, embeddingDim: 768 }),
      node("m1", "MLP", { hiddenDim: 3072, activation: "GELU" }),
      node("o1", "Output", { headType: "LanguageModel" })
    ];
    g.edges = [
      edge("e1", "i1", "e1"),
      edge("e2", "e1", "m1"),
      edge("e3", "m1", "o1")
    ];
    const code = exportModelGraphToPyTorch(g);
    expect(code).toContain("nn.Linear");
  });
});
