import { describe, expect, it } from "vitest";
import type { BlockEdge, BlockNode, ModelGraph } from "@neural-playground/block-schema";
import { validateGraph } from "./index";

// ─── Helpers ────────────────────────────────────────────────────────────────

function node(id: string, type: BlockNode["type"], config: Record<string, string | number | boolean> = {}): BlockNode {
  return { id, type, position: { x: 0, y: 0 }, config };
}

function edge(id: string, source: string, target: string): BlockEdge {
  return { id, source, target };
}

function graph(nodes: BlockNode[], edges: BlockEdge[]): ModelGraph {
  return {
    nodes,
    edges,
    training: {
      optimizer: "AdamW",
      loss: "CrossEntropy",
      learningRate: 0.0003,
      activation: "GELU"
    }
  };
}

function validGraph(): ModelGraph {
  return graph(
    [
      node("i1", "Input", { sequenceLength: 512 }),
      node("e1", "Embedding", { vocabSize: 32000, embeddingDim: 768 }),
      node("t1", "TransformerBlock", { dModel: 768, numHeads: 12, ffnHidden: 3072, dropout: 0.1, preLN: true }),
      node("o1", "Output", { headType: "LanguageModel" })
    ],
    [edge("e1", "i1", "e1"), edge("e2", "e1", "t1"), edge("e3", "t1", "o1")]
  );
}

// ─── Graph-level structure ───────────────────────────────────────────────────

describe("validateGraph — structure", () => {
  it("returns no issues for a valid graph", () => {
    const issues = validateGraph(validGraph());
    expect(issues).toHaveLength(0);
  });

  it("errors when there are no Input nodes", () => {
    const g = validGraph();
    g.nodes = g.nodes.filter((n) => n.type !== "Input");
    g.edges = g.edges.filter((e) => e.source !== "i1" && e.target !== "i1");
    const issues = validateGraph(g);
    expect(issues.some((i) => i.code === "missing_input")).toBe(true);
  });

  it("errors when there are no Output nodes", () => {
    const g = validGraph();
    g.nodes = g.nodes.filter((n) => n.type !== "Output");
    g.edges = g.edges.filter((e) => e.source !== "o1" && e.target !== "o1");
    const issues = validateGraph(g);
    expect(issues.some((i) => i.code === "missing_output")).toBe(true);
  });

  it("warns when there are 2 Input nodes", () => {
    const g = validGraph();
    g.nodes.push(node("i2", "Input", { sequenceLength: 512 }));
    const issues = validateGraph(g);
    const issue = issues.find((i) => i.code === "export_input_count");
    expect(issue).toBeDefined();
    expect(issue?.severity).toBe("warning");
  });

  it("warns when there are 2 Output nodes", () => {
    const g = validGraph();
    g.nodes.push(node("o2", "Output", { headType: "LanguageModel" }));
    const issues = validateGraph(g);
    const issue = issues.find((i) => i.code === "export_output_count");
    expect(issue).toBeDefined();
    expect(issue?.severity).toBe("warning");
  });

  it("warns when there is no Embedding node", () => {
    const g = validGraph();
    g.nodes = g.nodes.filter((n) => n.type !== "Embedding");
    g.edges = g.edges.filter((e) => e.source !== "e1" && e.target !== "e1");
    const issues = validateGraph(g);
    const issue = issues.find((i) => i.code === "export_embedding_count");
    expect(issue).toBeDefined();
    expect(issue?.severity).toBe("warning");
  });

  it("warns when there are 2 Embedding nodes", () => {
    const g = validGraph();
    g.nodes.push(node("e2", "Embedding", { vocabSize: 32000, embeddingDim: 768 }));
    const issues = validateGraph(g);
    const issue = issues.find((i) => i.code === "export_embedding_count");
    expect(issue).toBeDefined();
    expect(issue?.severity).toBe("warning");
  });

  it("errors on a cyclic graph", () => {
    const g = graph(
      [node("a", "Input"), node("b", "TransformerBlock"), node("c", "Output")],
      [edge("e1", "a", "b"), edge("e2", "b", "c"), edge("e3", "c", "b")]
    );
    const issues = validateGraph(g);
    expect(issues.some((i) => i.code === "graph_cycle")).toBe(true);
  });
});

// ─── Node connectivity ───────────────────────────────────────────────────────

describe("validateGraph — connectivity", () => {
  it("errors when a non-Input node has no incoming connection", () => {
    const g = validGraph();
    g.edges = g.edges.filter((e) => e.target !== "e1");
    const issues = validateGraph(g);
    expect(issues.some((i) => i.code === "missing_incoming_edge" && i.nodeId === "e1")).toBe(true);
  });

  it("errors when a non-Output node has no outgoing connection", () => {
    const g = validGraph();
    g.edges = g.edges.filter((e) => e.source !== "t1");
    const issues = validateGraph(g);
    expect(issues.some((i) => i.code === "missing_outgoing_edge" && i.nodeId === "t1")).toBe(true);
  });
});

// ─── Node config validation ──────────────────────────────────────────────────

describe("validateGraph — TransformerBlock config", () => {
  it("errors when d_model is 0", () => {
    const g = validGraph();
    g.nodes = g.nodes.map((n) => n.id === "t1" ? { ...n, config: { ...n.config, dModel: 0 } } : n);
    const issues = validateGraph(g);
    expect(issues.some((i) => i.code === "invalid_d_model")).toBe(true);
  });

  it("errors when num_heads is 0", () => {
    const g = validGraph();
    g.nodes = g.nodes.map((n) => n.id === "t1" ? { ...n, config: { ...n.config, numHeads: 0 } } : n);
    const issues = validateGraph(g);
    expect(issues.some((i) => i.code === "invalid_num_heads")).toBe(true);
  });

  it("errors when d_model is not divisible by num_heads", () => {
    const g = validGraph();
    g.nodes = g.nodes.map((n) => n.id === "t1" ? { ...n, config: { ...n.config, dModel: 100, numHeads: 3 } } : n);
    const issues = validateGraph(g);
    expect(issues.some((i) => i.code === "heads_dimension_mismatch")).toBe(true);
  });

  it("no error when d_model is divisible by num_heads", () => {
    const g = validGraph();
    g.nodes = g.nodes.map((n) => n.id === "t1" ? { ...n, config: { ...n.config, dModel: 768, numHeads: 12 } } : n);
    const issues = validateGraph(g);
    expect(issues.some((i) => i.code === "heads_dimension_mismatch")).toBe(false);
  });

  it("errors when ffn_hidden is 0", () => {
    const g = validGraph();
    g.nodes = g.nodes.map((n) => n.id === "t1" ? { ...n, config: { ...n.config, ffnHidden: 0 } } : n);
    const issues = validateGraph(g);
    expect(issues.some((i) => i.code === "invalid_ffn_hidden")).toBe(true);
  });
});

describe("validateGraph — Embedding config", () => {
  it("errors when vocab_size is 0", () => {
    const g = validGraph();
    g.nodes = g.nodes.map((n) => n.id === "e1" ? { ...n, config: { ...n.config, vocabSize: 0 } } : n);
    const issues = validateGraph(g);
    expect(issues.some((i) => i.code === "invalid_vocab_size")).toBe(true);
  });

  it("errors when embedding_dim is 0", () => {
    const g = validGraph();
    g.nodes = g.nodes.map((n) => n.id === "e1" ? { ...n, config: { ...n.config, embeddingDim: 0 } } : n);
    const issues = validateGraph(g);
    expect(issues.some((i) => i.code === "invalid_embedding_dim")).toBe(true);
  });
});

describe("validateGraph — Input config", () => {
  it("errors when sequence_length is 1", () => {
    const g = validGraph();
    g.nodes = g.nodes.map((n) => n.id === "i1" ? { ...n, config: { ...n.config, sequenceLength: 1 } } : n);
    const issues = validateGraph(g);
    expect(issues.some((i) => i.code === "invalid_sequence_length")).toBe(true);
  });

  it("no error when sequence_length is 2 or more", () => {
    const g = validGraph();
    g.nodes = g.nodes.map((n) => n.id === "i1" ? { ...n, config: { ...n.config, sequenceLength: 512 } } : n);
    const issues = validateGraph(g);
    expect(issues.some((i) => i.code === "invalid_sequence_length")).toBe(false);
  });
});

// ─── Severity ────────────────────────────────────────────────────────────────

describe("validateGraph — severity", () => {
  it("export warnings have severity: warning", () => {
    const g = validGraph();
    g.nodes.push(node("i2", "Input", { sequenceLength: 512 }));
    const issues = validateGraph(g);
    const warnings = issues.filter((i) => i.severity === "warning");
    expect(warnings.length).toBeGreaterThan(0);
  });

  it("structural errors have severity: error", () => {
    const g = validGraph();
    g.edges = g.edges.filter((e) => e.target !== "e1");
    const issues = validateGraph(g);
    const errors = issues.filter((i) => (i.severity ?? "error") === "error");
    expect(errors.length).toBeGreaterThan(0);
  });
});
