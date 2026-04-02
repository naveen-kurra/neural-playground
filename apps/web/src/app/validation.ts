import type { ValidationIssue } from "@neural-playground/validator";

export type FormattedValidationIssue = {
  title: string;
  severity: "error" | "warning";
  message: string;
};

const ISSUE_META: Record<string, { title: string; severity: "error" | "warning" }> = {
  graph_cycle: { title: "Cycle Detected", severity: "error" },
  export_input_count: { title: "Input Count", severity: "error" },
  export_output_count: { title: "Output Count", severity: "error" },
  export_embedding_count: { title: "Embedding Count", severity: "error" },
  decoder_output_head: { title: "Output Head Type", severity: "error" },
  dangling_edge: { title: "Broken Connection", severity: "error" },
  shape_mismatch: { title: "Connection Type Mismatch", severity: "error" },
  dimension_mismatch: { title: "Dimension Mismatch", severity: "error" },
  missing_input: { title: "Missing Input", severity: "error" },
  missing_output: { title: "Missing Output", severity: "error" },
  missing_incoming_edge: { title: "Unconnected Input", severity: "warning" },
  missing_outgoing_edge: { title: "Unconnected Output", severity: "warning" },
  invalid_d_model: { title: "Invalid Model Dimension", severity: "error" },
  invalid_num_heads: { title: "Invalid Head Count", severity: "error" },
  heads_dimension_mismatch: { title: "Head Split Mismatch", severity: "error" },
  invalid_ffn_hidden: { title: "Invalid FFN Hidden Size", severity: "error" },
  invalid_vocab_size: { title: "Invalid Vocab Size", severity: "error" },
  invalid_embedding_dim: { title: "Invalid Embedding Dimension", severity: "error" },
  invalid_hidden_dim: { title: "Invalid Hidden Dimension", severity: "error" },
  unknown_mlp_input_dim: { title: "Unknown MLP Input Size", severity: "warning" },
  invalid_sequence_length: { title: "Invalid Sequence Length", severity: "error" },
  unknown_layernorm_dim: { title: "Unknown LayerNorm Dimension", severity: "warning" },
  unknown_output_dim: { title: "Unknown Output Dimension", severity: "warning" },
  config_dim_mismatch: { title: "Dimension Mismatch", severity: "error" },
  config_vocab_mismatch: { title: "Vocab Size Mismatch", severity: "error" },
  invalid_kv_heads: { title: "Invalid KV Heads", severity: "error" },
  head_dim_mismatch: { title: "Head Dim Mismatch", severity: "error" },
  invalid_expert_count: { title: "Invalid Expert Count", severity: "error" },
  invalid_top_k: { title: "Invalid Top-K", severity: "error" },
  invalid_expert_hidden: { title: "Invalid Expert Hidden Size", severity: "error" }
};

function humanizeCode(code: string): string {
  return code
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

export function formatValidationIssue(issue: ValidationIssue): FormattedValidationIssue {
  const meta = ISSUE_META[issue.code] ?? {
    title: humanizeCode(issue.code),
    severity: issue.code.startsWith("unknown_") ? "warning" : "error"
  };

  return {
    title: meta.title,
    severity: meta.severity,
    message: issue.message
  };
}
