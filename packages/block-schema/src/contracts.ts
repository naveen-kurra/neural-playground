import type { BlockCategory, BlockField, BlockType } from "./types";

export type ShapeArity = "sequence" | "logits" | "tokens";

export type ShapeDimensionRef =
  | "seq_len"
  | "vocab_size"
  | "d_model"
  | "hidden_dim"
  | "unknown";

export type ShapeContract = {
  kind: ShapeArity;
  dims: ShapeDimensionRef[];
};

export type RuleSeverity = "error" | "warning";

export type BlockRuleSpec = {
  code: string;
  severity: RuleSeverity;
  description: string;
};

export type BlockDefinition = {
  type: BlockType;
  label: string;
  category: BlockCategory;
  description: string;
  inputs: ShapeArity[];
  outputs: ShapeArity[];
  inputContracts: ShapeContract[];
  outputContracts: ShapeContract[];
  ruleSpecs: BlockRuleSpec[];
  fields: BlockField[];
  /** Config field that holds the explicit sequence/hidden dimension for this block (e.g. "dModel", "embeddingDim"). If absent, dimension is inferred from incoming edges. */
  sequenceDimField?: string;
  /** Config field that holds the vocabulary size for this block. Used for cross-node vocab mismatch detection. */
  vocabSizeField?: string;
};

