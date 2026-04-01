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
};

