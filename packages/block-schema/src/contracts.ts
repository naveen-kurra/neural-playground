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

export type BlockRuleKind =
  | "number_gt"
  | "number_in_range"
  | "number_lte_field"
  | "number_divisible"
  | "number_lte_and_divides_field"
  | "number_equals_floor_div"
  | "sequence_dim_known"
  | "output_dim_known";

export type BlockRuleCondition = {
  field: string;
  equals?: string | number | boolean;
  notEquals?: string | number | boolean;
};

export type BlockRuleSpec = {
  code: string;
  severity: RuleSeverity;
  description: string;
  kind: BlockRuleKind;
  field?: string;
  otherField?: string;
  divisorField?: string;
  min?: number;
  max?: number;
  when?: BlockRuleCondition;
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
