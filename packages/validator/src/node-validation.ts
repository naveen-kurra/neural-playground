import { getBlockDefinition, type BlockRuleCondition, type BlockRuleSpec, type ModelGraph } from "@neural-playground/block-schema";

import { inferNodeSequenceDim, numberConfig } from "./inference";
import type { ValidationIssue } from "./issues";

function fieldNumber(node: ModelGraph["nodes"][number], key: string | undefined): number | null {
  if (!key) return null;
  return numberConfig((node.config as Record<string, unknown>)[key]);
}

function conditionMatches(node: ModelGraph["nodes"][number], when: BlockRuleCondition | undefined): boolean {
  if (!when) return true;
  const value = (node.config as Record<string, unknown>)[when.field];
  if (when.equals !== undefined) {
    return value === when.equals;
  }
  if (when.notEquals !== undefined) {
    return value !== when.notEquals;
  }
  return true;
}

function issue(node: ModelGraph["nodes"][number], spec: BlockRuleSpec, message?: string): ValidationIssue {
  return {
    code: spec.code,
    message: message ?? spec.description,
    nodeId: node.id
  };
}

function missingRuleMetadataIssue(node: ModelGraph["nodes"][number], spec: BlockRuleSpec, fieldName: string): ValidationIssue {
  return issue(node, {
    ...spec,
    code: "unknown_validation_rule",
    description: `${node.type} rule '${spec.code}' is missing required metadata: ${fieldName}.`
  });
}

function validateRule(
  node: ModelGraph["nodes"][number],
  inferredSequenceDims: Map<string, number | null>,
  spec: BlockRuleSpec
): ValidationIssue[] {
  if (!conditionMatches(node, spec.when)) {
    return [];
  }

  switch (spec.kind) {
    case "number_gt": {
      if (!spec.field || spec.min === undefined) {
        return [missingRuleMetadataIssue(node, spec, "field|min")];
      }
      const value = fieldNumber(node, spec.field);
      if (value !== null && value > spec.min) {
        return [];
      }
      return [issue(node, spec, `${spec.description} Found ${spec.field}=${value ?? "null"}.`)];
    }
    case "number_in_range": {
      if (!spec.field || spec.min === undefined || spec.max === undefined) {
        return [missingRuleMetadataIssue(node, spec, "field|min|max")];
      }
      const value = fieldNumber(node, spec.field);
      if (value !== null && value >= spec.min && value <= spec.max) {
        return [];
      }
      return [issue(node, spec, `${spec.description} Found ${spec.field}=${value ?? "null"}.`)];
    }
    case "number_lte_field": {
      if (!spec.field || !spec.otherField || spec.min === undefined) {
        return [missingRuleMetadataIssue(node, spec, "field|otherField|min")];
      }
      const value = fieldNumber(node, spec.field);
      const maxValue = fieldNumber(node, spec.otherField);
      if (value !== null && maxValue !== null && value > spec.min && value <= maxValue) {
        return [];
      }
      return [issue(node, spec, `${spec.description} Found ${spec.field}=${value ?? "null"}, ${spec.otherField}=${maxValue ?? "null"}.`)];
    }
    case "number_divisible": {
      if (!spec.field || !spec.otherField) {
        return [missingRuleMetadataIssue(node, spec, "field|otherField")];
      }
      const numerator = fieldNumber(node, spec.field);
      const denominator = fieldNumber(node, spec.otherField);
      if (numerator !== null && denominator !== null && denominator > 0 && numerator % denominator === 0) {
        return [];
      }
      return [issue(node, spec, `${spec.description} Found ${spec.field}=${numerator ?? "null"}, ${spec.otherField}=${denominator ?? "null"}.`)];
    }
    case "number_lte_and_divides_field": {
      if (!spec.field || !spec.otherField || spec.min === undefined) {
        return [missingRuleMetadataIssue(node, spec, "field|otherField|min")];
      }
      const factor = fieldNumber(node, spec.field);
      const total = fieldNumber(node, spec.otherField);
      if (factor !== null && total !== null && factor > spec.min && factor <= total && total % factor === 0) {
        return [];
      }
      return [issue(node, spec, `${spec.description} Found ${spec.field}=${factor ?? "null"}, ${spec.otherField}=${total ?? "null"}.`)];
    }
    case "number_equals_floor_div": {
      if (!spec.field || !spec.otherField || !spec.divisorField) {
        return [missingRuleMetadataIssue(node, spec, "field|otherField|divisorField")];
      }
      const actual = fieldNumber(node, spec.field);
      const numerator = fieldNumber(node, spec.otherField);
      const divisor = fieldNumber(node, spec.divisorField);
      if (actual !== null && numerator !== null && divisor !== null && divisor > 0 && actual === Math.floor(numerator / divisor)) {
        return [];
      }
      return [issue(node, spec, `${spec.description} Found ${spec.field}=${actual ?? "null"}, ${spec.otherField}=${numerator ?? "null"}, ${spec.divisorField}=${divisor ?? "null"}.`)];
    }
    case "sequence_dim_known": {
      const inferredDim = inferNodeSequenceDim(node, inferredSequenceDims);
      if (inferredDim !== null) {
        return [];
      }
      return [issue(node, spec)];
    }
    case "output_dim_known": {
      const inferredDim = inferNodeSequenceDim(node, inferredSequenceDims);
      if (inferredDim !== null) {
        return [];
      }
      return [issue(node, spec)];
    }
    default:
      return [
        issue(node, {
          ...spec,
          code: "unknown_validation_rule",
          description: `${node.type} declares unsupported validation kind '${String((spec as { kind?: unknown }).kind)}'.`
        })
      ];
  }
}

export function validateNodeConfig(
  node: ModelGraph["nodes"][number],
  inferredSequenceDims: Map<string, number | null>
): ValidationIssue[] {
  const definition = getBlockDefinition(node.type);
  const issues: ValidationIssue[] = [];

  for (const spec of definition.ruleSpecs) {
    issues.push(...validateRule(node, inferredSequenceDims, spec));
  }

  return issues;
}
