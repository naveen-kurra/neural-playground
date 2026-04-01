export type ValidationIssue = {
  code: string;
  message: string;
  severity?: "error" | "warning";
  nodeId?: string;
  edgeId?: string;
};

export type ValidationMode =
  | "playground-valid"
  | "pytorch-export-valid"
  | "decoder-training-valid";

export function createValidationIssue(code: string, message: string, severity: "error" | "warning" = "error"): ValidationIssue {
  return { code, message, severity };
}
