export type ValidationIssue = {
  code: string;
  message: string;
  nodeId?: string;
  edgeId?: string;
};

export type ValidationMode =
  | "playground-valid"
  | "pytorch-export-valid"
  | "decoder-training-valid";

export function createValidationIssue(code: string, message: string): ValidationIssue {
  return { code, message };
}
