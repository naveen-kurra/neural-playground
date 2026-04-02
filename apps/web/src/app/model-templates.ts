export type ModelTemplate = {
  id: string;
  label: string;
  family: "gpt2" | "llama";
  modelId: string;
  description: string;
  defaultBlockCount: number;
};

export const modelTemplates: ModelTemplate[] = [
  {
    id: "gpt2",
    label: "GPT-2",
    family: "gpt2",
    modelId: "gpt2",
    description: "Baseline GPT-2 decoder stack.",
    defaultBlockCount: 12
  },
  {
    id: "llama",
    label: "LLaMA",
    family: "llama",
    modelId: "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    description: "Public LLaMA-family decoder stack.",
    defaultBlockCount: 22
  }
];

export function resolveTemplate(selection: string): ModelTemplate | null {
  const normalized = selection.trim().toLowerCase();
  if (!normalized) {
    return null;
  }

  const match = modelTemplates.find(
    (template) =>
      template.id.toLowerCase() === normalized ||
      template.label.toLowerCase() === normalized ||
      template.modelId.toLowerCase() === normalized
  );

  return match ?? null;
}
