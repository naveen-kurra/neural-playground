import { getBlockDefinition, type BlockType } from "@neural-playground/block-schema";

export function getRuleMessage(blockType: BlockType, code: string, fallback: string): string {
  const definition = getBlockDefinition(blockType);
  const spec = definition.ruleSpecs.find((rule) => rule.code === code);
  return spec?.description ?? fallback;
}

