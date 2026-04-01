import type { BlockNode } from "@neural-playground/block-schema";

export function fieldValueAsString(value: string | number | boolean): string {
  if (typeof value === "boolean") {
    return value ? "true" : "false";
  }
  return String(value);
}

export function getNodeAnchor(node: BlockNode) {
  return {
    left: node.position.x,
    top: node.position.y,
    right: node.position.x + 200,
    centerY: node.position.y + 32
  };
}
