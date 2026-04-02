import type { BlockNode, ModelGraph } from "@neural-playground/block-schema";

export type ExportTarget = "pytorch";
export type ProjectFileMap = Record<string, string>;

export type ExportContext = {
  graph: ModelGraph;
  orderedNodes: BlockNode[];
  warnings: string[];
  vocabSize: number;
  sequenceLength: number;
  embeddingDim: number;
  transformerCount: number;
  defaultHeads: number;
  defaultFfnHidden: number;
  defaultActivation: string;
  optimizerName: string;
  lossName: string;
  lastSequenceDim: number;
};
