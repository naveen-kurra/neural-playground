export function countDense(inDim: number, outDim: number, bias = true): number {
  return inDim * outDim + (bias ? outDim : 0);
}

export function countEmbedding(vocabSize: number, hiddenSize: number): number {
  return vocabSize * hiddenSize;
}

export function countLayerNorm(hiddenSize: number, withBias = true): number {
  return withBias ? 2 * hiddenSize : hiddenSize;
}

export function countGpt2Attention(hiddenSize: number): number {
  return 4 * hiddenSize * hiddenSize + 4 * hiddenSize;
}

export function countGpt2Mlp(hiddenSize: number, intermediateSize: number): number {
  return 2 * hiddenSize * intermediateSize + intermediateSize + hiddenSize;
}

export function countTopKRouter(hiddenSize: number, numExperts: number, bias = true): number {
  return countDense(hiddenSize, numExperts, bias);
}

export function countTopKExperts(
  hiddenSize: number,
  expertHidden: number,
  numExperts: number,
  kind: "gpt2" | "llama"
): number {
  const perExpert =
    kind === "gpt2"
      ? 2 * hiddenSize * expertHidden + expertHidden + hiddenSize
      : hiddenSize * expertHidden * 2 + expertHidden * hiddenSize;
  return numExperts * perExpert;
}

export function countLlamaAttention(hiddenSize: number, numHeads: number, numKeyValueHeads: number, headDim: number, bias: boolean): number {
  return (
    countDense(hiddenSize, numHeads * headDim, bias) +
    countDense(hiddenSize, numKeyValueHeads * headDim, bias) +
    countDense(hiddenSize, numKeyValueHeads * headDim, bias) +
    countDense(numHeads * headDim, hiddenSize, bias)
  );
}

export function countLlamaMlp(hiddenSize: number, intermediateSize: number, bias: boolean): number {
  return (
    countDense(hiddenSize, intermediateSize, bias) +
    countDense(hiddenSize, intermediateSize, bias) +
    countDense(intermediateSize, hiddenSize, bias)
  );
}

