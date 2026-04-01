import { type ModelGraph } from "@neural-playground/block-schema";
import { buildContext, exportActivationName, normalizeActivation, sanitizeNodeId } from "./context";
import type { ExportContext } from "./types";

function renderModuleDefinitions(ctx: ExportContext): string[] {
  const lines: string[] = [];
  let currentDim = ctx.embeddingDim;

  for (const node of ctx.orderedNodes) {
    const name = sanitizeNodeId(node.id);

    if (node.type === "Input" || node.type === "Output") {
      continue;
    }

    if (node.type === "Embedding") {
      lines.push(`        self.${name}_token_emb = nn.Embedding(vocab_size, ${ctx.embeddingDim})`);
      lines.push(`        self.${name}_pos_emb = nn.Embedding(seq_len, ${ctx.embeddingDim})`);
      currentDim = ctx.embeddingDim;
      continue;
    }

    if (node.type === "TransformerBlock") {
      const dModel = Number(node.config.dModel ?? currentDim);
      const nHeads = Number(node.config.numHeads ?? ctx.defaultHeads);
      const ffnHidden = Number(node.config.ffnHidden ?? ctx.defaultFfnHidden);
      const activation = normalizeActivation(node.config.activation ?? ctx.defaultActivation);
      lines.push(
        `        self.${name} = TransformerBlock(d_model=${dModel}, n_heads=${nHeads}, ffn_hidden=${ffnHidden}, activation_name="${activation}")`
      );
      currentDim = dModel;
      continue;
    }

    if (node.type === "MLP") {
      const hiddenDim = Number(node.config.hiddenDim ?? currentDim * 4);
      const activation = normalizeActivation(node.config.activation ?? ctx.defaultActivation);
      lines.push(`        self.${name} = FeedForward(d_model=${currentDim}, ffn_hidden=${hiddenDim}, activation_name="${activation}")`);
      continue;
    }

    if (node.type === "LayerNorm") {
      const epsilon = Number(node.config.epsilon ?? 0.00001);
      lines.push(`        self.${name} = nn.LayerNorm(${currentDim}, eps=${epsilon})`);
      continue;
    }

    if (node.type === "Softmax") {
      const axis = Number(node.config.axis ?? -1);
      lines.push(`        self.${name} = nn.Softmax(dim=${axis})`);
    }
  }

  lines.push(`        self.lm_head = nn.Linear(${ctx.lastSequenceDim}, vocab_size)`);
  return lines;
}

function renderForward(ctx: ExportContext): string[] {
  const lines: string[] = [
    "    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:",
    "        bsz, seq_len = input_ids.shape",
    "        if seq_len > self.seq_len:",
    '            raise ValueError(f"Sequence length {seq_len} exceeds configured maximum {self.seq_len}")'
  ];

  let currentTensor = "input_ids";
  let embeddingSeen = false;

  for (const node of ctx.orderedNodes) {
    const name = sanitizeNodeId(node.id);

    if (node.type === "Input") {
      continue;
    }

    if (node.type === "Embedding") {
      lines.push("        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, seq_len)");
      lines.push(`        x = self.${name}_token_emb(input_ids) + self.${name}_pos_emb(positions)`);
      currentTensor = "x";
      embeddingSeen = true;
      continue;
    }

    if (!embeddingSeen) {
      throw new Error("Project export requires Embedding to appear before sequence-processing blocks.");
    }

    if (node.type === "Output") {
      lines.push(`        return self.lm_head(${currentTensor})`);
      return lines;
    }

    lines.push(`        ${currentTensor} = self.${name}(${currentTensor})`);
  }

  lines.push(`        return self.lm_head(${currentTensor})`);
  return lines;
}

export function exportModelGraphToPyTorch(graph: ModelGraph): string {
  const ctx = buildContext(graph);
  const activationName = exportActivationName(graph, ctx.defaultActivation);
  const warningHeader =
    ctx.warnings.length === 0 ? "" : ctx.warnings.map((warning) => `# Warning: ${warning}`).join("\n") + "\n\n";

  return `${warningHeader}from __future__ import annotations

import torch
from torch import nn


def build_causal_mask(seq_len: int, device: torch.device | None = None) -> torch.Tensor:
    return torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))


def get_activation(name: str) -> nn.Module:
    key = name.lower().strip()
    registry: dict[str, nn.Module] = {
        "gelu": nn.GELU(),
        "relu": nn.ReLU(),
        "silu": nn.SiLU(),
    }
    if key == "${activationName}" and key not in registry:
        raise ValueError("Implement custom activation '${activationName}' in model.py")
    if key not in registry:
        raise ValueError(f"Unsupported activation: {name}")
    return registry[key]


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        mask = build_causal_mask(seq_len, device=x.device)
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, ffn_hidden: int, activation_name: str = "gelu") -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, ffn_hidden)
        self.act = get_activation(activation_name)
        self.fc2 = nn.Linear(ffn_hidden, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_hidden: int, activation_name: str = "gelu") -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model=d_model, n_heads=n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model=d_model, ffn_hidden=ffn_hidden, activation_name=activation_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class DecoderLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_layers: int,
        d_model: int,
        n_heads: int,
        ffn_hidden: int,
        seq_len: int,
        activation_name: str = "${activationName}",
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
${renderModuleDefinitions(ctx).join("\n")}

${renderForward(ctx).join("\n")}
`;
}
