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
      lines.push(`        self.${name} = TransformerBlock(d_model=${dModel}, n_heads=${nHeads})`);
      currentDim = dModel;
      continue;
    }

    if (node.type === "MoE") {
      const expertHidden = Number(node.config.expertHidden ?? ctx.defaultFfnHidden);
      const numExperts = Number(node.config.numExperts ?? 8);
      const topK = Number(node.config.topK ?? 2);
      const activation = normalizeActivation(node.config.activation ?? ctx.defaultActivation);
      lines.push(
        `        self.${name} = TopKMoE(d_model=${currentDim}, expert_hidden=${expertHidden}, num_experts=${numExperts}, top_k=${topK}, activation_name="${activation}")`
      );
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

  const incomingByNode = new Map<string, string[]>();
  for (const node of ctx.graph.nodes) {
    incomingByNode.set(node.id, []);
  }
  for (const edge of ctx.graph.edges) {
    incomingByNode.get(edge.target)?.push(edge.source);
  }

  const tensorByNode = new Map<string, string>();

  for (const node of ctx.orderedNodes) {
    const name = sanitizeNodeId(node.id);
    const tensorName = `tensor_${name}`;
    const incoming = incomingByNode.get(node.id) ?? [];

    if (node.type === "Input") {
      tensorByNode.set(node.id, "input_ids");
      continue;
    }

    if (node.type === "Embedding") {
      lines.push("        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, seq_len)");
      lines.push(`        ${tensorName} = self.${name}_token_emb(input_ids) + self.${name}_pos_emb(positions)`);
      tensorByNode.set(node.id, tensorName);
      continue;
    }

    if (node.type === "Add") {
      if (incoming.length !== 2) {
        throw new Error("Add export currently requires exactly two incoming edges.");
      }
      const left = tensorByNode.get(incoming[0]!);
      const right = tensorByNode.get(incoming[1]!);
      if (!left || !right) {
        throw new Error("Add export requires both inputs to be available earlier in topological order.");
      }
      lines.push(`        ${tensorName} = ${left} + ${right}`);
      tensorByNode.set(node.id, tensorName);
      continue;
    }

    if (node.type === "Output") {
      if (incoming.length !== 1) {
        throw new Error("Output export currently requires exactly one incoming edge.");
      }
      const sourceTensor = tensorByNode.get(incoming[0]!);
      if (!sourceTensor) {
        throw new Error("Output export requires its source tensor to be available.");
      }
      lines.push(`        return self.lm_head(${sourceTensor})`);
      return lines;
    }

    if (incoming.length !== 1) {
      throw new Error(`${node.type} export currently requires exactly one incoming edge.`);
    }
    const sourceTensor = tensorByNode.get(incoming[0]!);
    if (!sourceTensor) {
      throw new Error(`${node.type} export requires its source tensor to be available.`);
    }
    lines.push(`        ${tensorName} = self.${name}(${sourceTensor})`);
    tensorByNode.set(node.id, tensorName);
  }

  const lastNode = ctx.orderedNodes[ctx.orderedNodes.length - 1];
  const lastTensor = lastNode ? tensorByNode.get(lastNode.id) : null;
  if (!lastTensor) {
    throw new Error("Project export could not resolve a final tensor.");
  }
  lines.push(`        return self.lm_head(${lastTensor})`);
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
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model=d_model, n_heads=n_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x


class TopKMoE(nn.Module):
    def __init__(self, d_model: int, expert_hidden: int, num_experts: int, top_k: int, activation_name: str = "gelu") -> None:
        super().__init__()
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if top_k <= 0 or top_k > num_experts:
            raise ValueError("top_k must be between 1 and num_experts")
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([
            FeedForward(d_model=d_model, ffn_hidden=expert_hidden, activation_name=activation_name)
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_logits = self.gate(x)
        topk_logits, topk_indices = torch.topk(gate_logits, k=self.top_k, dim=-1)
        topk_weights = torch.softmax(topk_logits, dim=-1)

        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=2)
        gather_index = topk_indices.unsqueeze(-1).expand(*topk_indices.shape, x.size(-1))
        selected_outputs = torch.gather(expert_outputs, 2, gather_index)
        mixed = (selected_outputs * topk_weights.unsqueeze(-1)).sum(dim=2)
        return x + mixed


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
