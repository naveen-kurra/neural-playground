import { type BlockNode, type ModelGraph } from "@neural-playground/block-schema";

export type ExportTarget = "pytorch";
export type ProjectFileMap = Record<string, string>;

type ExportContext = {
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

function findSingleInput(graph: ModelGraph): BlockNode {
  const inputs = graph.nodes.filter((node) => node.type === "Input");
  if (inputs.length !== 1) {
    throw new Error("Project export currently requires exactly one Input node.");
  }
  return inputs[0]!;
}

function topologicalSort(graph: ModelGraph): BlockNode[] {
  const indegree = new Map<string, number>();
  const outgoing = new Map<string, string[]>();

  for (const node of graph.nodes) {
    indegree.set(node.id, 0);
    outgoing.set(node.id, []);
  }

  for (const edge of graph.edges) {
    indegree.set(edge.target, (indegree.get(edge.target) ?? 0) + 1);
    outgoing.get(edge.source)?.push(edge.target);
  }

  const ready: string[] = graph.nodes.filter((node) => (indegree.get(node.id) ?? 0) === 0).map((node) => node.id);
  const orderedIds: string[] = [];

  while (ready.length > 0) {
    const nextId = ready.shift()!;
    orderedIds.push(nextId);

    for (const targetId of outgoing.get(nextId) ?? []) {
      const nextDegree = (indegree.get(targetId) ?? 0) - 1;
      indegree.set(targetId, nextDegree);
      if (nextDegree === 0) {
        ready.push(targetId);
      }
    }
  }

  if (orderedIds.length !== graph.nodes.length) {
    throw new Error("Project export requires an acyclic graph.");
  }

  return orderedIds
    .map((id) => graph.nodes.find((node) => node.id === id))
    .filter((node): node is BlockNode => node !== undefined);
}

function sanitizeNodeId(nodeId: string): string {
  return nodeId.replace(/[^a-zA-Z0-9_]/g, "_").toLowerCase();
}

function normalizeActivation(value: unknown): string {
  return String(value ?? "gelu").toLowerCase();
}

function normalizeOptimizer(value: unknown): string {
  const normalized = String(value ?? "AdamW").toLowerCase();
  return normalized === "sgd" ? "sgd" : "adamw";
}

function normalizeLoss(value: unknown): string {
  const normalized = String(value ?? "CrossEntropy").toLowerCase();
  return normalized === "crossentropy" ? "cross_entropy" : normalized;
}

function buildContext(graph: ModelGraph): ExportContext {
  const orderedNodes = topologicalSort(graph);
  const inputNode = findSingleInput(graph);
  const embeddingNode = orderedNodes.find((node) => node.type === "Embedding");
  if (!embeddingNode) {
    throw new Error("Project export currently requires an Embedding node.");
  }

  const warnings: string[] = [];
  if (graph.nodes.some((node) => graph.edges.filter((edge) => edge.target === node.id).length > 1)) {
    warnings.push("Branch merges are flattened in export order. Full multi-input graph execution is not implemented yet.");
  }

  const firstTransformer = orderedNodes.find((node) => node.type === "TransformerBlock");
  const firstMlp = orderedNodes.find((node) => node.type === "MLP");

  let currentDim = Number(embeddingNode.config.embeddingDim ?? 768);
  for (const node of orderedNodes) {
    if (node.type === "TransformerBlock") {
      currentDim = Number(node.config.dModel ?? currentDim);
    }
  }

  return {
    orderedNodes,
    warnings,
    vocabSize: Number(embeddingNode.config.vocabSize ?? 32000),
    sequenceLength: Number(inputNode.config.sequenceLength ?? 1024),
    embeddingDim: Number(embeddingNode.config.embeddingDim ?? 768),
    transformerCount: orderedNodes.filter((node) => node.type === "TransformerBlock").length,
    defaultHeads: Number(firstTransformer?.config.numHeads ?? 12),
    defaultFfnHidden: Number(firstTransformer?.config.ffnHidden ?? firstMlp?.config.hiddenDim ?? currentDim * 4),
    defaultActivation: normalizeActivation(firstMlp?.config.activation ?? graph.training.activation ?? "gelu"),
    optimizerName: normalizeOptimizer(graph.training.optimizer),
    lossName: normalizeLoss(graph.training.loss),
    lastSequenceDim: currentDim
  };
}

function exportActivationName(graph: ModelGraph, fallback: string): string {
  if (graph.training.activation === "Custom") {
    const customName = (graph.training.activationCustomName ?? "").trim();
    return customName || "custom_activation";
  }
  return normalizeActivation(graph.training.activation ?? fallback);
}

function exportLossName(graph: ModelGraph, fallback: string): string {
  if (graph.training.loss === "Custom") {
    const customName = (graph.training.lossCustomName ?? "").trim();
    return customName || "custom_loss";
  }
  return normalizeLoss(graph.training.loss ?? fallback);
}

function exportOptimizerName(graph: ModelGraph, fallback: string): string {
  if (graph.training.optimizer === "Custom") {
    const customName = (graph.training.optimizerCustomName ?? "").trim();
    return customName || "custom_optimizer";
  }
  return normalizeOptimizer(graph.training.optimizer ?? fallback);
}

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
      continue;
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

function exportModelYaml(graph: ModelGraph): string {
  const ctx = buildContext(graph);
  const activationName = exportActivationName(graph, ctx.defaultActivation);
  return `n_layers: ${ctx.transformerCount}
d_model: ${ctx.embeddingDim}
n_heads: ${ctx.defaultHeads}
ffn_hidden: ${ctx.defaultFfnHidden}
seq_len: ${ctx.sequenceLength}
vocab_size: ${ctx.vocabSize}
activation_name: ${activationName}
`;
}

function exportTrainYaml(graph: ModelGraph): string {
  const ctx = buildContext(graph);
  const optimizerName = exportOptimizerName(graph, ctx.optimizerName);
  const lossName = exportLossName(graph, ctx.lossName);
  return `seed: 42
learning_rate: ${graph.training.learningRate}
weight_decay: 0.1
warmup_steps: 100
max_steps: 100000
grad_clip: 1.0
grad_accum_steps: 4
log_every_steps: 10
eval_every_steps: 100
save_every_steps: 500
optimizer: ${optimizerName}
loss_name: ${lossName}
`;
}

function exportConfigPy(): string {
  return `from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ModelConfig:
    n_layers: int
    d_model: int
    n_heads: int
    ffn_hidden: int
    seq_len: int
    vocab_size: int
    activation_name: str = "gelu"


@dataclass(frozen=True)
class TrainConfig:
    seed: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    max_steps: int
    grad_clip: float
    grad_accum_steps: int
    log_every_steps: int
    eval_every_steps: int
    save_every_steps: int
    optimizer: str = "adamw"
    loss_name: str = "cross_entropy"


@dataclass(frozen=True)
class AppConfig:
    model: ModelConfig
    train: TrainConfig


def _load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping config at {path}")
    return data


def load_configs(model_path: str | Path, train_path: str | Path) -> AppConfig:
    model_raw = _load_yaml(model_path)
    train_raw = _load_yaml(train_path)
    model_cfg = ModelConfig(**model_raw)
    train_cfg = TrainConfig(**train_raw)
    return AppConfig(model=model_cfg, train=train_cfg)
`;
}

function exportTrainModulePy(graph: ModelGraph): string {
  const ctx = buildContext(graph);
  const lossName = exportLossName(graph, ctx.lossName);
  return `from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Any, Callable

import torch
from torch import nn
from torch.nn import functional as F


def is_finite_loss(value: float | torch.Tensor) -> bool:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return False
        return bool(torch.isfinite(value).item())
    return math.isfinite(float(value))


def cross_entropy_next_token(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


def get_loss_function(name: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    key = name.lower().strip()
    registry: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
        "cross_entropy": cross_entropy_next_token,
    }
    if key == "${lossName}" and key not in registry:
        raise ValueError("Implement custom loss '${lossName}' in src/kurra_ai_cb/train.py")
    if key not in registry:
        raise ValueError(f"Unsupported loss function: {name}")
    return registry[key]


def _unpack_batch(batch: Any) -> torch.Tensor:
    if isinstance(batch, dict):
        if "input_ids" not in batch:
            raise KeyError("Batch dict must include 'input_ids'")
        return batch["input_ids"]
    if isinstance(batch, torch.Tensor):
        return batch
    raise TypeError("Batch must be torch.Tensor or dict with input_ids")


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Any,
    *,
    grad_clip: float = 1.0,
    use_amp: bool = False,
    scaler: torch.amp.GradScaler | None = None,
    amp_dtype: torch.dtype = torch.float16,
    grad_accum_steps: int = 1,
    accumulation_step: int = 0,
    loss_name: str = "cross_entropy",
    return_metrics: bool = False,
) -> Any:
    if grad_accum_steps <= 0:
        raise ValueError("grad_accum_steps must be >= 1")

    input_ids = _unpack_batch(batch)
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [B, T]")
    if input_ids.size(1) < 2:
        raise ValueError("Sequence length must be at least 2")

    model.train()
    autocast_ctx = nullcontext()
    if use_amp and input_ids.device.type in {"cuda", "mps"}:
        autocast_ctx = torch.amp.autocast(device_type=input_ids.device.type, dtype=amp_dtype)

    loss_fn = get_loss_function(loss_name)

    with autocast_ctx:
        logits = model(input_ids[:, :-1])
        targets = input_ids[:, 1:]
        loss = loss_fn(logits, targets)

    if not is_finite_loss(loss):
        optimizer.zero_grad(set_to_none=True)
        if return_metrics:
            return {"loss": float("nan"), "grad_norm": float("nan"), "status": "non_finite_loss"}
        return torch.tensor(float("nan"), device=input_ids.device)

    scaled_loss = loss / grad_accum_steps
    if scaler is not None and use_amp and input_ids.device.type == "cuda":
        scaler.scale(scaled_loss).backward()
    else:
        scaled_loss.backward()

    grad_norm_value = float("nan")
    should_step = (accumulation_step + 1) % grad_accum_steps == 0
    if should_step:
        if scaler is not None and use_amp and input_ids.device.type == "cuda":
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            grad_norm_value = float(grad_norm.item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            grad_norm_value = float(grad_norm.item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    if return_metrics:
        return {"loss": float(loss.detach().item()), "grad_norm": grad_norm_value}

    return loss.detach()
`;
}

function exportScriptTrainPy(graph: ModelGraph): string {
  const ctx = buildContext(graph);
  const optimizerName = exportOptimizerName(graph, ctx.optimizerName);
  return `#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import math
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from kurra_ai_cb.checkpoint import load_checkpoint, save_checkpoint
from kurra_ai_cb.config import load_configs
from kurra_ai_cb.data import PackedShardIterator
from kurra_ai_cb.eval import evaluate
from kurra_ai_cb.logging_utils import JsonlLogger
from kurra_ai_cb.model import DecoderLM
from kurra_ai_cb.schedule import lr_scale
from kurra_ai_cb.train import train_step


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train exported decoder-only LM")
    p.add_argument("--model-config", default="configs/model.yaml")
    p.add_argument("--train-config", default="configs/train.yaml")
    p.add_argument("--train-shards-glob", default="")
    p.add_argument("--val-shards-glob", default="")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--eval-max-batches", type=int, default=8)
    p.add_argument("--output-dir", default="artifacts/run")
    p.add_argument("--resume", default="", help="Checkpoint path or 'latest'")
    p.add_argument("--early-stop-patience", type=int, default=-1)
    p.add_argument("--use-amp", action="store_true")
    return p.parse_args()


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def _glob_shards(pattern: str) -> list[Path]:
    if not pattern:
        return []
    return [Path(p) for p in sorted(glob.glob(pattern))]


def _to_torch_batches(iterator: Iterable[np.ndarray], device: torch.device, max_batches: int | None = None) -> Iterable[torch.Tensor]:
    count = 0
    for arr in iterator:
        yield torch.tensor(arr, dtype=torch.long, device=device)
        count += 1
        if max_batches is not None and count >= max_batches:
            break


def build_model(cfg, seq_len_override: int | None = None) -> DecoderLM:
    seq_len = seq_len_override or cfg.model.seq_len
    return DecoderLM(
        vocab_size=cfg.model.vocab_size,
        n_layers=cfg.model.n_layers,
        d_model=cfg.model.d_model,
        n_heads=cfg.model.n_heads,
        ffn_hidden=cfg.model.ffn_hidden,
        seq_len=seq_len,
        activation_name=cfg.model.activation_name,
    )


def build_optimizer(model: torch.nn.Module, cfg) -> torch.optim.Optimizer:
    name = cfg.train.optimizer.lower().strip()
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg.train.learning_rate)
    if name == "${optimizerName}" and name not in {"adamw", "sgd"}:
        raise ValueError("Implement custom optimizer '${optimizerName}' in scripts/train.py")
    return torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay)


def _save_checkpoint_bundle(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    runtime: dict,
    scaler: torch.amp.GradScaler | None = None,
) -> None:
    save_checkpoint(path, model, optimizer, step=step, scaler=scaler, extra={"runtime": runtime})


def main() -> None:
    args = parse_args()
    cfg = load_configs(args.model_config, args.train_config)

    model_seq_len = args.seq_len or cfg.model.seq_len
    max_steps = args.max_steps or cfg.train.max_steps
    device = _resolve_device()
    model = build_model(cfg, seq_len_override=model_seq_len).to(device)
    optimizer = build_optimizer(model, cfg)

    scaler = None
    if args.use_amp and device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")

    train_shards = _glob_shards(args.train_shards_glob)
    val_shards = _glob_shards(args.val_shards_glob)
    if not train_shards:
        print("Train entrypoint ready. Provide --train-shards-glob and optional --val-shards-glob.")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt = output_dir / "latest.pt"
    best_ckpt = output_dir / "best.pt"
    logger = JsonlLogger(output_dir / "train_log.jsonl")

    step = 0
    epoch = 0
    tokens_seen = 0
    best_val_loss = math.inf
    no_improve_evals = 0

    if args.resume:
        ckpt_path = latest_ckpt if args.resume == "latest" else Path(args.resume)
        state = load_checkpoint(ckpt_path, model, optimizer, scaler=scaler, map_location=device)
        runtime = state.get("extra", {}).get("runtime", {})
        step = int(runtime.get("step", state.get("step", 0)))
        epoch = int(runtime.get("epoch", 0))
        tokens_seen = int(runtime.get("tokens_seen", 0))
        best_val_loss = float(runtime.get("best_val_loss", math.inf))
        no_improve_evals = int(runtime.get("no_improve_evals", 0))

    accum_idx = 0
    early_stop = False

    while step < max_steps:
        epoch += 1
        train_it = PackedShardIterator(train_shards, batch_size=args.batch_size, shuffle=True, seed=cfg.train.seed + epoch)

        for np_batch in train_it:
            batch = torch.tensor(np_batch, dtype=torch.long, device=device)
            lr = cfg.train.learning_rate * lr_scale(step=step, warmup_steps=cfg.train.warmup_steps, max_steps=max_steps)
            _set_optimizer_lr(optimizer, lr)

            t0 = time.time()
            out = train_step(
                model,
                optimizer,
                batch,
                grad_clip=cfg.train.grad_clip,
                use_amp=args.use_amp,
                scaler=scaler,
                grad_accum_steps=cfg.train.grad_accum_steps,
                accumulation_step=accum_idx,
                loss_name=cfg.train.loss_name,
                return_metrics=True,
            )
            dt = max(1e-9, time.time() - t0)
            accum_idx += 1
            tokens_seen += int(batch.numel())
            should_step = accum_idx % cfg.train.grad_accum_steps == 0
            if not should_step:
                continue

            step += 1
            tokens_per_sec = float(batch.numel()) / dt
            if step % cfg.train.log_every_steps == 0 or step == 1:
                logger.log(
                    {
                        "event": "train",
                        "step": step,
                        "epoch": epoch,
                        "loss": out["loss"],
                        "lr": lr,
                        "grad_norm": out["grad_norm"],
                        "tokens_seen": tokens_seen,
                        "tokens_per_sec": tokens_per_sec,
                    }
                )

            if val_shards and (step % cfg.train.eval_every_steps == 0):
                val_it = PackedShardIterator(val_shards, batch_size=args.batch_size, shuffle=False, seed=cfg.train.seed)
                val_batches = list(_to_torch_batches(val_it, device=device, max_batches=args.eval_max_batches))
                metrics = evaluate(model, val_batches)
                val_loss = float(metrics["loss"])
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improve_evals = 0
                    runtime = {
                        "step": step,
                        "epoch": epoch,
                        "tokens_seen": tokens_seen,
                        "best_val_loss": best_val_loss,
                        "no_improve_evals": no_improve_evals,
                    }
                    _save_checkpoint_bundle(best_ckpt, model, optimizer, step=step, runtime=runtime, scaler=scaler)
                else:
                    no_improve_evals += 1
                    if args.early_stop_patience >= 0 and no_improve_evals > args.early_stop_patience:
                        early_stop = True

            if step % cfg.train.save_every_steps == 0:
                runtime = {
                    "step": step,
                    "epoch": epoch,
                    "tokens_seen": tokens_seen,
                    "best_val_loss": best_val_loss,
                    "no_improve_evals": no_improve_evals,
                }
                _save_checkpoint_bundle(latest_ckpt, model, optimizer, step=step, runtime=runtime, scaler=scaler)

            if early_stop or step >= max_steps:
                break

        if early_stop:
            break


if __name__ == "__main__":
    main()
`;
}

function exportRequirementsTxt(): string {
  return `numpy
pyyaml
sentencepiece
torch
`;
}

function exportReadme(graph: ModelGraph): string {
  const ctx = buildContext(graph);
  const activationName = exportActivationName(graph, ctx.defaultActivation);
  const optimizerName = exportOptimizerName(graph, ctx.optimizerName);
  const lossName = exportLossName(graph, ctx.lossName);
  return `# Exported Neural Playground Project

This project was generated from Neural Playground.

## Model Summary

- vocab size: ${ctx.vocabSize}
- sequence length: ${ctx.sequenceLength}
- transformer blocks: ${ctx.transformerCount}
- default model dim: ${ctx.embeddingDim}
- optimizer: ${optimizerName}
- loss: ${lossName}
- activation: ${activationName}

## Train

\`\`\`bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python scripts/train.py \\
  --model-config configs/model.yaml \\
  --train-config configs/train.yaml \\
  --train-shards-glob "data/packed/train/*.npy" \\
  --val-shards-glob "data/packed/val/*.npy"
\`\`\`
`;
}

function exportCheckpointPy(): string {
  return `from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    scheduler: Any | None = None,
    scaler: torch.amp.GradScaler | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    ckpt_path = Path(path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = ckpt_path.with_suffix(ckpt_path.suffix + ".tmp")
    state: dict[str, Any] = {
        "step": int(step),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "extra": extra or {},
    }
    torch.save(state, tmp_path)
    tmp_path.replace(ckpt_path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    scheduler: Any | None = None,
    scaler: torch.amp.GradScaler | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    ckpt_path = Path(path)
    state: dict[str, Any] = torch.load(ckpt_path, map_location=map_location)
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])
    return state
`;
}

function exportCustomHookNotesPy(graph: ModelGraph): string {
  const ctx = buildContext(graph);
  const activationName = exportActivationName(graph, ctx.defaultActivation);
  const optimizerName = exportOptimizerName(graph, ctx.optimizerName);
  const lossName = exportLossName(graph, ctx.lossName);
  const lines = [
    "# Generated custom hook notes",
    "# Implement any referenced custom names in the exported project files."
  ];

  if (graph.training.activation === "Custom") {
    lines.push(`# Activation hook: implement '${activationName}' in src/kurra_ai_cb/model.py:get_activation`);
  }
  if (graph.training.loss === "Custom") {
    lines.push(`# Loss hook: implement '${lossName}' in src/kurra_ai_cb/train.py:get_loss_function`);
  }
  if (graph.training.optimizer === "Custom") {
    lines.push(`# Optimizer hook: implement '${optimizerName}' in scripts/train.py:build_optimizer`);
  }

  return lines.join("\n") + "\n";
}

function exportDataPy(): string {
  return `from __future__ import annotations

from pathlib import Path
from typing import Iterator, Sequence

import numpy as np


class PackedShardIterator:
    def __init__(
        self,
        shards: Sequence[str | Path],
        batch_size: int,
        *,
        shuffle: bool = True,
        seed: int = 42,
        mmap_mode: str | None = "r",
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not shards:
            raise ValueError("shards must not be empty")
        self.shards = [Path(s) for s in shards]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.mmap_mode = mmap_mode

    def __iter__(self) -> Iterator[np.ndarray]:
        rng = np.random.default_rng(self.seed)
        shard_paths = list(self.shards)
        if self.shuffle:
            rng.shuffle(shard_paths)

        for shard_path in shard_paths:
            arr = np.load(shard_path, allow_pickle=False, mmap_mode=self.mmap_mode)
            indices = np.arange(arr.shape[0])
            if self.shuffle:
                rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                take = indices[start : start + self.batch_size]
                if len(take) < self.batch_size:
                    continue
                yield np.asarray(arr[take])
`;
}

function exportEvalPy(): string {
  return `from __future__ import annotations

import math
from typing import Iterable

import torch

from kurra_ai_cb.train import cross_entropy_next_token


def perplexity_from_loss(loss: float) -> float:
    return float(math.exp(loss))


def evaluate(model: torch.nn.Module, batches: Iterable[torch.Tensor]) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch in batches:
            logits = model(batch[:, :-1])
            targets = batch[:, 1:]
            loss = cross_entropy_next_token(logits, targets)
            losses.append(float(loss.item()))
    if not losses:
        return {"loss": float("nan"), "perplexity": float("nan")}
    avg_loss = sum(losses) / len(losses)
    return {"loss": avg_loss, "perplexity": perplexity_from_loss(avg_loss)}
`;
}

function exportLoggingUtilsPy(): string {
  return `from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JsonlLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=True))
            f.write("\\n")
            f.flush()
`;
}

function exportSchedulePy(): string {
  return `from __future__ import annotations

import math


def lr_scale(step: int, warmup_steps: int, max_steps: int) -> float:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be >= 0")
    step = max(0, min(step, max_steps))
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    if max_steps == warmup_steps:
        return 1.0
    progress = (step - warmup_steps) / float(max_steps - warmup_steps)
    progress = max(0.0, min(progress, 1.0))
    return 0.5 * (1.0 + math.cos(math.pi * progress))
`;
}

export function exportProjectFiles(graph: ModelGraph): ProjectFileMap {
  return {
    "README.md": exportReadme(graph),
    "requirements.txt": exportRequirementsTxt(),
    "configs/model.yaml": exportModelYaml(graph),
    "configs/train.yaml": exportTrainYaml(graph),
    "scripts/train.py": exportScriptTrainPy(graph),
    "src/kurra_ai_cb/__init__.py": "",
    "src/kurra_ai_cb/model.py": exportModelGraphToPyTorch(graph),
    "src/kurra_ai_cb/config.py": exportConfigPy(),
    "src/kurra_ai_cb/checkpoint.py": exportCheckpointPy(),
    "src/kurra_ai_cb/data.py": exportDataPy(),
    "src/kurra_ai_cb/eval.py": exportEvalPy(),
    "src/kurra_ai_cb/logging_utils.py": exportLoggingUtilsPy(),
    "src/kurra_ai_cb/schedule.py": exportSchedulePy(),
    "src/kurra_ai_cb/train.py": exportTrainModulePy(graph),
    "CUSTOM_HOOKS.md": exportCustomHookNotesPy(graph)
  };
}

export function exporterStatus(): string {
  return "PyTorch project exporter ready.";
}
