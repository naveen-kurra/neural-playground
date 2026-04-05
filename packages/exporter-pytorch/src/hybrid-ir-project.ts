import type { TrainingConfig } from "@neural-playground/block-schema";
import type { HybridDecoderArchitectureSpec } from "@neural-playground/ir-schema";
import {
  exportCheckpointPy,
  exportDataPy,
  exportEvalPy,
  exportLoggingUtilsPy,
  exportRequirementsTxt,
  exportSchedulePy,
  exportScriptTrainPyForOptimizer,
  exportTrainModulePyForLoss
} from "./runtime-templates";
import { exportHybridIrToPyTorch } from "./hybrid-ir-export";
import type { ProjectFileMap } from "./types";

function exportHybridModelYaml(spec: HybridDecoderArchitectureSpec): string {
  const blockYaml = spec.operators.blocks
    .map((block) =>
      block.family === "gpt2"
        ? `  - family: gpt2
    hidden_size: ${block.hiddenSize}
    intermediate_size: ${block.intermediateSize}
    num_attention_heads: ${block.numHeads}
    layer_norm_epsilon: ${block.layerNormEpsilon}
    activation_function: ${block.activation}
    attn_pdrop: ${block.attnDropout}
    resid_pdrop: ${block.residDropout}`
        : `  - family: ${block.family}
    hidden_size: ${block.hiddenSize}
    intermediate_size: ${block.intermediateSize}
    num_attention_heads: ${block.numHeads}
    num_key_value_heads: ${block.numKeyValueHeads}
    head_dim: ${block.headDim}
    rms_norm_eps: ${block.rmsNormEpsilon}
    rope_theta: ${block.ropeTheta}
    hidden_act: ${block.activation}
    attention_bias: ${String(block.attentionBias).toLowerCase()}
    attention_dropout: ${block.attentionDropout}
    mlp_bias: ${String(block.mlpBias).toLowerCase()}`
    )
    .join("\n");

  return `model_family: hybrid_decoder
vocab_size: ${spec.config.vocabSize}
hidden_size: ${spec.config.hiddenSize}
max_position_embeddings: ${spec.config.maxPositionEmbeddings}
tie_word_embeddings: ${String(spec.config.tieWordEmbeddings).toLowerCase()}
embedding_family: ${spec.config.embeddingFamily}
embd_pdrop: ${spec.operators.embedding.family === "gpt2" ? spec.operators.embedding.embdDropout : 0}
final_norm_family: ${spec.config.finalNormFamily}
final_norm_epsilon: ${spec.operators.finalNorm.epsilon}
blocks:
${blockYaml}
`;
}

function exportHybridTrainYaml(training: TrainingConfig): string {
  const optimizer = training.optimizer === "Custom" ? (training.optimizerCustomName || "custom_optimizer") : training.optimizer.toLowerCase();
  const loss = training.loss === "Custom" ? (training.lossCustomName || "custom_loss") : "cross_entropy";
  return `seed: 42
learning_rate: ${training.learningRate}
weight_decay: 0.1
warmup_steps: 100
max_steps: 100000
grad_clip: 1.0
grad_accum_steps: 4
log_every_steps: 10
eval_every_steps: 100
save_every_steps: 500
optimizer: ${optimizer}
loss_name: ${loss}
`;
}

function exportHybridConfigPy(): string {
  return `from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


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
    model: dict[str, Any]
    train: TrainConfig


def _load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping config at {path}")
    return data


def load_configs(model_path: str | Path, train_path: str | Path) -> AppConfig:
    return AppConfig(model=_load_yaml(model_path), train=TrainConfig(**_load_yaml(train_path)))
`;
}

function withHybridBuildModel(modelPy: string): string {
  return `${modelPy}


def build_model(cfg, seq_len_override: int | None = None) -> HybridForCausalLM:
    model_cfg = dict(cfg.model)
    if seq_len_override is not None:
        model_cfg["max_position_embeddings"] = seq_len_override
    allowed_keys = {
        "vocab_size",
        "hidden_size",
        "max_position_embeddings",
        "tie_word_embeddings",
        "embd_pdrop",
        "final_norm_epsilon",
        "max_llama_head_dim",
        "max_rope_theta",
    }
    return HybridForCausalLM(HybridConfig(**{k: v for k, v in model_cfg.items() if k in allowed_keys}))
`;
}

function exportHybridTrainModulePy(lossName: string): string {
  return `from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Any, Callable

import torch
from torch import nn
from torch.nn import functional as F


def is_finite_loss(value: float | torch.Tensor) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.numel() == 1 and torch.isfinite(value).item())
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
    input_ids = batch["input_ids"] if isinstance(batch, dict) else batch
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
        return {"loss": float("nan"), "grad_norm": float("nan"), "status": "non_finite_loss"} if return_metrics else loss.detach()

    scaled_loss = loss / grad_accum_steps
    if scaler is not None and use_amp and input_ids.device.type == "cuda":
        scaler.scale(scaled_loss).backward()
    else:
        scaled_loss.backward()

    grad_norm_value = float("nan")
    if (accumulation_step + 1) % grad_accum_steps == 0:
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

    return {"loss": float(loss.detach().item()), "grad_norm": grad_norm_value} if return_metrics else loss.detach()
`;
}

function exportHybridScriptTrainPy(training: TrainingConfig): string {
  const optimizer = training.optimizer === "Custom" ? (training.optimizerCustomName || "custom_optimizer") : training.optimizer.toLowerCase();
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
from kurra_ai_cb.model import HybridConfig, HybridForCausalLM
from kurra_ai_cb.schedule import lr_scale
from kurra_ai_cb.train import train_step


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train exported hybrid decoder LM")
    p.add_argument("--model-config", default="configs/model.yaml")
    p.add_argument("--train-config", default="configs/train.yaml")
    p.add_argument("--train-shards-glob", default="")
    p.add_argument("--val-shards-glob", default="")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--eval-max-batches", type=int, default=8)
    p.add_argument("--output-dir", default="artifacts/run")
    p.add_argument("--resume", default="", help="Checkpoint path or 'latest'")
    p.add_argument("--use-amp", action="store_true")
    return p.parse_args()


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(model_cfg: dict) -> HybridForCausalLM:
    return HybridForCausalLM(HybridConfig(**{k: v for k, v in model_cfg.items() if k != "blocks"}))


def build_optimizer(model: torch.nn.Module, cfg) -> torch.optim.Optimizer:
    name = cfg.train.optimizer.lower().strip()
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg.train.learning_rate)
    if name == "${optimizer}" and name not in {"adamw", "sgd"}:
        raise ValueError("Implement custom optimizer '${optimizer}' in scripts/train.py")
    return torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay)


def _glob_shards(pattern: str) -> list[Path]:
    return [Path(p) for p in sorted(glob.glob(pattern))] if pattern else []


def _to_torch_batches(iterator: Iterable[np.ndarray], device: torch.device, max_batches: int | None = None) -> Iterable[torch.Tensor]:
    count = 0
    for arr in iterator:
        yield torch.tensor(arr, dtype=torch.long, device=device)
        count += 1
        if max_batches is not None and count >= max_batches:
            break


def main() -> None:
    args = parse_args()
    cfg = load_configs(args.model_config, args.train_config)
    device = _resolve_device()
    max_steps = args.max_steps or cfg.train.max_steps
    model = build_model(cfg.model).to(device)
    optimizer = build_optimizer(model, cfg)
    scaler = torch.amp.GradScaler("cuda") if args.use_amp and device.type == "cuda" else None

    train_shards = _glob_shards(args.train_shards_glob)
    val_shards = _glob_shards(args.val_shards_glob)
    if not train_shards:
        print("Train entrypoint ready. Provide --train-shards-glob and optional --val-shards-glob.")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt = output_dir / "latest.pt"
    logger = JsonlLogger(output_dir / "train_log.jsonl")

    step = 0
    accum_idx = 0
    tokens_seen = 0
    best_val_loss = math.inf

    if args.resume:
        ckpt_path = latest_ckpt if args.resume == "latest" else Path(args.resume)
        load_checkpoint(ckpt_path, model, optimizer, scaler=scaler, map_location=device)

    while step < max_steps:
        train_it = PackedShardIterator(train_shards, batch_size=args.batch_size, shuffle=True, seed=cfg.train.seed + step)
        for np_batch in train_it:
            batch = torch.tensor(np_batch, dtype=torch.long, device=device)
            lr = cfg.train.learning_rate * lr_scale(step=step, warmup_steps=cfg.train.warmup_steps, max_steps=max_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr
            t0 = time.time()
            out = train_step(model, optimizer, batch, grad_clip=cfg.train.grad_clip, use_amp=args.use_amp, scaler=scaler, grad_accum_steps=cfg.train.grad_accum_steps, accumulation_step=accum_idx, loss_name=cfg.train.loss_name, return_metrics=True)
            dt = max(1e-9, time.time() - t0)
            accum_idx += 1
            tokens_seen += int(batch.numel())
            if accum_idx % cfg.train.grad_accum_steps != 0:
                continue
            step += 1
            if step % cfg.train.log_every_steps == 0 or step == 1:
                logger.log({"event": "train", "step": step, "loss": out["loss"], "lr": lr, "grad_norm": out["grad_norm"], "tokens_seen": tokens_seen, "tokens_per_sec": float(batch.numel()) / dt})
            if val_shards and step % cfg.train.eval_every_steps == 0:
                val_it = PackedShardIterator(val_shards, batch_size=args.batch_size, shuffle=False, seed=cfg.train.seed)
                metrics = evaluate(model, list(_to_torch_batches(val_it, device=device, max_batches=args.eval_max_batches)))
                best_val_loss = min(best_val_loss, float(metrics["loss"]))
            if step % cfg.train.save_every_steps == 0:
                save_checkpoint(latest_ckpt, model, optimizer, step=step, scaler=scaler, extra={"runtime": {"step": step, "tokens_seen": tokens_seen, "best_val_loss": best_val_loss}})
            if step >= max_steps:
                break


if __name__ == "__main__":
    main()
`;
}

export function exportHybridIrProjectFiles(spec: HybridDecoderArchitectureSpec, training: TrainingConfig): ProjectFileMap {
  const optimizerName = training.optimizer === "Custom" ? (training.optimizerCustomName || "custom_optimizer") : training.optimizer.toLowerCase();
  const lossName = training.loss === "Custom" ? (training.lossCustomName || "custom_loss") : "cross_entropy";
  return {
    "README.md": `# Exported Hybrid Decoder Project\n\nThis project was generated from Neural Playground using the true mixed-block exporter.\n`,
    "requirements.txt": exportRequirementsTxt(),
    "configs/model.yaml": exportHybridModelYaml(spec),
    "configs/train.yaml": exportHybridTrainYaml(training),
    "scripts/train.py": exportScriptTrainPyForOptimizer(optimizerName),
    "src/kurra_ai_cb/__init__.py": "",
    "src/kurra_ai_cb/model.py": withHybridBuildModel(exportHybridIrToPyTorch(spec)),
    "src/kurra_ai_cb/config.py": exportHybridConfigPy(),
    "src/kurra_ai_cb/checkpoint.py": exportCheckpointPy(),
    "src/kurra_ai_cb/data.py": exportDataPy(),
    "src/kurra_ai_cb/eval.py": exportEvalPy(),
    "src/kurra_ai_cb/logging_utils.py": exportLoggingUtilsPy(),
    "src/kurra_ai_cb/schedule.py": exportSchedulePy(),
    "src/kurra_ai_cb/train.py": exportTrainModulePyForLoss(lossName),
    "CUSTOM_HOOKS.md": "# Generated custom hook notes\n"
  };
}
