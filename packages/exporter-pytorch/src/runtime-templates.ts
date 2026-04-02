import { type ModelGraph } from "@neural-playground/block-schema";
import { buildContext, exportActivationName, exportLossName, exportOptimizerName } from "./context";

export function exportTrainModulePyForLoss(lossName: string): string {
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

export function exportTrainModulePy(graph: ModelGraph): string {
  const ctx = buildContext(graph);
  return exportTrainModulePyForLoss(exportLossName(graph, ctx.lossName));
}

export function exportScriptTrainPyForOptimizer(optimizerName: string): string {
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
from kurra_ai_cb.model import build_model
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
    print("[train] loading configs...", flush=True)
    cfg = load_configs(args.model_config, args.train_config)

    print("[train] resolving device...", flush=True)
    max_steps = args.max_steps or cfg.train.max_steps
    device = _resolve_device()
    print(f"[train] device={device}", flush=True)
    print("[train] building model...", flush=True)
    model = build_model(cfg, seq_len_override=args.seq_len).to(device)
    print(f"[train] model parameters={sum(p.numel() for p in model.parameters()):,}", flush=True)
    print("[train] building optimizer...", flush=True)
    optimizer = build_optimizer(model, cfg)

    scaler = None
    if args.use_amp and device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")

    print("[train] discovering shards...", flush=True)
    train_shards = _glob_shards(args.train_shards_glob)
    val_shards = _glob_shards(args.val_shards_glob)
    print(f"[train] train_shards={len(train_shards)} val_shards={len(val_shards)}", flush=True)
    if not train_shards:
        print("Train entrypoint ready. Provide --train-shards-glob and optional --val-shards-glob.")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train] output_dir={output_dir}", flush=True)
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
        print(f"[train] resuming from {ckpt_path}", flush=True)
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
        print(f"[train] starting epoch={epoch} step={step}", flush=True)
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
                print(
                    f"[train] step={step} loss={out['loss']:.4f} lr={lr:.6g} grad_norm={out['grad_norm']:.4f} tokens_seen={tokens_seen}",
                    flush=True,
                )
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
                print(f"[train] evaluating at step={step}...", flush=True)
                val_it = PackedShardIterator(val_shards, batch_size=args.batch_size, shuffle=False, seed=cfg.train.seed)
                val_batches = list(_to_torch_batches(val_it, device=device, max_batches=args.eval_max_batches))
                metrics = evaluate(model, val_batches)
                val_loss = float(metrics["loss"])
                print(f"[train] val_loss={val_loss:.4f}", flush=True)
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
                    print(f"[train] saving best checkpoint -> {best_ckpt}", flush=True)
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
                print(f"[train] saving checkpoint -> {latest_ckpt}", flush=True)
                _save_checkpoint_bundle(latest_ckpt, model, optimizer, step=step, runtime=runtime, scaler=scaler)

            if early_stop or step >= max_steps:
                break

        if early_stop:
            break

    print(f"[train] finished at step={step}", flush=True)


if __name__ == "__main__":
    main()
`;
}

export function exportScriptTrainPy(graph: ModelGraph): string {
  const ctx = buildContext(graph);
  return exportScriptTrainPyForOptimizer(exportOptimizerName(graph, ctx.optimizerName));
}

export function exportRequirementsTxt(): string {
  return `numpy
pyyaml
sentencepiece
torch
`;
}

export function exportReadme(graph: ModelGraph): string {
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

export function exportCheckpointPy(): string {
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

export function exportCustomHookNotesPy(graph: ModelGraph): string {
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

export function exportDataPy(): string {
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

export function exportEvalPy(): string {
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

export function exportLoggingUtilsPy(): string {
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

export function exportSchedulePy(): string {
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
