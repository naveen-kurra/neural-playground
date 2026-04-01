import { type ModelGraph } from "@neural-playground/block-schema";
import { buildContext, exportActivationName, exportLossName, exportOptimizerName } from "./context";

export function exportModelYaml(graph: ModelGraph): string {
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

export function exportTrainYaml(graph: ModelGraph): string {
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

export function exportConfigPy(): string {
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
