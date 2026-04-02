import type { LlamaArchitectureSpec, LlamaBlockOp } from "@neural-playground/ir-schema";

function pythonBool(value: boolean): string {
  return value ? "True" : "False";
}

export function renderLlamaModel(spec: LlamaArchitectureSpec, primitiveSections: string): string {
  const blockLines = spec.operators
    .filter((op): op is LlamaBlockOp => op.kind === "llama_block")
    .map(() => `            LlamaDecoderLayer(config)`)
    .join(",\n");

  return `from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LlamaConfig:
    vocab_size: int = ${spec.config.vocabSize}
    hidden_size: int = ${spec.config.hiddenSize}
    intermediate_size: int = ${spec.config.intermediateSize}
    num_hidden_layers: int = ${spec.config.numHiddenLayers}
    num_attention_heads: int = ${spec.config.numAttentionHeads}
    num_key_value_heads: int = ${spec.config.numKeyValueHeads}
    max_position_embeddings: int = ${spec.config.maxPositionEmbeddings}
    rms_norm_eps: float = ${spec.config.rmsNormEpsilon}
    rope_theta: float = ${spec.config.ropeTheta}
    hidden_act: str = "${spec.config.hiddenActivation}"
    attention_bias: bool = ${pythonBool(spec.config.attentionBias)}
    attention_dropout: float = ${spec.config.attentionDropout}
    mlp_bias: bool = ${pythonBool(spec.config.mlpBias)}
    tie_word_embeddings: bool = ${pythonBool(spec.config.tieWordEmbeddings)}
    head_dim: int = ${spec.config.headDim}


${primitiveSections}


class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = LlamaRotaryEmbedding(config.head_dim, base=config.rope_theta)
        self.layers = nn.ModuleList([
${blockLines}
        ])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.max_position_embeddings:
            raise ValueError(
                f"Sequence length {seq_len} exceeds configured maximum {self.config.max_position_embeddings}"
            )
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)

        hidden_states = self.embed_tokens(input_ids)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states, position_ids=position_ids, position_embeddings=position_embeddings)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class LlamaForCausalLM(nn.Module):
    def __init__(self, config: LlamaConfig | None = None) -> None:
        super().__init__()
        self.config = config or LlamaConfig()
        self.model = LlamaModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        hidden_states = self.model(input_ids=input_ids, position_ids=position_ids)
        return self.lm_head(hidden_states)
`;
}

