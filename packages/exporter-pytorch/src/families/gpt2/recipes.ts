import type { GPT2ArchitectureSpec, GPT2BlockOp } from "@neural-playground/ir-schema";

function pythonBool(value: boolean): string {
  return value ? "True" : "False";
}

export function renderGpt2Model(spec: GPT2ArchitectureSpec, primitiveSections: string): string {
  const blockLines = spec.operators
    .filter((op): op is GPT2BlockOp => op.kind === "gpt2_block")
    .map(
      (_op: GPT2BlockOp, index: number) => `            GPT2Block(
                config,
                layer_idx=${index},
            )`
    )
    .join(",\n");

  return `from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class GPT2Config:
    vocab_size: int = ${spec.config.vocabSize}
    n_positions: int = ${spec.config.maxPositionEmbeddings}
    n_embd: int = ${spec.config.hiddenSize}
    n_layer: int = ${spec.config.numHiddenLayers}
    n_head: int = ${spec.config.numAttentionHeads}
    n_inner: int = ${spec.config.intermediateSize}
    activation_function: str = "${spec.config.activationFunction}"
    embd_pdrop: float = ${spec.config.embdDropout}
    attn_pdrop: float = ${spec.config.attnDropout}
    resid_pdrop: float = ${spec.config.residDropout}
    layer_norm_epsilon: float = ${spec.config.layerNormEpsilon}
    scale_attn_weights: bool = ${pythonBool(spec.config.scaleAttnWeights)}
    scale_attn_by_inverse_layer_idx: bool = ${pythonBool(spec.config.scaleAttnByInverseLayerIdx)}
    reorder_and_upcast_attn: bool = ${pythonBool(spec.config.reorderAndUpcastAttn)}
    tie_word_embeddings: bool = ${pythonBool(spec.config.tieWordEmbeddings)}


${primitiveSections}


class GPT2Model(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([
${blockLines}
        ])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.config = config

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        if seq_len > self.config.n_positions:
            raise ValueError(f"Sequence length {seq_len} exceeds configured maximum {self.config.n_positions}")
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, seq_len)

        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)
        for block in self.h:
            hidden_states = block(hidden_states)
        hidden_states = self.ln_f(hidden_states)
        return hidden_states


class GPT2LMHeadModel(nn.Module):
    def __init__(self, config: GPT2Config | None = None) -> None:
        super().__init__()
        self.config = config or GPT2Config()
        self.transformer = GPT2Model(self.config)
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.transformer.wte.weight

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        hidden_states = self.transformer(input_ids=input_ids, position_ids=position_ids)
        return self.lm_head(hidden_states)
`;
}

