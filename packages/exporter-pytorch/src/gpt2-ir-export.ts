import type { GPT2ArchitectureSpec, GPT2BlockOp } from "@neural-playground/ir-schema";

function pythonBool(value: boolean): string {
  return value ? "True" : "False";
}

export function exportGPT2IrToPyTorch(spec: GPT2ArchitectureSpec): string {
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


def get_activation(name: str) -> nn.Module:
    key = name.lower().strip()
    registry: dict[str, nn.Module] = {
        "gelu": nn.GELU(),
        "relu": nn.ReLU(),
        "silu": nn.SiLU(),
        "gelu_new": NewGELUActivation(),
    }
    if key not in registry:
        raise ValueError(f"Unsupported activation: {name}")
    return registry[key]


class NewGELUActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class Conv1D(nn.Module):
    def __init__(self, nf: int, nx: int) -> None:
        super().__init__()
        self.nf = nf
        self.weight = nn.Parameter(torch.empty(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        return x.view(size_out)


class GPT2Attention(nn.Module):
    def __init__(self, config: GPT2Config, layer_idx: int | None = None) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.embed_dim = config.n_embd
        self.num_heads = config.n_head
        self.head_dim = self.embed_dim // self.num_heads
        self.scale_attn_weights = config.scale_attn_weights
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn
        self.layer_idx = layer_idx or 0
        self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        new_shape = tensor.size()[:-1] + (self.num_heads, self.head_dim)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)

    def _merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (self.embed_dim,)
        return tensor.view(new_shape)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        query, key, value = self.c_attn(hidden_states).split(self.embed_dim, dim=2)
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        if self.scale_attn_weights:
            attn_weights = attn_weights / math.sqrt(self.head_dim)
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        q_len, k_len = query.size(-2), key.size(-2)
        causal_mask = torch.tril(torch.ones((q_len, k_len), dtype=torch.bool, device=hidden_states.device))
        attn_weights = attn_weights.masked_fill(~causal_mask, torch.finfo(attn_weights.dtype).min)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value)
        attn_output = self._merge_heads(attn_output)
        attn_output = self.c_proj(attn_output)
        return self.resid_dropout(attn_output)


class GPT2MLP(nn.Module):
    def __init__(self, intermediate_size: int, config: GPT2Config) -> None:
        super().__init__()
        self.c_fc = Conv1D(intermediate_size, config.n_embd)
        self.c_proj = Conv1D(config.n_embd, intermediate_size)
        self.act = get_activation(config.activation_function)
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        return self.dropout(hidden_states)


class GPT2Block(nn.Module):
    def __init__(self, config: GPT2Config, layer_idx: int) -> None:
        super().__init__()
        hidden_size = config.n_embd
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size
        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(inner_dim, config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        hidden_states = self.attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


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
