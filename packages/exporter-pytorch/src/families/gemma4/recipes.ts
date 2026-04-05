import type { Gemma4ArchitectureSpec } from "@neural-playground/ir-schema";

function pythonBool(value: boolean): string {
  return value ? "True" : "False";
}

export function renderGemma4Model(spec: Gemma4ArchitectureSpec, primitiveSections: string): string {
  const layerTypes = spec.config.layerTypes.map((layerType) => `"${layerType}"`).join(", ");
  const ropeParameters = Object.entries(spec.config.ropeParameters as Record<string, { ropeType: string; ropeTheta: number }>)
    .map(
      ([key, value]) =>
        `        "${key}": {"rope_type": "${value.ropeType}", "rope_theta": ${value.ropeTheta}},`
    )
    .join("\n");

  return `from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Gemma4Config:
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
    initializer_range: float = 0.02
    use_cache: bool = True
    pad_token_id: int = 0
    eos_token_id: int = 1
    bos_token_id: int = 2
    sliding_window: int = ${spec.config.slidingWindow}
    layer_types: tuple[str, ...] = (${layerTypes})
    final_logit_softcapping: float | None = None
    use_bidirectional_attention: str | None = None
    hidden_size_per_layer_input: int = 0
    vocab_size_per_layer_input: int = ${spec.config.vocabSize}
    num_global_key_value_heads: int | None = ${spec.config.numGlobalKeyValueHeads === null ? "None" : spec.config.numGlobalKeyValueHeads}
    global_head_dim: int = ${spec.config.globalHeadDim}
    attention_k_eq_v: bool = ${pythonBool(spec.config.attentionKEqV)}
    num_kv_shared_layers: int = ${spec.config.numKvSharedLayers}
    enable_moe_block: bool = False
    use_double_wide_mlp: bool = False
    num_experts: int | None = None
    top_k_experts: int | None = None
    moe_intermediate_size: int | None = None
    rope_parameters: dict[str, dict[str, float]] | None = None

    def __post_init__(self) -> None:
        if self.rope_parameters is None:
            self.rope_parameters = {
${ropeParameters}
            }
        for layer_type in set(self.layer_types):
            self.rope_parameters.setdefault(layer_type, {"rope_type": "default", "rope_theta": self.rope_theta})


${primitiveSections}


def create_causal_mask(hidden_states: torch.Tensor) -> torch.Tensor:
    seq_len = hidden_states.shape[1]
    mask = torch.full((seq_len, seq_len), torch.finfo(hidden_states.dtype).min, device=hidden_states.device, dtype=hidden_states.dtype)
    mask = torch.triu(mask, diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)


def create_sliding_window_causal_mask(hidden_states: torch.Tensor, sliding_window: int) -> torch.Tensor:
    seq_len = hidden_states.shape[1]
    mask = torch.full((seq_len, seq_len), torch.finfo(hidden_states.dtype).min, device=hidden_states.device, dtype=hidden_states.dtype)
    for row in range(seq_len):
        start = max(0, row - sliding_window + 1)
        mask[row, start : row + 1] = 0
    return mask.unsqueeze(0).unsqueeze(0)


class Gemma4TextModel(nn.Module):
    def __init__(self, config: Gemma4Config) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.embed_tokens = Gemma4TextScaledWordEmbedding(
            config.vocab_size, config.hidden_size, self.padding_idx, embed_scale=config.hidden_size ** 0.5
        )
        self.layers = nn.ModuleList(
            [Gemma4TextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Gemma4TextRotaryEmbedding(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.max_position_embeddings:
            raise ValueError(f"Sequence length {seq_len} exceeds configured maximum {self.config.max_position_embeddings}")
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        hidden_states = self.embed_tokens(input_ids)
        if attention_mask is None:
            causal_mask_mapping = {}
            for layer_type in set(self.config.layer_types):
                if layer_type == "sliding_attention":
                    causal_mask_mapping[layer_type] = create_sliding_window_causal_mask(hidden_states, self.config.sliding_window)
                else:
                    causal_mask_mapping[layer_type] = create_causal_mask(hidden_states)
        else:
            causal_mask_mapping = {layer_type: attention_mask for layer_type in set(self.config.layer_types)}
        position_embeddings = {
            layer_type: self.rotary_emb(hidden_states, position_ids, layer_type)
            for layer_type in set(self.config.layer_types)
        }
        for i, decoder_layer in enumerate(self.layers):
            layer_type = self.config.layer_types[i]
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings[layer_type],
                attention_mask=causal_mask_mapping[layer_type],
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states


class Gemma4ForCausalLM(nn.Module):
    def __init__(self, config: Gemma4Config | None = None) -> None:
        super().__init__()
        self.config = config or Gemma4Config()
        self.model = Gemma4TextModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
        logits = self.lm_head(hidden_states)
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping
        return logits
`;
}
