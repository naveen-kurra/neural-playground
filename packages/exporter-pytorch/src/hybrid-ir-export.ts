import type { HybridDecoderArchitectureSpec, HybridBlockOp } from "@neural-playground/ir-schema";

function pythonBool(value: boolean): string {
  return value ? "True" : "False";
}

function renderBlockInit(block: HybridBlockOp): string {
  if (block.family === "gpt2") {
    return `            GPT2Block(HybridGPT2BlockConfig(
                hidden_size=${block.hiddenSize},
                intermediate_size=${block.intermediateSize},
                num_attention_heads=${block.numHeads},
                layer_norm_epsilon=${block.layerNormEpsilon},
                activation_function="${block.activation}",
                feedforward_type="${block.feedforwardType}",
                num_experts=${block.numExperts},
                top_k=${block.topK},
                expert_hidden=${block.expertHidden},
                attn_pdrop=${block.attnDropout},
                resid_pdrop=${block.residDropout},
                scale_attn_weights=${pythonBool(block.scaleAttnWeights)},
                scale_attn_by_inverse_layer_idx=${pythonBool(block.scaleAttnByInverseLayerIdx)},
                reorder_and_upcast_attn=${pythonBool(block.reorderAndUpcastAttn)},
                layer_idx=${0},
            ))`;
  }

  return `            LlamaDecoderLayer(HybridLlamaBlockConfig(
                hidden_size=${block.hiddenSize},
                intermediate_size=${block.intermediateSize},
                num_attention_heads=${block.numHeads},
                num_key_value_heads=${block.numKeyValueHeads},
                head_dim=${block.headDim},
                rms_norm_eps=${block.rmsNormEpsilon},
                rope_theta=${block.ropeTheta},
                hidden_act="${block.activation}",
                attention_bias=${pythonBool(block.attentionBias)},
                attention_dropout=${block.attentionDropout},
                mlp_bias=${pythonBool(block.mlpBias)},
            ))`;
}

export function exportHybridIrToPyTorch(spec: HybridDecoderArchitectureSpec): string {
  const blockLines = spec.operators.blocks.map(renderBlockInit).join(",\n");
  const embeddingInit =
    spec.operators.embedding.family === "gpt2"
      ? `        self.embedding_type = "gpt2"
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.wpe = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.drop = nn.Dropout(config.embd_pdrop)`
      : `        self.embedding_type = "llama"
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)`;
  const embeddingForward =
    spec.operators.embedding.family === "gpt2"
      ? `        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        hidden_states = self.wte(input_ids) + self.wpe(positions)
        hidden_states = self.drop(hidden_states)`
      : `        hidden_states = self.embed_tokens(input_ids)`;
  const finalNormInit =
    spec.operators.finalNorm.family === "gpt2"
      ? `        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.final_norm_epsilon)`
      : `        self.final_norm = LlamaRMSNorm(config.hidden_size, eps=config.final_norm_epsilon)`;
  const maybeRotary = spec.operators.blocks.some((block) => block.family === "llama")
    ? `        self.rotary_emb = LlamaRotaryEmbedding(config.max_llama_head_dim, base=config.max_rope_theta)`
    : "";
  const blockForward = `        for block in self.blocks:
            if isinstance(block, GPT2Block):
                hidden_states = block(hidden_states)
            else:
                position_embeddings = self.rotary_emb(hidden_states, position_ids)
                hidden_states = block(hidden_states, position_ids=position_ids, position_embeddings=position_embeddings)`;

  return `from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class HybridGPT2BlockConfig:
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    layer_norm_epsilon: float
    activation_function: str
    attn_pdrop: float
    resid_pdrop: float
    feedforward_type: str = "mlp"
    num_experts: int = 8
    top_k: int = 2
    expert_hidden: int = 3072
    scale_attn_weights: bool = True
    scale_attn_by_inverse_layer_idx: bool = False
    reorder_and_upcast_attn: bool = False
    layer_idx: int = 0


@dataclass(frozen=True)
class HybridLlamaBlockConfig:
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    hidden_act: str
    attention_bias: bool = False
    attention_dropout: float = 0.0
    mlp_bias: bool = False


@dataclass(frozen=True)
class HybridConfig:
    vocab_size: int = ${spec.config.vocabSize}
    hidden_size: int = ${spec.config.hiddenSize}
    max_position_embeddings: int = ${spec.config.maxPositionEmbeddings}
    tie_word_embeddings: bool = ${pythonBool(spec.config.tieWordEmbeddings)}
    embd_pdrop: float = ${spec.operators.embedding.family === "gpt2" ? spec.operators.embedding.embdDropout : 0.0}
    final_norm_epsilon: float = ${spec.operators.finalNorm.epsilon}
    max_llama_head_dim: int = ${Math.max(1, ...spec.operators.blocks.filter((b) => b.family === "llama").map((b) => b.headDim ?? 1), 1)}
    max_rope_theta: float = ${Math.max(10000, ...spec.operators.blocks.filter((b) => b.family === "llama").map((b) => b.ropeTheta ?? 10000), 10000)}


class NewGELUActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


def get_gpt2_activation(name: str) -> nn.Module:
    key = name.lower().strip()
    registry: dict[str, nn.Module] = {
        "gelu": nn.GELU(),
        "relu": nn.ReLU(),
        "silu": nn.SiLU(),
        "gelu_new": NewGELUActivation(),
    }
    if key not in registry:
        raise ValueError(f"Unsupported GPT-2 activation: {name}")
    return registry[key]


def get_llama_activation(name: str) -> nn.Module:
    key = name.lower().strip()
    if key in {"silu", "swish"}:
        return nn.SiLU()
    if key == "relu":
        return nn.ReLU()
    if key == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported LLaMA activation: {name}")


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
    def __init__(self, config: HybridGPT2BlockConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale_attn_weights = config.scale_attn_weights
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = config.layer_idx
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
    def __init__(self, config: HybridGPT2BlockConfig) -> None:
        super().__init__()
        self.c_fc = Conv1D(config.intermediate_size, config.hidden_size)
        self.c_proj = Conv1D(config.hidden_size, config.intermediate_size)
        self.act = get_gpt2_activation(config.activation_function)
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        return self.dropout(hidden_states)


class GPT2MoE(nn.Module):
    def __init__(self, config: HybridGPT2BlockConfig) -> None:
        super().__init__()
        if config.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if config.top_k <= 0 or config.top_k > config.num_experts:
            raise ValueError("top_k must be between 1 and num_experts")
        self.top_k = config.top_k
        self.gate = Conv1D(config.num_experts, config.hidden_size)
        expert_hidden = config.expert_hidden or config.intermediate_size
        self.experts = nn.ModuleList(
            [
                GPT2MLP(
                    HybridGPT2BlockConfig(
                        hidden_size=config.hidden_size,
                        intermediate_size=expert_hidden,
                        num_attention_heads=config.num_attention_heads,
                        layer_norm_epsilon=config.layer_norm_epsilon,
                        activation_function=config.activation_function,
                        feedforward_type="mlp",
                        num_experts=config.num_experts,
                        top_k=config.top_k,
                        expert_hidden=expert_hidden,
                        attn_pdrop=config.attn_pdrop,
                        resid_pdrop=config.resid_pdrop,
                        scale_attn_weights=config.scale_attn_weights,
                        scale_attn_by_inverse_layer_idx=config.scale_attn_by_inverse_layer_idx,
                        reorder_and_upcast_attn=config.reorder_and_upcast_attn,
                        layer_idx=config.layer_idx,
                    )
                )
                for _ in range(config.num_experts)
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_logits = self.gate(hidden_states)
        topk_logits, topk_indices = torch.topk(gate_logits, k=self.top_k, dim=-1)
        topk_weights = torch.softmax(topk_logits, dim=-1)
        expert_outputs = torch.stack([expert(hidden_states) for expert in self.experts], dim=2)
        gather_index = topk_indices.unsqueeze(-1).expand(*topk_indices.shape, hidden_states.size(-1))
        selected_outputs = torch.gather(expert_outputs, 2, gather_index)
        return (selected_outputs * topk_weights.unsqueeze(-1)).sum(dim=2)


class GPT2Block(nn.Module):
    def __init__(self, config: HybridGPT2BlockConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        if config.feedforward_type == "moe":
            self.mlp = GPT2MoE(config)
        else:
            self.mlp = GPT2MLP(config)

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


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq.to(device=x.device)
        position_ids = position_ids.to(device=x.device, dtype=torch.float32)
        freqs = torch.einsum("bs,d->bsd", position_ids, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=x.dtype)
        sin = emb.sin().to(dtype=x.dtype)
        return cos.unsqueeze(1), sin.unsqueeze(1)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class LlamaAttention(nn.Module):
    def __init__(self, config: HybridLlamaBlockConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attention_dropout = config.attention_dropout
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

    def forward(self, hidden_states: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        causal_mask = torch.tril(torch.ones((q_len, q_len), dtype=torch.bool, device=hidden_states.device))
        attn_weights = attn_weights.masked_fill(~causal_mask, torch.finfo(attn_weights.dtype).min)
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.num_heads * self.head_dim)
        return self.o_proj(attn_output)


class LlamaMLP(nn.Module):
    def __init__(self, config: HybridLlamaBlockConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)
        self.act_fn = get_llama_activation(config.hidden_act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: HybridLlamaBlockConfig) -> None:
        super().__init__()
        self.self_attn = LlamaAttention(config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings=position_embeddings)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class HybridDecoderModel(nn.Module):
    def __init__(self, config: HybridConfig | None = None) -> None:
        super().__init__()
        self.config = config or HybridConfig()
${embeddingInit}
${maybeRotary}
        self.blocks = nn.ModuleList([
${blockLines}
        ])
${finalNormInit}

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.max_position_embeddings:
            raise ValueError(f"Sequence length {seq_len} exceeds configured maximum {self.config.max_position_embeddings}")
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
${embeddingForward}
${blockForward}
        hidden_states = self.final_norm(hidden_states)
        return hidden_states


class HybridForCausalLM(nn.Module):
    def __init__(self, config: HybridConfig | None = None) -> None:
        super().__init__()
        self.config = config or HybridConfig()
        self.model = HybridDecoderModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings:
            if hasattr(self.model, "wte"):
                self.lm_head.weight = self.model.wte.weight
            else:
                self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        hidden_states = self.model(input_ids=input_ids, position_ids=position_ids)
        return self.lm_head(hidden_states)
`;
}
