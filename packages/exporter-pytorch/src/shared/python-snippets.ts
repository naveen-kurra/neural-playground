export function renderNewGeluActivation(): string {
  return `class NewGELUActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
`;
}

export function renderActivationRegistry(kind: "gpt2" | "llama" | "phi3", fnName = "get_activation"): string {
  if (kind === "gpt2") {
    return `def ${fnName}(name: str) -> nn.Module:
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
`;
  }

  return `def ${fnName}(name: str) -> nn.Module:
    key = name.lower().strip()
    if key in {"silu", "swish"}:
        return nn.SiLU()
    if key == "relu":
        return nn.ReLU()
    if key == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")
`;
}

export function renderConv1D(): string {
  return `class Conv1D(nn.Module):
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
`;
}

export function renderGpt2Attention(configType: string): string {
  return `class GPT2Attention(nn.Module):
    def __init__(self, config: ${configType}${configType === "GPT2Config" ? ", layer_idx: int | None = None" : ""}) -> None:
        super().__init__()
        if config.${configType === "GPT2Config" ? "n_embd" : "hidden_size"} % config.${configType === "GPT2Config" ? "n_head" : "num_attention_heads"} != 0:
            raise ValueError("${configType === "GPT2Config" ? "n_embd must be divisible by n_head" : "hidden_size must be divisible by num_attention_heads"}")
        self.embed_dim = config.${configType === "GPT2Config" ? "n_embd" : "hidden_size"}
        self.num_heads = config.${configType === "GPT2Config" ? "n_head" : "num_attention_heads"}
        self.head_dim = self.embed_dim // self.num_heads
        self.scale_attn_weights = config.scale_attn_weights
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = config.layer_idx${configType === "GPT2Config" ? " if layer_idx is None else layer_idx" : ""}
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
`;
}

export function renderGpt2MLP(configType: string, activationFnName: string): string {
  const inter = configType === "GPT2Config" ? "intermediate_size: int, config: GPT2Config" : "config: HybridGPT2BlockConfig";
  const cfc = configType === "GPT2Config" ? "intermediate_size" : "config.intermediate_size";
  const nembd = configType === "GPT2Config" ? "config.n_embd" : "config.hidden_size";
  const act = configType === "GPT2Config" ? "config.activation_function" : "config.activation_function";
  return `class GPT2MLP(nn.Module):
    def __init__(self, ${inter}) -> None:
        super().__init__()
        self.c_fc = Conv1D(${cfc}, ${nembd})
        self.c_proj = Conv1D(${nembd}, ${cfc})
        self.act = ${activationFnName}(${act})
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        return self.dropout(hidden_states)
`;
}

export function renderGpt2MoE(configType: string): string {
  return `class GPT2MoE(nn.Module):
    def __init__(self, config: ${configType}) -> None:
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
${configType === "HybridGPT2BlockConfig"
  ? `                    HybridGPT2BlockConfig(
                        hidden_size=config.hidden_size,
                        intermediate_size=expert_hidden,
                        num_attention_heads=config.num_attention_heads,
                        layer_norm_epsilon=config.layer_norm_epsilon,
                        activation_function=config.activation_function,
                        attn_pdrop=config.attn_pdrop,
                        resid_pdrop=config.resid_pdrop,
                        feedforward_type="mlp",
                        num_experts=config.num_experts,
                        top_k=config.top_k,
                        expert_hidden=expert_hidden,
                        scale_attn_weights=config.scale_attn_weights,
                        scale_attn_by_inverse_layer_idx=config.scale_attn_by_inverse_layer_idx,
                        reorder_and_upcast_attn=config.reorder_and_upcast_attn,
                        layer_idx=config.layer_idx,
                    )`
  : `                    expert_hidden,
                    config`}
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
`;
}

export function renderGpt2Block(configType: string, options: { hybrid: boolean }): string {
  const initArg = configType === "GPT2Config" ? "config: GPT2Config, layer_idx: int" : "config: HybridGPT2BlockConfig";
  const hidden = configType === "GPT2Config" ? "config.n_embd" : "config.hidden_size";
  const eps = "config.layer_norm_epsilon";
  const attnInit = configType === "GPT2Config" ? "GPT2Attention(config, layer_idx=layer_idx)" : "GPT2Attention(config)";
  const inner = configType === "GPT2Config" ? "config.n_inner if config.n_inner is not None else 4 * hidden_size" : "";
  const mlpInit = configType === "GPT2Config" ? "GPT2MLP(inner_dim, config)" : "GPT2MLP(config)";
  const hybridBranch = options.hybrid
    ? `        if config.feedforward_type == "moe":
            self.mlp = GPT2MoE(config)
        else:
            self.mlp = GPT2MLP(config)`
    : `        self.mlp = ${mlpInit}`;
  return `class GPT2Block(nn.Module):
    def __init__(self, ${initArg}) -> None:
        super().__init__()
        hidden_size = ${hidden}
${configType === "GPT2Config" ? `        inner_dim = ${inner}` : ""}
        self.ln_1 = nn.LayerNorm(hidden_size, eps=${eps})
        self.attn = ${attnInit}
        self.ln_2 = nn.LayerNorm(hidden_size, eps=${eps})
${hybridBranch}

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
`;
}

export function renderLlamaNormAndRotaryHelpers(): string {
  return `class LlamaRMSNorm(nn.Module):
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


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
`;
}

export function renderLlamaAttention(configType: string): string {
  return `class LlamaAttention(nn.Module):
    def __init__(self, config: ${configType}) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
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

    def forward(${configType === "LlamaConfig"
  ? `
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:`
  : `
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:`}
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
`;
}

export function renderLlamaMLP(configType: string, activationFnName: string): string {
  return `class LlamaMLP(nn.Module):
    def __init__(self, config: ${configType}) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)
        self.act_fn = ${activationFnName}(config.hidden_act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
`;
}

export function renderLlamaMoE(configType: string): string {
  return `class LlamaMoE(nn.Module):
    def __init__(self, config: ${configType}) -> None:
        super().__init__()
        if config.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if config.top_k <= 0 or config.top_k > config.num_experts:
            raise ValueError("top_k must be between 1 and num_experts")
        self.top_k = config.top_k
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        expert_hidden = config.expert_hidden or config.intermediate_size
        self.experts = nn.ModuleList(
            [
                LlamaMLP(
                    HybridLlamaBlockConfig(
                        hidden_size=config.hidden_size,
                        intermediate_size=expert_hidden,
                        num_attention_heads=config.num_attention_heads,
                        num_key_value_heads=config.num_key_value_heads,
                        head_dim=config.head_dim,
                        rms_norm_eps=config.rms_norm_eps,
                        rope_theta=config.rope_theta,
                        hidden_act=config.hidden_act,
                        feedforward_type="mlp",
                        num_experts=config.num_experts,
                        top_k=config.top_k,
                        expert_hidden=expert_hidden,
                        attention_bias=config.attention_bias,
                        attention_dropout=config.attention_dropout,
                        mlp_bias=config.mlp_bias,
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
`;
}

export function renderLlamaDecoderLayer(configType: string, options: { hybrid: boolean }): string {
  const attnCall = configType === "LlamaConfig"
    ? "self.self_attn(hidden_states, position_ids=position_ids, position_embeddings=position_embeddings)"
    : "self.self_attn(hidden_states, position_embeddings=position_embeddings)";
  const hybridBranch = options.hybrid
    ? `        if config.feedforward_type == "moe":
            self.mlp = LlamaMoE(config)
        else:
            self.mlp = LlamaMLP(config)`
    : `        self.mlp = LlamaMLP(config)`;
  return `class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: ${configType}) -> None:
        super().__init__()
        self.self_attn = LlamaAttention(config)
${hybridBranch}
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
${configType === "LlamaConfig" ? "        position_ids: torch.Tensor,\n" : ""}        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = ${attnCall}
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
`;
}
