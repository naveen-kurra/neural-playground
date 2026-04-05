export function renderGemma4Primitives(): string {
  return [
    `def get_activation(name: str) -> nn.Module:
    key = name.lower().strip()
    if key == "gelu_pytorch_tanh":
        return nn.GELU(approximate="tanh")
    if key == "gelu":
        return nn.GELU()
    if key in {"silu", "swish"}:
        return nn.SiLU()
    if key == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported activation: {name}")
`,
    `class Gemma4RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if self.with_scale:
            self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        return hidden_states * torch.pow(mean_squared, -0.5)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normed_output = self._norm(hidden_states.float())
        if self.with_scale:
            normed_output = normed_output * self.weight.float()
        return normed_output.type_as(hidden_states)
`,
    `class Gemma4TextScaledWordEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int = 0, embed_scale: float = 1.0) -> None:
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.register_buffer("embed_scale", torch.tensor(embed_scale), persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)
`,
    `class Gemma4TextRotaryEmbedding(nn.Module):
    def __init__(self, config: Gemma4Config) -> None:
        super().__init__()
        self.config = config
        self.layer_types = set(config.layer_types)
        for layer_type in self.layer_types:
            base = config.rope_parameters[layer_type]["rope_theta"]
            dim = config.global_head_dim if layer_type == "full_attention" and config.global_head_dim else config.head_dim
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
            self.register_buffer(f"{layer_type}_inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor, layer_type: str) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = getattr(self, f"{layer_type}_inv_freq").to(device=x.device)
        position_ids = position_ids.to(device=x.device, dtype=torch.float32)
        freqs = torch.einsum("bs,d->bsd", position_ids, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)
`,
    `def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
`,
    `def apply_gemma4_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 2) -> torch.Tensor:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)
`,
    `def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
`,
    `class Gemma4TextMLP(nn.Module):
    def __init__(self, config: Gemma4Config, layer_idx: int) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = get_activation(config.hidden_act)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))
`,
    `class Gemma4TextAttention(nn.Module):
    def __init__(self, config: Gemma4Config, layer_idx: int) -> None:
        super().__init__()
        self.layer_type = config.layer_types[layer_idx] if config.layer_types else "sliding_attention"
        self.layer_idx = layer_idx
        self.is_sliding = self.layer_type == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_sliding else None
        self.head_dim = config.global_head_dim if not self.is_sliding and config.global_head_dim else config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_global_key_value_heads if (config.attention_k_eq_v and not self.is_sliding) else config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        # Gemma 4 HF sets attention scaling to 1.0 and relies on q/k normalization.
        self.scaling = 1.0
        self.attention_dropout = config.attention_dropout
        self.q_norm = Gemma4RMSNorm(dim=self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma4RMSNorm(dim=self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = Gemma4RMSNorm(dim=self.head_dim, eps=config.rms_norm_eps, with_scale=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        cos, sin = position_embeddings

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        query_states = self.q_norm(query_states)
        query_states = apply_gemma4_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)
        query_states = query_states.transpose(1, 2)

        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape)
        key_states = self.k_norm(key_states)
        key_states = apply_gemma4_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2)
        key_states = key_states.transpose(1, 2)
        value_states = self.v_norm(value_states).transpose(1, 2)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        if attention_mask is None:
            q_len = hidden_states.shape[1]
            causal_mask = torch.tril(torch.ones((q_len, q_len), dtype=torch.bool, device=hidden_states.device))
            attn_weights = attn_weights.masked_fill(~causal_mask, torch.finfo(attn_weights.dtype).min)
        else:
            attn_weights = attn_weights + attention_mask[:, :, : attn_weights.shape[-2], : attn_weights.shape[-1]]
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
`,
    `class Gemma4TextDecoderLayer(nn.Module):
    def __init__(self, config: Gemma4Config, layer_idx: int) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = Gemma4TextAttention(config=config, layer_idx=layer_idx)
        self.mlp = Gemma4TextMLP(config, layer_idx)
        self.input_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.register_buffer("layer_scalar", torch.ones(1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = hidden_states * self.layer_scalar
        return hidden_states
`,
  ].join("\n");
}
