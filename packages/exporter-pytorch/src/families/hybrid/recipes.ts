import type { HybridBlockOp, HybridDecoderArchitectureSpec } from "@neural-playground/ir-schema";

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
                attn_pdrop=${block.attnDropout},
                resid_pdrop=${block.residDropout},
                feedforward_type="${block.feedforwardType}",
                num_experts=${block.numExperts},
                top_k=${block.topK},
                expert_hidden=${block.expertHidden},
                scale_attn_weights=${pythonBool(block.scaleAttnWeights)},
                scale_attn_by_inverse_layer_idx=${pythonBool(block.scaleAttnByInverseLayerIdx)},
                reorder_and_upcast_attn=${pythonBool(block.reorderAndUpcastAttn)},
                layer_idx=${0},
            ))`;
  }

  return `            ${block.family === "phi3" ? "Phi3DecoderLayer" : "LlamaDecoderLayer"}(HybridLlamaBlockConfig(
                hidden_size=${block.hiddenSize},
                intermediate_size=${block.intermediateSize},
                num_attention_heads=${block.numHeads},
                num_key_value_heads=${block.numKeyValueHeads},
                head_dim=${block.headDim},
                rms_norm_eps=${block.rmsNormEpsilon},
                rope_theta=${block.ropeTheta},
                hidden_act="${block.activation}",
                feedforward_type="${block.feedforwardType}",
                num_experts=${block.numExperts},
                top_k=${block.topK},
                expert_hidden=${block.expertHidden},
                attention_bias=${pythonBool(block.attentionBias)},
                attention_dropout=${block.attentionDropout},
                mlp_bias=${pythonBool(block.mlpBias)},
            ))`;
}

function renderDataclasses(spec: HybridDecoderArchitectureSpec): string {
  const sections: string[] = [];

  if (spec.operators.blocks.some((block) => block.family === "gpt2")) {
    sections.push(`@dataclass(frozen=True)
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
`);
  }

  if (spec.operators.blocks.some((block) => block.family === "llama" || block.family === "phi3")) {
    sections.push(`@dataclass(frozen=True)
class HybridLlamaBlockConfig:
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    hidden_act: str
    feedforward_type: str = "mlp"
    num_experts: int = 8
    top_k: int = 2
    expert_hidden: int = 11008
    attention_bias: bool = False
    attention_dropout: float = 0.0
    mlp_bias: bool = False
`);
  }

  const maxLlamaHeadDim = Math.max(
    1,
    ...spec.operators.blocks.filter((b) => b.family === "llama" || b.family === "phi3").map((b) => b.headDim ?? 1),
    1
  );
  const maxRopeTheta = Math.max(
    10000,
    ...spec.operators.blocks
      .filter((b) => b.family === "llama" || b.family === "phi3")
      .map((b) => b.ropeTheta ?? 10000),
    10000
  );

  sections.push(`@dataclass(frozen=True)
class HybridConfig:
    vocab_size: int = ${spec.config.vocabSize}
    hidden_size: int = ${spec.config.hiddenSize}
    max_position_embeddings: int = ${spec.config.maxPositionEmbeddings}
    tie_word_embeddings: bool = ${pythonBool(spec.config.tieWordEmbeddings)}
    embd_pdrop: float = ${spec.operators.embedding.family === "gpt2" ? spec.operators.embedding.embdDropout : 0.0}
    final_norm_epsilon: float = ${spec.operators.finalNorm.epsilon}
    max_llama_head_dim: int = ${maxLlamaHeadDim}
    max_rope_theta: float = ${maxRopeTheta}
`);

  return sections.join("\n");
}

function renderEmbeddingInit(spec: HybridDecoderArchitectureSpec): string {
  return spec.operators.embedding.family === "gpt2"
    ? `        self.embedding_type = "gpt2"
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.wpe = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.drop = nn.Dropout(config.embd_pdrop)`
    : `        self.embedding_type = "${spec.operators.embedding.family}"
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)`;
}

function renderEmbeddingForward(spec: HybridDecoderArchitectureSpec): string {
  return spec.operators.embedding.family === "gpt2"
    ? `        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        hidden_states = self.wte(input_ids) + self.wpe(positions)
        hidden_states = self.drop(hidden_states)`
    : `        hidden_states = self.embed_tokens(input_ids)`;
}

function renderFinalNormInit(spec: HybridDecoderArchitectureSpec): string {
  return spec.operators.finalNorm.family === "gpt2"
    ? `        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.final_norm_epsilon)`
    : `        self.final_norm = ${spec.operators.finalNorm.family === "phi3" ? "Phi3RMSNorm" : "LlamaRMSNorm"}(config.hidden_size, eps=config.final_norm_epsilon)`;
}

function renderRotaryInit(spec: HybridDecoderArchitectureSpec): string {
  const needsRotary = spec.operators.blocks.some((block) => block.family === "llama" || block.family === "phi3");
  const rotaryClass = spec.operators.blocks.some((block) => block.family === "phi3") ? "Phi3RotaryEmbedding" : "LlamaRotaryEmbedding";
  return needsRotary
    ? `        self.rotary_emb = ${rotaryClass}(config.max_llama_head_dim, base=config.max_rope_theta)`
    : "";
}

function renderBlockForward(spec: HybridDecoderArchitectureSpec): string {
  const hasRotaryBlocks = spec.operators.blocks.some((block) => block.family === "llama" || block.family === "phi3");
  if (!hasRotaryBlocks) {
    return `        for block in self.blocks:
            hidden_states = block(hidden_states)`;
  }

  if (!spec.operators.blocks.some((block) => block.family === "gpt2")) {
    return `        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for block in self.blocks:
            hidden_states = block(hidden_states, position_embeddings=position_embeddings)`;
  }

  return `        for block in self.blocks:
            if isinstance(block, GPT2Block):
                hidden_states = block(hidden_states)
            else:
                position_embeddings = self.rotary_emb(hidden_states, position_ids)
                hidden_states = block(hidden_states, position_embeddings=position_embeddings)`;
}

export function renderHybridModel(spec: HybridDecoderArchitectureSpec, primitiveSections: string): string {
  const blockLines = spec.operators.blocks.map(renderBlockInit).join(",\n");

  return `from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


${renderDataclasses(spec)}

${primitiveSections}


class HybridDecoderModel(nn.Module):
    def __init__(self, config: HybridConfig | None = None) -> None:
        super().__init__()
        self.config = config or HybridConfig()
${renderEmbeddingInit(spec)}
${renderRotaryInit(spec)}
        self.blocks = nn.ModuleList([
${blockLines}
        ])
${renderFinalNormInit(spec)}

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.max_position_embeddings:
            raise ValueError(f"Sequence length {seq_len} exceeds configured maximum {self.config.max_position_embeddings}")
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
${renderEmbeddingForward(spec)}
${renderBlockForward(spec)}
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
