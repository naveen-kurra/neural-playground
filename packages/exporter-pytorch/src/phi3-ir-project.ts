import type { TrainingConfig } from "@neural-playground/block-schema";
import type { Phi3ArchitectureSpec } from "@neural-playground/ir-schema";
import { exportLlamaIrProjectFiles } from "./llama-ir-project";
import { exportPhi3IrToPyTorch } from "./phi3-ir-export";
import type { ProjectFileMap } from "./types";

function asLlamaCompatibleSpec(spec: Phi3ArchitectureSpec) {
  return {
    ...spec,
    family: "llama" as const
  };
}

function withPhi3BuildModel(modelPy: string): string {
  return `${modelPy}


def build_model(cfg, seq_len_override: int | None = None) -> Phi3ForCausalLM:
    model_cfg = cfg.model
    max_positions = seq_len_override or model_cfg.max_position_embeddings
    return Phi3ForCausalLM(
        config=Phi3Config(
            vocab_size=model_cfg.vocab_size,
            hidden_size=model_cfg.hidden_size,
            intermediate_size=model_cfg.intermediate_size,
            num_hidden_layers=model_cfg.num_hidden_layers,
            num_attention_heads=model_cfg.num_attention_heads,
            num_key_value_heads=model_cfg.num_key_value_heads,
            head_dim=model_cfg.head_dim,
            hidden_act=model_cfg.hidden_act,
            max_position_embeddings=max_positions,
            rms_norm_eps=model_cfg.rms_norm_eps,
            rope_theta=model_cfg.rope_theta,
            attention_bias=model_cfg.attention_bias,
            attention_dropout=model_cfg.attention_dropout,
            mlp_bias=model_cfg.mlp_bias,
            tie_word_embeddings=model_cfg.tie_word_embeddings,
        )
    )
`;
}

export function exportPhi3IrProjectFiles(spec: Phi3ArchitectureSpec, training: TrainingConfig): ProjectFileMap {
  const base = exportLlamaIrProjectFiles(asLlamaCompatibleSpec(spec), training);
  return {
    ...base,
    "README.md": String(base["README.md"]).replaceAll("LLaMA", "Phi-3").replaceAll("family: llama", "family: phi3"),
    "configs/model.yaml": String(base["configs/model.yaml"]).replace("model_family: llama", "model_family: phi3"),
    "src/kurra_ai_cb/model.py": withPhi3BuildModel(exportPhi3IrToPyTorch(spec))
  };
}
