import type { TrainingConfig } from "@neural-playground/block-schema";
import type { Gemma4ArchitectureSpec } from "@neural-playground/ir-schema";
import { exportLlamaIrProjectFiles } from "./llama-ir-project";
import { exportGemma4IrToPyTorch } from "./gemma4-ir-export";
import type { ProjectFileMap } from "./types";

function asLlamaCompatibleSpec(spec: Gemma4ArchitectureSpec) {
  return {
    ...spec,
    family: "llama" as const
  };
}

function withGemma4BuildModel(modelPy: string): string {
  return `${modelPy}


def build_model(cfg, seq_len_override: int | None = None) -> Gemma4ForCausalLM:
    model_cfg = cfg.model
    max_positions = int(model_cfg["max_position_embeddings"])
    if seq_len_override is not None:
        max_positions = seq_len_override
    return Gemma4ForCausalLM(
        config=Gemma4Config(
            vocab_size=int(model_cfg["vocab_size"]),
            hidden_size=int(model_cfg["hidden_size"]),
            intermediate_size=int(model_cfg["intermediate_size"]),
            num_hidden_layers=int(model_cfg["num_hidden_layers"]),
            num_attention_heads=int(model_cfg["num_attention_heads"]),
            num_key_value_heads=int(model_cfg["num_key_value_heads"]),
            max_position_embeddings=max_positions,
            rms_norm_eps=float(model_cfg["rms_norm_eps"]),
            rope_theta=float(model_cfg["rope_theta"]),
            hidden_act=str(model_cfg["hidden_act"]),
            attention_bias=bool(model_cfg["attention_bias"]),
            attention_dropout=float(model_cfg["attention_dropout"]),
            mlp_bias=bool(model_cfg["mlp_bias"]),
            tie_word_embeddings=bool(model_cfg["tie_word_embeddings"]),
            head_dim=int(model_cfg["head_dim"]),
            sliding_window=int(model_cfg.get("sliding_window", 512)),
            layer_types=tuple(model_cfg.get("layer_types", tuple("sliding_attention" for _ in range(int(model_cfg["num_hidden_layers"]) - 1)) + ("full_attention",))),
            num_global_key_value_heads=(
                None if model_cfg.get("num_global_key_value_heads") is None else int(model_cfg["num_global_key_value_heads"])
            ),
            global_head_dim=int(model_cfg.get("global_head_dim", model_cfg["head_dim"])),
            attention_k_eq_v=bool(model_cfg.get("attention_k_eq_v", False)),
            num_kv_shared_layers=int(model_cfg.get("num_kv_shared_layers", 0)),
            rope_parameters=model_cfg.get("rope_parameters"),
        )
    )
`;
}

export function exportGemma4IrProjectFiles(spec: Gemma4ArchitectureSpec, training: TrainingConfig): ProjectFileMap {
  const base = exportLlamaIrProjectFiles(asLlamaCompatibleSpec(spec), training);
  return {
    ...base,
    "README.md": String(base["README.md"]).replaceAll("LLaMA", "Gemma 4").replaceAll("family: llama", "family: gemma4"),
    "configs/model.yaml": String(base["configs/model.yaml"]).replace("model_family: llama", "model_family: gemma4"),
    "src/kurra_ai_cb/model.py": withGemma4BuildModel(exportGemma4IrToPyTorch(spec))
  };
}
