from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare pure generated LLaMA against hybrid-generated LLaMA blocks.")
    parser.add_argument("--pure-model-file", required=True, help="Path to generated pure LLaMA model.py")
    parser.add_argument("--hybrid-model-file", required=True, help="Path to generated hybrid model.py")
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--intermediate-size", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--positions", type=int, default=32)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=8)
    return parser.parse_args()


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import generated module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def build_models(args: argparse.Namespace):
    import torch

    pure_mod = load_module(Path(args.pure_model_file).resolve(), "pure_generated_llama_model")
    hybrid_mod = load_module(Path(args.hybrid_model_file).resolve(), "hybrid_generated_llama_model")

    pure_cfg = pure_mod.LlamaConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.layers,
        num_attention_heads=args.heads,
        num_key_value_heads=args.kv_heads,
        max_position_embeddings=args.positions,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        hidden_act="silu",
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        tie_word_embeddings=False,
        head_dim=args.hidden_size // args.heads,
    )
    hybrid_cfg = hybrid_mod.HybridConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        max_position_embeddings=args.positions,
        tie_word_embeddings=False,
        embd_pdrop=0.0,
        final_norm_epsilon=1e-6,
        max_llama_head_dim=args.hidden_size // args.heads,
        max_rope_theta=10000.0,
    )

    pure = pure_mod.LlamaForCausalLM(pure_cfg).eval()
    hybrid = hybrid_mod.HybridForCausalLM(hybrid_cfg).eval()
    return torch, pure, hybrid


def build_key_mapping(pure_state: dict[str, object], hybrid_state: dict[str, object]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    direct = {
        "model.embed_tokens.weight": "model.embed_tokens.weight",
        "model.norm.weight": "model.final_norm.weight",
        "lm_head.weight": "lm_head.weight",
    }
    for pure_key, hybrid_key in direct.items():
        if pure_key in pure_state and hybrid_key in hybrid_state:
            mapping[pure_key] = hybrid_key

    for pure_key in pure_state:
        if pure_key.startswith("model.layers."):
            hybrid_key = pure_key.replace("model.layers.", "model.blocks.", 1)
            if hybrid_key in hybrid_state:
                mapping[pure_key] = hybrid_key

    return mapping


def main() -> int:
    args = parse_args()
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        print(f"Missing dependency: {exc}")
        return 2

    torch_mod, pure_model, hybrid_model = build_models(args)
    pure_state = pure_model.state_dict()
    hybrid_state = hybrid_model.state_dict()
    mapping = build_key_mapping(pure_state, hybrid_state)

    transferred = 0
    skipped: list[str] = []
    for pure_key, value in pure_state.items():
        hybrid_key = mapping.get(pure_key)
        if hybrid_key is None:
            skipped.append(pure_key)
            continue
        if tuple(value.shape) != tuple(hybrid_state[hybrid_key].shape):
            skipped.append(f"{pure_key} -> {hybrid_key} shape mismatch")
            continue
        hybrid_state[hybrid_key] = value.clone()
        transferred += 1

    load_result = hybrid_model.load_state_dict(hybrid_state, strict=False)

    input_ids = torch_mod.randint(0, args.vocab_size, (2, args.seq_len), dtype=torch.long)
    with torch_mod.no_grad():
        pure_logits = pure_model(input_ids)
        hybrid_logits = hybrid_model(input_ids)

    diff = (pure_logits - hybrid_logits).abs()
    print("Pure-vs-hybrid LLaMA parity report")
    print(f"pure_model_file: {Path(args.pure_model_file).resolve()}")
    print(f"hybrid_model_file: {Path(args.hybrid_model_file).resolve()}")
    print(f"pure_state_keys: {len(pure_state)}")
    print(f"hybrid_state_keys: {len(hybrid_state)}")
    print(f"transferred_keys: {transferred}")
    print(f"skipped_keys: {len(skipped)}")
    print(f"load_missing_keys: {len(load_result.missing_keys)}")
    print(f"load_unexpected_keys: {len(load_result.unexpected_keys)}")
    print(f"max_abs_diff: {float(diff.max().item()):.8f}")
    print(f"mean_abs_diff: {float(diff.mean().item()):.8f}")

    if skipped:
        print("\nSkipped mappings:")
        for item in skipped[:20]:
            print(f"  - {item}")

    if load_result.missing_keys:
        print("\nLoad missing keys:")
        for item in load_result.missing_keys[:20]:
            print(f"  - {item}")

    if load_result.unexpected_keys:
        print("\nLoad unexpected keys:")
        for item in load_result.unexpected_keys[:20]:
            print(f"  - {item}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
