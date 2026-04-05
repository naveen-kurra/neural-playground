from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare exported Mistral model.py against native Transformers Mistral.")
    parser.add_argument("--model-file", required=True, help="Path to generated Mistral model.py")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--intermediate-size", type=int, default=512)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--positions", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=8)
    return parser.parse_args()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("generated_mistral_model", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import generated module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["generated_mistral_model"] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    args = parse_args()

    try:
        import torch
        from transformers import MistralConfig as HFMistralConfig
        from transformers import MistralForCausalLM as HFMistralForCausalLM
    except Exception as exc:  # pragma: no cover
        print(f"Missing dependency: {exc}")
        return 2

    if args.hidden_size % args.heads != 0:
        print("hidden-size must be divisible by heads")
        return 2

    model_path = Path(args.model_file).resolve()
    generated = load_module(model_path)

    model_config = dict(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.layers,
        num_attention_heads=args.heads,
        num_key_value_heads=args.kv_heads,
        head_dim=args.head_dim,
        hidden_act="silu",
        max_position_embeddings=args.positions,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        attention_bias=False,
        attention_dropout=0.0,
        tie_word_embeddings=False,
        sliding_window=min(32, args.positions),
        use_cache=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )

    hf_config = HFMistralConfig(**model_config)
    hf_model = HFMistralForCausalLM(hf_config).eval()

    generated_config = generated.MistralConfig(
        **{
            key: value
            for key, value in model_config.items()
            if key not in {"use_cache", "pad_token_id", "bos_token_id", "eos_token_id"}
        },
        mlp_bias=False,
    )
    generated_model = generated.MistralForCausalLM(generated_config).eval()

    hf_modules = sorted(name for name, _ in hf_model.named_modules())
    generated_modules = sorted(name for name, _ in generated_model.named_modules())
    missing_from_generated = [name for name in hf_modules if name not in generated_modules]
    extra_in_generated = [name for name in generated_modules if name not in hf_modules]

    hf_state = hf_model.state_dict()
    generated_state = generated_model.state_dict()
    common_keys = sorted(set(hf_state) & set(generated_state))
    missing_state_keys = sorted(set(hf_state) - set(generated_state))
    extra_state_keys = sorted(set(generated_state) - set(hf_state))
    shape_mismatches = [
        key
        for key in common_keys
        if tuple(hf_state[key].shape) != tuple(generated_state[key].shape)
    ]

    transferable = {
        key: value
        for key, value in hf_state.items()
        if key in generated_state and tuple(value.shape) == tuple(generated_state[key].shape)
    }
    load_result = generated_model.load_state_dict(transferable, strict=False)

    input_ids = torch.randint(0, args.vocab_size, (2, args.seq_len), dtype=torch.long)
    with torch.no_grad():
        hf_logits = hf_model(input_ids=input_ids, use_cache=False).logits
        generated_logits = generated_model(input_ids)

    max_abs_diff = float((hf_logits - generated_logits).abs().max().item())
    mean_abs_diff = float((hf_logits - generated_logits).abs().mean().item())

    print("Mistral parity report")
    print(f"model_file: {model_path}")
    print(f"hf_module_count: {len(hf_modules)}")
    print(f"generated_module_count: {len(generated_modules)}")
    print(f"missing_modules_from_generated: {len(missing_from_generated)}")
    print(f"extra_modules_in_generated: {len(extra_in_generated)}")
    print(f"common_state_keys: {len(common_keys)}")
    print(f"missing_state_keys: {len(missing_state_keys)}")
    print(f"extra_state_keys: {len(extra_state_keys)}")
    print(f"shape_mismatches: {len(shape_mismatches)}")
    print(f"load_missing_keys: {len(load_result.missing_keys)}")
    print(f"load_unexpected_keys: {len(load_result.unexpected_keys)}")
    print(f"max_abs_diff: {max_abs_diff:.8f}")
    print(f"mean_abs_diff: {mean_abs_diff:.8f}")

    if missing_from_generated:
        print("\nMissing module names:")
        for name in missing_from_generated[:20]:
            print(f"  - {name}")
    if extra_in_generated:
        print("\nExtra module names:")
        for name in extra_in_generated[:20]:
            print(f"  - {name}")
    if missing_state_keys:
        print("\nMissing state keys:")
        for key in missing_state_keys[:20]:
            print(f"  - {key}")
    if extra_state_keys:
        print("\nExtra state keys:")
        for key in extra_state_keys[:20]:
            print(f"  - {key}")
    if shape_mismatches:
        print("\nShape mismatches:")
        for key in shape_mismatches[:20]:
            print(f"  - {key}: hf={tuple(hf_state[key].shape)} generated={tuple(generated_state[key].shape)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
