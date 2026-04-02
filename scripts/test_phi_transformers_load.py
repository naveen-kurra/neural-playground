from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test Phi loading with Transformers.")
    parser.add_argument(
        "--model",
        default="microsoft/Phi-3-mini-4k-instruct",
        help="HF model id or local model directory.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=40,
        help="Maximum number of generated tokens.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Execution device. Defaults to CPU because Phi-3-mini often does not fit on small GPUs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available.")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    load_kwargs = {
        "trust_remote_code": False,
        "attn_implementation": "eager",
        "low_cpu_mem_usage": True,
    }
    if args.device == "cuda":
        load_kwargs["device_map"] = "auto"
        load_kwargs["torch_dtype"] = torch.float16
    else:
        load_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    if args.device == "cpu":
        model = model.to("cpu")
    model.eval()
    print(f"Loaded with native Transformers implementation on {args.device}.")

    messages = [
        {"role": "user", "content": "Who are you?"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
    print(generated)


if __name__ == "__main__":
    main()
