from __future__ import annotations

import argparse
import json
import os
import gc
import re
import shutil
import sys
from pathlib import Path
from typing import Any

from safetensors import safe_open
from safetensors.torch import load_file, save_file

MODEL_LOAD_MAX_BYTES = 8 * 1024 * 1024 * 1024

LAYER_COUNT_CANDIDATES = [
    "num_hidden_layers",
    "n_layer",
    "num_layers",
    "n_layers",
    "text_config.num_hidden_layers",
    "language_config.num_hidden_layers",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Broad whole-layer checkpoint pruning.")
    parser.add_argument("--input-dir", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", default="")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--hf-token", default="")
    parser.add_argument("--strategy", default="drop")
    parser.add_argument("--layer-prefix", default="")
    parser.add_argument("--layer-count-key", default="")
    parser.add_argument("--dropped-layers", default="")
    return parser.parse_args()


def get_nested(data: dict[str, Any], path: str) -> Any:
    current: Any = data
    for segment in path.split("."):
        if not isinstance(current, dict) or segment not in current:
            return None
        current = current[segment]
    return current


def set_nested(data: dict[str, Any], path: str, value: Any) -> bool:
    current: Any = data
    segments = path.split(".")
    for segment in segments[:-1]:
        if not isinstance(current, dict) or segment not in current:
            return False
        current = current[segment]
    if not isinstance(current, dict):
        return False
    current[segments[-1]] = value
    return True


def detect_layer_count_key(config: dict[str, Any]) -> str | None:
    for path in LAYER_COUNT_CANDIDATES:
        value = get_nested(config, path)
        if isinstance(value, int) and value > 0:
            return path
    return None


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def log_progress(stage: str, message: str, **data: Any) -> None:
    payload = {
        "type": "log",
        "stage": stage,
        "message": message,
        **data,
    }
    print(json.dumps(payload), file=sys.stderr, flush=True)


def resolve_input_dir(args: argparse.Namespace) -> tuple[Path, bool]:
    if args.input_dir:
        input_dir = Path(args.input_dir).expanduser().resolve()
        if not input_dir.exists():
            raise SystemExit(f"Input directory does not exist: {input_dir}")
        log_progress("resolve-input", "Using local model directory.", inputDir=str(input_dir))
        return input_dir, False

    if not args.model_id:
        raise SystemExit("Provide either --input-dir or --model-id.")

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "Automatic model download requires `huggingface_hub`. Install it in the prune service Python environment."
        ) from exc

    cache_dir = (
        Path(args.cache_dir).expanduser().resolve()
        if args.cache_dir
        else Path(os.environ.get("NEURAL_PLAYGROUND_HF_CACHE", Path.home() / ".cache" / "neural-playground" / "hf")).expanduser().resolve()
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    log_progress("download", "Downloading model snapshot from Hugging Face.", modelId=args.model_id, cacheDir=str(cache_dir))

    download_path = snapshot_download(
        repo_id=args.model_id,
        cache_dir=str(cache_dir),
        token=args.hf_token or None,
    )
    log_progress("download", "Finished downloading model snapshot.", modelId=args.model_id, inputDir=str(download_path))
    return Path(download_path).resolve(), True


def collect_weight_keys(input_dir: Path, weight_index: dict[str, Any] | None) -> list[str]:
    if weight_index is not None:
        weight_map = weight_index.get("weight_map")
        if isinstance(weight_map, dict):
          return list(weight_map.keys())
    safetensor_files = sorted(input_dir.glob("*.safetensors"))
    if not safetensor_files:
        raise SystemExit("No safetensors files found in the input directory.")
    keys: list[str] = []
    with safe_open(str(safetensor_files[0]), framework="pt") as handle:
        keys.extend(list(handle.keys()))
    return keys


def detect_layer_prefix(keys: list[str]) -> tuple[str | None, list[int]]:
    prefix_matches: dict[str, set[int]] = {}
    for key in keys:
        match = re.match(r"^(.*?)(\d+)\.(.+)$", key)
        if not match:
            continue
        prefix = match.group(1)
        index = int(match.group(2))
        prefix_matches.setdefault(prefix, set()).add(index)

    ranked = sorted(
        ((prefix, sorted(indices)) for prefix, indices in prefix_matches.items()),
        key=lambda item: len(item[1]),
        reverse=True,
    )
    if not ranked:
        return None, []
    return ranked[0][0], ranked[0][1]


def build_layer_remap(kept_layers: list[int]) -> dict[int, int]:
    return {old_index: new_index for new_index, old_index in enumerate(kept_layers)}


def remap_layer_key(key: str, prefix: str, remap: dict[int, int]) -> str | None:
    if not key.startswith(prefix):
        return key
    suffix = key[len(prefix) :]
    match = re.match(r"^(\d+)\.(.+)$", suffix)
    if not match:
        return key
    old_index = int(match.group(1))
    if old_index not in remap:
        return None
    return f"{prefix}{remap[old_index]}.{match.group(2)}"


def copy_support_files(input_dir: Path, output_dir: Path, skip_names: set[str]) -> None:
    for path in input_dir.iterdir():
        if not path.is_file():
            continue
        if path.name in skip_names:
            continue
        if path.suffix == ".safetensors":
            continue
        shutil.copy2(path, output_dir / path.name)


def save_index(path: Path, weight_map: dict[str, str], total_size: int) -> None:
    payload = {
        "metadata": {
            "total_size": total_size
        },
        "weight_map": weight_map
    }
    path.write_text(json.dumps(payload, indent=2))


def prune_sharded_weights(input_dir: Path, output_dir: Path, weight_index: dict[str, Any], prefix: str, remap: dict[int, int]) -> tuple[list[str], int]:
    weight_map = weight_index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise SystemExit("Invalid model.safetensors.index.json: missing weight_map.")

    shards: dict[str, list[str]] = {}
    for tensor_key, shard_name in weight_map.items():
        if isinstance(shard_name, str):
            shards.setdefault(shard_name, []).append(tensor_key)

    new_weight_map: dict[str, str] = {}
    total_size = 0
    written_files: list[str] = []

    for shard_name, shard_keys in shards.items():
        log_progress("rewrite-shard", "Rewriting checkpoint shard.", shard=shard_name, tensorCount=len(shard_keys))
        source_path = input_dir / shard_name
        tensors = load_file(str(source_path))
        next_tensors = {}
        for key in shard_keys:
            tensor = tensors[key]
            new_key = remap_layer_key(key, prefix, remap)
            if new_key is None:
                continue
            next_tensors[new_key] = tensor
            new_weight_map[new_key] = shard_name
            total_size += tensor.numel() * tensor.element_size()

        if next_tensors:
            target_path = output_dir / shard_name
            save_file(next_tensors, str(target_path))
            written_files.append(target_path.name)
            log_progress("rewrite-shard", "Finished checkpoint shard.", shard=shard_name, keptTensorCount=len(next_tensors))

    save_index(output_dir / "model.safetensors.index.json", new_weight_map, total_size)
    written_files.append("model.safetensors.index.json")
    return written_files, total_size


def prune_single_safetensors(input_dir: Path, output_dir: Path, prefix: str, remap: dict[int, int]) -> tuple[list[str], int]:
    source_files = sorted(input_dir.glob("*.safetensors"))
    if not source_files:
        raise SystemExit("No safetensors files found in the input directory.")

    source_path = source_files[0]
    log_progress("rewrite-weights", "Rewriting single safetensors checkpoint.", source=str(source_path))
    tensors = load_file(str(source_path))
    next_tensors = {}
    total_size = 0
    for key, tensor in tensors.items():
        new_key = remap_layer_key(key, prefix, remap)
        if new_key is None:
            continue
        next_tensors[new_key] = tensor
        total_size += tensor.numel() * tensor.element_size()

    target_path = output_dir / source_path.name
    save_file(next_tensors, str(target_path))
    log_progress("rewrite-weights", "Finished single safetensors checkpoint.", target=str(target_path), keptTensorCount=len(next_tensors))
    return [target_path.name], total_size


def validate_pruned_output(
    input_dir: Path,
    output_dir: Path,
    layer_prefix: str,
    layer_count_key: str | None,
    original_layer_indices: list[int],
    kept_layers: list[int],
    total_size: int,
) -> dict[str, Any]:
    log_progress("validate", "Starting post-prune validation.")
    output_config = load_json(output_dir / "config.json")
    config_report = {
        "ok": True,
        "layerCountKey": layer_count_key,
        "expectedLayerCount": len(kept_layers),
        "configuredLayerCount": None,
    }
    if layer_count_key:
        configured_layer_count = get_nested(output_config, layer_count_key)
        config_report["configuredLayerCount"] = configured_layer_count
        config_report["ok"] = configured_layer_count == len(kept_layers)

    output_weight_index_path = output_dir / "model.safetensors.index.json"
    output_weight_index = load_json(output_weight_index_path) if output_weight_index_path.exists() else None
    output_keys = collect_weight_keys(output_dir, output_weight_index)
    output_detected_prefix, output_detected_indices = detect_layer_prefix(output_keys)
    checkpoint_report = {
        "ok": output_detected_prefix == layer_prefix and output_detected_indices == list(range(len(kept_layers))),
        "expectedLayerPrefix": layer_prefix,
        "detectedLayerPrefix": output_detected_prefix,
        "expectedLayerIndices": list(range(len(kept_layers))),
        "detectedLayerIndices": output_detected_indices,
        "writtenWeightBytes": total_size,
        "originalLayerCount": len(original_layer_indices),
        "keptLayerCount": len(kept_layers),
    }

    load_report: dict[str, Any] = {
        "ok": True,
        "originalConfigLoadOk": False,
        "prunedConfigLoadOk": False,
        "originalModelLoadOk": False,
        "prunedModelLoadOk": False,
        "skippedModelLoadReason": None,
        "warnings": [],
        "memory": {
            "device": "cpu",
            "originalModelBytes": None,
            "prunedModelBytes": None,
            "originalModelMiB": None,
            "prunedModelMiB": None,
        },
    }

    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError:
        load_report["ok"] = False
        load_report["warnings"].append("transformers is not installed; skipped model load validation.")
        log_progress("validate", "Skipped model load validation because transformers is not installed.")
    else:
        try:
            AutoConfig.from_pretrained(str(input_dir), trust_remote_code=False, local_files_only=True)
            load_report["originalConfigLoadOk"] = True
            AutoConfig.from_pretrained(str(output_dir), trust_remote_code=False, local_files_only=True)
            load_report["prunedConfigLoadOk"] = True
        except Exception as exc:
            load_report["ok"] = False
            load_report["warnings"].append(f"Config load validation failed: {exc}")

        if total_size > MODEL_LOAD_MAX_BYTES:
            reason = f"Skipped full model load validation because rewritten checkpoint is larger than {MODEL_LOAD_MAX_BYTES} bytes."
            load_report["skippedModelLoadReason"] = reason
            load_report["warnings"].append(reason)
            log_progress("validate", reason)
        else:
            def load_model_cpu(model_path: Path) -> tuple[bool, int | None, str | None]:
                try:
                    gc.collect()
                    model = AutoModelForCausalLM.from_pretrained(
                        str(model_path),
                        trust_remote_code=False,
                        local_files_only=True,
                        low_cpu_mem_usage=True,
                        attn_implementation="eager",
                        dtype=torch.float32,
                    )
                    model.eval()
                    bytes_used = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
                    del model
                    gc.collect()
                    return True, bytes_used, None
                except Exception as exc:
                    gc.collect()
                    return False, None, str(exc)

            try:
                original_ok, original_bytes, original_error = load_model_cpu(input_dir)
                load_report["originalModelLoadOk"] = original_ok
                load_report["memory"]["originalModelBytes"] = original_bytes
                load_report["memory"]["originalModelMiB"] = round(original_bytes / (1024 * 1024), 2) if original_bytes is not None else None
                if original_ok:
                    log_progress(
                        "validate",
                        "Loaded original model successfully on CPU.",
                        memoryMiB=load_report["memory"]["originalModelMiB"],
                    )
                elif original_error:
                    load_report["warnings"].append(f"Original model load validation failed: {original_error}")

                pruned_ok, pruned_bytes, pruned_error = load_model_cpu(output_dir)
                load_report["prunedModelLoadOk"] = pruned_ok
                load_report["memory"]["prunedModelBytes"] = pruned_bytes
                load_report["memory"]["prunedModelMiB"] = round(pruned_bytes / (1024 * 1024), 2) if pruned_bytes is not None else None
                if pruned_ok:
                    log_progress(
                        "validate",
                        "Loaded pruned model successfully on CPU.",
                        memoryMiB=load_report["memory"]["prunedModelMiB"],
                    )
                elif pruned_error:
                    load_report["warnings"].append(f"Pruned model load validation failed: {pruned_error}")
            except Exception as exc:
                load_report["warnings"].append(f"Unexpected model load validation error: {exc}")

        if not load_report["prunedConfigLoadOk"]:
            load_report["ok"] = False
        if load_report["skippedModelLoadReason"] is None and not load_report["prunedModelLoadOk"]:
            load_report["ok"] = False

    validation_report = {
        "ok": bool(config_report["ok"]) and bool(checkpoint_report["ok"]) and bool(load_report["ok"]),
        "config": config_report,
        "checkpoint": checkpoint_report,
        "load": load_report,
    }
    log_progress("validate", "Finished post-prune validation.", validationOk=validation_report["ok"])
    return validation_report


def main() -> None:
    args = parse_args()
    if args.strategy != "drop":
        raise SystemExit("Only `drop` strategy is supported right now.")

    log_progress("start", "Starting checkpoint pruning.", strategy=args.strategy)
    input_dir, downloaded_from_hub = resolve_input_dir(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_progress("prepare-output", "Prepared output directory.", outputDir=str(output_dir))

    config_path = input_dir / "config.json"
    if not config_path.exists():
        raise SystemExit("Input directory must contain config.json")
    config = load_json(config_path)

    layer_count_key = args.layer_count_key or detect_layer_count_key(config)
    weight_index_path = input_dir / "model.safetensors.index.json"
    weight_index = load_json(weight_index_path) if weight_index_path.exists() else None

    all_keys = collect_weight_keys(input_dir, weight_index)
    detected_prefix, detected_indices = detect_layer_prefix(all_keys)
    layer_prefix = args.layer_prefix or detected_prefix
    if not layer_prefix:
        raise SystemExit("Could not detect a repeating layer prefix.")
    if not detected_indices:
        raise SystemExit("Could not detect any layer indices.")

    dropped_layers = sorted(
        {int(value) for value in args.dropped_layers.split(",") if value.strip()}
    )
    kept_layers = [index for index in detected_indices if index not in dropped_layers]
    if not kept_layers:
        raise SystemExit("Pruning cannot remove every layer.")
    log_progress(
        "plan",
        "Computed pruning plan.",
        detectedLayerCount=len(detected_indices),
        keptLayerCount=len(kept_layers),
        droppedLayerCount=len(dropped_layers),
        layerPrefix=layer_prefix,
    )

    remap = build_layer_remap(kept_layers)
    if layer_count_key:
        set_nested(config, layer_count_key, len(kept_layers))
        log_progress("config", "Updated layer count in config.", layerCountKey=layer_count_key, newLayerCount=len(kept_layers))

    written_weight_files, total_size = (
        prune_sharded_weights(input_dir, output_dir, weight_index, layer_prefix, remap)
        if weight_index is not None
        else prune_single_safetensors(input_dir, output_dir, layer_prefix, remap)
    )

    (output_dir / "config.json").write_text(json.dumps(config, indent=2))
    log_progress("write-config", "Wrote updated config.json.", outputDir=str(output_dir))
    manifest = {
        "version": 1,
        "modelId": args.model_id,
        "strategy": args.strategy,
        "layerPrefix": layer_prefix,
        "layerCountKey": layer_count_key,
        "originalLayerIndices": detected_indices,
        "keptLayerIndices": kept_layers,
        "droppedLayerIndices": dropped_layers,
        "remap": [
            {"newIndex": new_index, "oldIndex": old_index}
            for old_index, new_index in remap.items()
        ],
        "totalSize": total_size,
    }
    (output_dir / "prune_manifest.json").write_text(json.dumps(manifest, indent=2))
    log_progress("write-manifest", "Wrote prune manifest.", outputDir=str(output_dir))

    skip_names = {
        "config.json",
        "prune_manifest.json",
        "model.safetensors.index.json",
        *written_weight_files,
    }
    copy_support_files(input_dir, output_dir, skip_names)
    log_progress("copy-support", "Copied support files.", copiedSupportFileCount=len([
        path for path in output_dir.iterdir()
        if path.is_file() and path.name not in {"config.json", "prune_manifest.json", *written_weight_files}
    ]))
    validation_report = validate_pruned_output(
        input_dir=input_dir,
        output_dir=output_dir,
        layer_prefix=layer_prefix,
        layer_count_key=layer_count_key,
        original_layer_indices=detected_indices,
        kept_layers=kept_layers,
        total_size=total_size,
    )

    result = {
        "ok": True,
        "downloadedFromHub": downloaded_from_hub,
        "inputDir": str(input_dir),
        "outputDir": str(output_dir),
        "layerPrefix": layer_prefix,
        "layerCountKey": layer_count_key,
        "keptLayerCount": len(kept_layers),
        "droppedLayerCount": len(dropped_layers),
        "writtenFiles": sorted(["config.json", "prune_manifest.json", *written_weight_files]),
        "copiedSupportFiles": sorted(
            [
                path.name
                for path in output_dir.iterdir()
                if path.is_file() and path.name not in {"config.json", "prune_manifest.json", *written_weight_files}
            ]
        ),
        "validation": validation_report,
    }
    log_progress("complete", "Checkpoint pruning completed.", outputDir=str(output_dir), writtenFileCount=len(result["writtenFiles"]))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
