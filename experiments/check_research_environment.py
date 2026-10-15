#!/usr/bin/env python3
"""Fail early on missing GPU/model/training APIs, and record runtime provenance."""

import argparse
import importlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--nproc", type=int, required=True)
    parser.add_argument("--training", action="store_true")
    parser.add_argument("--cpu-only", action="store_true", help="Metadata/API checks only; explicitly defer CUDA and full-model execution")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packages = ["torch", "transformers", "ortools", "scipy", "psutil"]
    if args.training:
        packages += ["datasets", "peft", "trl"]
        if args.nproc > 1 or args.cpu_only:
            packages.append("deepspeed")
    versions = {}
    for package in packages:
        importlib.import_module(package)
        versions[package] = version(package)
    import torch
    if not args.cpu_only and (not torch.cuda.is_available() or torch.cuda.device_count() < args.nproc):
        raise RuntimeError(f"Requested {args.nproc} GPU processes; available CUDA devices={torch.cuda.device_count()}")
    from transformers import AutoConfig, AutoTokenizer
    config = AutoConfig.from_pretrained(args.model_path, local_files_only=True)
    if not list(args.model_path.glob("*.safetensors")) and not list(args.model_path.glob("*.bin")):
        raise RuntimeError(f"Missing model weights: {args.model_path}")
    if str(config.model_type).startswith("qwen3_5"):
        from transformers import AutoModelForMultimodalLM, AutoProcessor
        _ = AutoModelForMultimodalLM._model_mapping[type(config)]
        processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
        tokenizer = processor.tokenizer
        prompt = "Customers 1 and 2 must share a vehicle."
        inference = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
            enable_thinking=False,
        )["input_ids"][0].tolist()
        rendered = tokenizer.apply_chat_template([{ "role": "user", "content": prompt}], tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=False)
        training = tokenizer.encode(rendered, add_special_tokens=False)
        if inference != training:
            raise RuntimeError("Qwen processor and training tokenizer produce different text prompt tokens")
    else:
        from transformers import AutoModelForCausalLM
        _ = AutoModelForCausalLM._model_mapping[type(config)]
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    if tokenizer.eos_token_id is None:
        raise RuntimeError("Training/inference requires an EOS token")
    if args.training:
        from trl import SFTConfig, SFTTrainer, DPOConfig, DPOTrainer, GRPOConfig, GRPOTrainer
    gpus = []
    for index in range(0 if args.cpu_only else args.nproc):
        with torch.cuda.device(index):
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError(f"GPU {index} does not support the configured BF16 training")
            a = torch.ones((8, 8), dtype=torch.bfloat16, device=f"cuda:{index}")
            _ = a @ a
            torch.cuda.synchronize(index)
            free, total = torch.cuda.mem_get_info(index)
            gpus.append({"index": index, "name": torch.cuda.get_device_name(index),
                         "free_bytes": free, "total_bytes": total})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"versions": versions, "python": sys.version,
                                     "platform": platform.platform(), "gpus": gpus,
                                     "model_type": config.model_type, "torch_cuda": torch.version.cuda, "status": "passed",
                                     "scope": "cpu_metadata_and_apis" if args.cpu_only else "cuda_metadata_and_apis",
                                     "deferred": ["full_model_forward", "gpu_memory", "nccl", "zero3"] if args.cpu_only else ["full_model_forward", "distributed_training"]}, indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed", "gpus": len(gpus), "versions": versions}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
