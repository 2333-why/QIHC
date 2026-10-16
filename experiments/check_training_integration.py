#!/usr/bin/env python3
"""Exercise actual SFT -> DPO -> GRPO -> adapter inference on a tiny local model.

No pretrained weights are downloaded. This checks APIs and weight persistence,
not the quality of the learned policy or distributed 35B memory requirements.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grpo-generation-batch-size", type=int, default=4)
    args = parser.parse_args()
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    from qihc.problems.cvrp.instance import generate_synthetic_instance

    root = args.output.resolve(); root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42); torch.set_num_threads(1)
    gold = json.dumps({"constraints": [{"type": "same_resource", "hard": True, "params": {"entities": [1, 2]}}]})
    rejected = json.dumps({"constraints": [{"type": "mutual_exclusion", "hard": True, "params": {"entities": [1, 2]}}]})
    prompt = "Customers 1 and 2 must use the same vehicle. Return Constraint IR JSON."
    instance = generate_synthetic_instance(4, 2, 40, seed=4, semantic_constraints=False)
    proposal = json.dumps({"destroy_customers": [1], "candidate_routes": {"1": [0, 1]}, "candidate_route_logits": {"1": {"0": 1, "1": -1}}, "confidence": 1})
    context = {"instance": instance.to_dict(), "routes": [[1, 2], [3, 4]], "allowed_routes": {"1": [0, 1]}, "seed": 3}
    rows = {
        "sft": [{"task": "constraint", "prompt": prompt, "completion": gold},
                {"task": "proposal", "prompt": "Choose a route for customer 1; allowed routes are 0 and 1.", "completion": proposal}],
        "dpo": [{"task": "constraint", "prompt": prompt, "chosen": gold, "rejected": rejected}],
        "grpo": [{"task": "constraint", "prompt": prompt, "gold_completion": gold},
                 {"task": "proposal", "prompt": "Choose a route for customer 1; allowed routes are 0 and 1.", "gold_completion": proposal, "reward_context": context}],
    }
    tokenizer_impl = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer_impl.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer_impl.decoder = decoders.ByteLevel()
    tokenizer_impl.train_from_iterator([prompt, gold, rejected, proposal] * 5,
        trainers.BpeTrainer(vocab_size=300, special_tokens=["<pad>", "<unk>", "<eos>"]))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_impl, pad_token="<pad>", unk_token="<unk>", eos_token="<eos>")
    tokenizer.chat_template = "{% for message in messages %}{{message['role'] + ': ' + message['content'] + '\\n'}}{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
    model_root = root / "model"
    tokenizer.save_pretrained(model_root)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=len(tokenizer), n_embd=32, n_layer=1, n_head=2,
                                      n_positions=512, bos_token_id=tokenizer.eos_token_id,
                                      eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id))
    model.save_pretrained(model_root)
    previous = None; summary = {"purpose": "training API integration only", "stages": {}}
    for stage in ("sft", "dpo", "grpo"):
        data = root / f"{stage}.jsonl"
        data.write_text("".join(json.dumps(row) + "\n" for row in rows[stage]), encoding="utf-8")
        adapter = root / "adapters" / stage
        command = [sys.executable, str(ROOT / "experiments/train_s2e_lora.py"), "--stage", stage,
                   "--model-path", str(model_root), "--data", str(data), "--output", str(adapter),
                   "--max-steps", "1", "--max-length", "512", "--max-completion-length", "32",
                   "--num-generations", "2", "--gradient-accumulation-steps", "2",
                   "--lora-rank", "2", "--lora-target-modules", "c_attn,c_proj,c_fc",
                   "--use-cpu", "--precision", "fp32", "--logging-steps", "1", "--save-steps", "1"]
        if previous is not None:
            command += ["--adapter-path", str(previous)]
        if stage == "dpo":
            command += ["--dpo-beta", "0.2", "--dpo-sft-weight", "0.2"]
        if stage == "grpo":
            command += ["--generation-batch-size", str(args.grpo_generation_batch_size)]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false"}
        for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
            env.pop(key, None)
        print(f"TRAINING SMOKE {stage}", flush=True)
        with (root / f"{stage}.log").open("w", encoding="utf-8") as handle:
            completed = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, env=env)
        if completed.returncode:
            raise RuntimeError(f"Actual {stage} training failed; see {root / (stage + '.log')}")
        manifest = json.loads((adapter / "qihc_training_manifest.json").read_text(encoding="utf-8"))
        weights = adapter / "adapter_model.safetensors"
        if not weights.is_file() or weights.stat().st_size == 0:
            raise RuntimeError(f"No saved weights after {stage}")
        metrics = manifest["training_metrics"]
        if any(not math.isfinite(float(item[key])) for item in metrics for key in ("loss", "train_loss") if key in item):
            raise RuntimeError(f"Non-finite training loss in {stage}")
        summary["stages"][stage] = {"weights_bytes": weights.stat().st_size, "metrics": metrics}
        if stage == "dpo":
            from safetensors.torch import load_file
            initial = load_file(str(previous / "adapter_model.safetensors"))
            reference = load_file(str(adapter / "ref/adapter_model.safetensors"))
            if initial.keys() != reference.keys() or any(not torch.equal(initial[key], reference[key]) for key in initial):
                raise RuntimeError("DPO reference weights are not the frozen SFT adapter")
            summary["stages"][stage]["reference_equals_sft_adapter"] = True
        previous = adapter
    from qihc.problems.cvrp.llm_selector import LocalLLMNeighborhoodSelector
    frontend = LocalLLMNeighborhoodSelector(str(model_root), adapter_path=str(previous), device="cpu", max_new_tokens=8)
    generated = frontend._generate(prompt)
    summary.update({"status": "complete", "adapter_inference_exercised": True,
                    "sample_output": generated, "random_tiny_model_is_not_a_quality_benchmark": True})
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Actual training and adapter inference completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
