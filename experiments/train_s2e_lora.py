#!/usr/bin/env python3
"""Offline four-GPU LoRA post-training for SFT, DPO or GRPO using TRL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["sft", "dpo", "grpo"], required=True); p.add_argument("--model-path", required=True)
    p.add_argument("--data", type=Path, required=True); p.add_argument("--output", required=True); p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--learning-rate", type=float, default=2e-5); p.add_argument("--max-length", type=int, default=4096)
    args = p.parse_args()
    from datasets import Dataset
    from peft import LoraConfig
    rows = [json.loads(x) for x in args.data.read_text(encoding="utf-8").splitlines() if x]
    if not rows: raise SystemExit(f"No training rows in {args.data}")
    dataset = Dataset.from_list(rows); peft = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM")
    common = dict(output_dir=args.output, max_steps=args.max_steps, learning_rate=args.learning_rate, per_device_train_batch_size=1, gradient_accumulation_steps=8, bf16=True, gradient_checkpointing=True, logging_steps=5, save_steps=100, report_to="none")
    if args.stage == "sft":
        from trl import SFTConfig, SFTTrainer
        dataset = dataset.map(lambda x: {"text": x["prompt"] + "\n" + x["completion"]})
        trainer = SFTTrainer(model=args.model_path, train_dataset=dataset, peft_config=peft, args=SFTConfig(**common, max_length=args.max_length, dataset_text_field="text"))
    elif args.stage == "dpo":
        from trl import DPOConfig, DPOTrainer
        trainer = DPOTrainer(model=args.model_path, train_dataset=dataset, peft_config=peft, args=DPOConfig(**common, max_length=args.max_length))
    else:
        from trl import GRPOConfig, GRPOTrainer
        gold_lookup = {r["prompt"]: r["gold_completion"] for r in rows}
        prompt_dataset = Dataset.from_list([{"prompt": r["prompt"]} for r in rows])
        def recorded_reward(prompts, completions, **kwargs):
            rewards = []
            for prompt, completion in zip(prompts, completions):
                try:
                    predicted = json.loads(completion)
                    gold = json.loads(gold_lookup[prompt])
                    schema = predicted.get("schema_version") == gold.get("schema_version") and isinstance(predicted.get("programs"), list)
                    exact = predicted.get("programs") == gold.get("programs")
                    rewards.append(2.0 if exact else (0.25 if schema else -2.0))
                except Exception:
                    rewards.append(-2.0)
            return rewards
        trainer = GRPOTrainer(model=args.model_path, train_dataset=prompt_dataset, reward_funcs=recorded_reward, peft_config=peft, args=GRPOConfig(**common, max_completion_length=min(2048, args.max_length), num_generations=4))
    trainer.train(); trainer.save_model(args.output)
    return 0


if __name__ == "__main__": raise SystemExit(main())
