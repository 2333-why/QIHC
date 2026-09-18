#!/usr/bin/env python3
"""Distributed LoRA post-training for SFT, DPO or GRPO using TRL."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["sft", "dpo", "grpo"], required=True); p.add_argument("--model-path", required=True)
    p.add_argument("--data", type=Path, required=True); p.add_argument("--output", required=True); p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--adapter-path", type=Path, help="Merge a previous LoRA stage before training this stage")
    p.add_argument("--learning-rate", type=float, default=2e-5); p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--lora-rank", type=int, default=16)
    args = p.parse_args()
    from datasets import Dataset
    from peft import LoraConfig
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rows = [json.loads(x) for x in args.data.read_text(encoding="utf-8").splitlines() if x]
    if not rows: raise SystemExit(f"No training rows in {args.data}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    # Preserve the supervised answer at the end when a long structural prompt
    # must be shortened to the configured context window.
    tokenizer.truncation_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    if args.adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(args.adapter_path), is_trainable=True)
    dataset = Dataset.from_list(rows); peft = LoraConfig(r=args.lora_rank, lora_alpha=2 * args.lora_rank, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM")
    trainer_peft = None if args.adapter_path else peft
    common = dict(output_dir=args.output, max_steps=args.max_steps, learning_rate=args.learning_rate, per_device_train_batch_size=1, gradient_accumulation_steps=8, bf16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}, logging_steps=5, save_steps=100, optim="adamw_torch_fused", report_to="none")

    def processing_kwargs(trainer_class):
        parameters = inspect.signature(trainer_class.__init__).parameters
        if "processing_class" in parameters:
            return {"processing_class": tokenizer}
        if "tokenizer" in parameters:
            return {"tokenizer": tokenizer}
        return {}

    if args.stage == "sft":
        from trl import SFTConfig, SFTTrainer
        columns = list(dataset.column_names)
        dataset = dataset.map(
            lambda x: {"text": x["prompt"] + "\n" + x["completion"]},
            remove_columns=columns,
        )
        config_kwargs = dict(common, max_length=args.max_length, dataset_text_field="text")
        if "completion_only_loss" in inspect.signature(SFTConfig).parameters:
            config_kwargs["completion_only_loss"] = False
        trainer = SFTTrainer(
            model=model, train_dataset=dataset, peft_config=trainer_peft,
            args=SFTConfig(**config_kwargs), **processing_kwargs(SFTTrainer),
        )
    elif args.stage == "dpo":
        from trl import DPOConfig, DPOTrainer
        trainer = DPOTrainer(
            model=model, train_dataset=dataset, peft_config=trainer_peft,
            args=DPOConfig(**common, max_length=args.max_length),
            **processing_kwargs(DPOTrainer),
        )
    else:
        from trl import GRPOConfig, GRPOTrainer
        gold_lookup = {r["prompt"]: r["gold_completion"] for r in rows}
        task_lookup = {r["prompt"]: r.get("task", "constraint") for r in rows}
        score_lookup = {r["prompt"]: float(r.get("recorded_reward", 0.0)) for r in rows}
        prompt_dataset = Dataset.from_list([{"prompt": r["prompt"]} for r in rows])
        def recorded_reward(prompts, completions, **kwargs):
            rewards = []
            for prompt, completion in zip(prompts, completions):
                try:
                    predicted = json.loads(completion)
                    gold = json.loads(gold_lookup[prompt])
                    if task_lookup[prompt] == "constraint":
                        schema = predicted.get("schema_version") == gold.get("schema_version") and isinstance(predicted.get("programs"), list)
                        exact = predicted.get("programs") == gold.get("programs")
                    else:
                        schema = isinstance(predicted.get("destroy_customers"), list) and isinstance(predicted.get("candidate_routes"), dict)
                        exact = predicted == gold
                    rewards.append((2.0 if exact else (0.25 if schema else -2.0)) + 0.05 * score_lookup[prompt])
                except Exception:
                    rewards.append(-2.0)
            return rewards
        trainer = GRPOTrainer(
            model=model, train_dataset=prompt_dataset, reward_funcs=recorded_reward,
            peft_config=trainer_peft,
            args=GRPOConfig(**common, max_completion_length=min(2048, args.max_length), num_generations=4),
            **processing_kwargs(GRPOTrainer),
        )
    trainer.train(); trainer.save_model(args.output)
    return 0


if __name__ == "__main__": raise SystemExit(main())
