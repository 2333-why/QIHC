#!/usr/bin/env python3
"""Distributed LoRA post-training for SFT, DPO or GRPO using TRL."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path


def latest_checkpoint(output: Path) -> Path | None:
    checkpoints = []
    for path in output.glob("checkpoint-*") if output.exists() else []:
        try:
            checkpoints.append((int(path.name.rsplit("-", 1)[1]), path))
        except ValueError:
            continue
    return max(checkpoints, default=(0, None))[1]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["sft", "dpo", "grpo"], required=True); p.add_argument("--model-path", required=True)
    p.add_argument("--data", type=Path, required=True); p.add_argument("--output", required=True); p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--adapter-path", type=Path, help="Merge a previous LoRA stage before training this stage")
    p.add_argument("--learning-rate", type=float, default=2e-5); p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,gate_up_proj",
        help="Comma-separated language-backbone linear module suffixes",
    )
    p.add_argument("--deepspeed-config", type=Path)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--num-generations", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    from datasets import Dataset
    from peft import LoraConfig
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    rows = [json.loads(x) for x in args.data.read_text(encoding="utf-8").splitlines() if x]
    if not rows: raise SystemExit(f"No training rows in {args.data}")
    model_config = AutoConfig.from_pretrained(args.model_path, local_files_only=True)
    model_type = str(getattr(model_config, "model_type", ""))
    model_class = AutoModelForCausalLM
    if model_type.startswith("qwen3_5"):
        from transformers import AutoModelForMultimodalLM, AutoProcessor
        processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
        tokenizer = processor.tokenizer
        model_class = AutoModelForMultimodalLM
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    # Preserve the supervised answer at the end when a long structural prompt
    # must be shortened to the configured context window.
    tokenizer.truncation_side = "left"
    model = model_class.from_pretrained(
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
    target_modules = [value.strip() for value in args.lora_target_modules.split(",") if value.strip()]
    dataset = Dataset.from_list(rows); peft = LoraConfig(r=args.lora_rank, lora_alpha=2 * args.lora_rank, lora_dropout=0.05, target_modules=target_modules, task_type="CAUSAL_LM")
    trainer_peft = None if args.adapter_path else peft
    common = dict(
        output_dir=args.output, max_steps=args.max_steps,
        learning_rate=args.learning_rate, per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False, logging_steps=args.logging_steps,
        save_steps=args.save_steps, save_total_limit=2,
        optim="adamw_torch_fused", report_to="none", seed=args.seed,
        deepspeed=str(args.deepspeed_config) if args.deepspeed_config else None,
    )

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
            args=GRPOConfig(**common, max_completion_length=min(2048, args.max_length), num_generations=args.num_generations),
            **processing_kwargs(GRPOTrainer),
        )
    checkpoint = latest_checkpoint(Path(args.output)) if args.resume else None
    trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
    trainer.save_model(args.output)
    if trainer.is_world_process_zero():
        (Path(args.output) / "qihc_training_manifest.json").write_text(
            json.dumps({
                "stage": args.stage,
                "model_path": args.model_path,
                "adapter_path": str(args.adapter_path) if args.adapter_path else None,
                "data": str(args.data),
                "rows": len(rows),
                "max_steps": args.max_steps,
                "max_length": args.max_length,
                "lora_rank": args.lora_rank,
                "lora_target_modules": target_modules,
                "model_type": model_type,
                "deepspeed_config": str(args.deepspeed_config) if args.deepspeed_config else None,
                "resumed_from": str(checkpoint) if checkpoint else None,
            }, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__": raise SystemExit(main())
