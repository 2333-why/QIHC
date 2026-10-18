#!/usr/bin/env python3
"""Distributed LoRA post-training for SFT, DPO or GRPO using TRL."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
from pathlib import Path
from datetime import timedelta


def initialize_zero_runtime(torch, deepspeed, environ=None):
    """Initialize BOTH Torch and DeepSpeed communication before zero.Init/config.

    torchrun sets environment variables; it does not initialize either backend.
    Model loading triggers zero.Init before Trainer/Accelerate can do so for us.
    """
    env = os.environ if environ is None else environ
    required = ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
    if any(k not in env for k in required):
        raise RuntimeError("ZeRO training must be launched through torchrun, including single-GPU runs")
    rank, local, world = (int(env[k]) for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    if world < 1 or not 0 <= rank < world or not 0 <= local < torch.cuda.device_count():
        raise RuntimeError("Invalid distributed rank/world size or local GPU index")
    torch.cuda.set_device(local)
    deepspeed.init_distributed(dist_backend="nccl", auto_mpi_discovery=False, timeout=timedelta(minutes=10))
    if not torch.distributed.is_initialized() or not deepspeed.comm.is_initialized():
        raise RuntimeError("Torch/DeepSpeed communication was not initialized")
    actual = torch.distributed.get_world_size()
    if actual != world or deepspeed.comm.get_world_size() != world or torch.distributed.get_rank() != rank:
        raise RuntimeError("Initialized communication disagrees with torchrun rank/world size")
    print(f"ZERO_RUNTIME_READY rank={rank} local_rank={local} world_size={actual}", flush=True)
    return actual


def latest_checkpoint(output: Path) -> Path | None:
    checkpoints = []
    for path in output.glob("checkpoint-*") if output.exists() else []:
        try:
            checkpoints.append((int(path.name.rsplit("-", 1)[1]), path))
        except ValueError:
            continue
    return max(checkpoints, default=(0, None))[1]


def resolve_generation_batch(requested: int, world_size: int, num_generations: int, accumulation: int) -> int:
    """Generation memory budget is independent from optimizer accumulation."""
    if requested < 0 or min(world_size, num_generations, accumulation) <= 0:
        raise ValueError("Require nonnegative batch override and positive world/generation/accumulation counts")
    multiple = math.lcm(world_size, num_generations)
    if requested:
        if requested % multiple:
            raise ValueError(f"Generation batch {requested} must be divisible by world size {world_size} and num_generations {num_generations}")
        return requested
    return math.ceil(world_size * accumulation / multiple) * multiple


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["sft", "dpo", "grpo"], required=True); p.add_argument("--model-path", required=True)
    p.add_argument("--data", type=Path, required=True); p.add_argument("--output", required=True); p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--adapter-path", type=Path, help="Continue a previous LoRA adapter; DPO freezes a separate reference copy")
    p.add_argument("--learning-rate", type=float, default=2e-5); p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--dpo-beta", type=float, default=0.1)
    p.add_argument("--dpo-sft-weight", type=float, default=0.0,
                   help="Add chosen-answer NLL to DPO to preserve output format")
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
    p.add_argument("--generation-batch-size", type=int, default=0, help="Global GRPO generation batch; 0 derives it from accumulation")
    p.add_argument("--reward-mode", choices=["rollout", "recorded"], default="rollout")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-cpu", action="store_true")
    p.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--max-completion-length", type=int, default=1536)
    p.add_argument("--require-reward-variance", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if args.dpo_beta <= 0 or args.dpo_sft_weight < 0:
        p.error("DPO beta must be positive and SFT weight nonnegative")
    if args.use_cpu and args.deepspeed_config:
        p.error("CPU training smoke cannot use DeepSpeed")
    from datasets import Dataset
    from peft import LoraConfig
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    rows = [json.loads(x) for x in args.data.read_text(encoding="utf-8").splitlines() if x]
    if not rows: raise SystemExit(f"No training rows in {args.data}")
    if args.stage == "grpo" and args.reward_mode == "rollout":
        missing = [row for row in rows if row.get("task") != "constraint" and not row.get("reward_context")]
        if missing:
            raise SystemExit(f"{len(missing)} proposal rows have no execution reward context; regenerate with --export-training-context")
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
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        rows = [{**row, "prompt": tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}], tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )} for row in rows]
    # Preserve the supervised answer at the end when a long structural prompt
    # must be shortened to the configured context window.
    tokenizer.truncation_side = "left"
    for row_idx, row in enumerate(rows):
        lengths = [len(tokenizer.encode(row["prompt"], add_special_tokens=False))]
        for key in ("completion", "chosen", "rejected"):
            if key in row:
                lengths.append(len(tokenizer.encode(row[key], add_special_tokens=False)))
        required = lengths[0] + (max(lengths[1:], default=0) + 1 if args.stage != "grpo" else args.max_completion_length)
        if required > args.max_length:
            raise SystemExit(f"Training row {row_idx} requires {required} tokens > --max-length {args.max_length}; refusing silent truncation")
    deepspeed_config = None
    deepspeed_model_config = None
    if args.deepspeed_config:
        from transformers.integrations import HfDeepSpeedConfig
        import deepspeed
        world = initialize_zero_runtime(torch, deepspeed)
        deepspeed_config = json.loads(args.deepspeed_config.read_text(encoding="utf-8"))
        deepspeed_config.update({"train_micro_batch_size_per_gpu": 1,
                                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                                "train_batch_size": world * args.gradient_accumulation_steps,
                                "gradient_clipping": 1.0})
        deepspeed_model_config = HfDeepSpeedConfig(deepspeed_config)
    model = model_class.from_pretrained(
        args.model_path,
        local_files_only=True,
        dtype=torch.float32 if args.precision == "fp32" else torch.bfloat16,
        low_cpu_mem_usage=not bool(deepspeed_model_config and deepspeed_model_config.is_zero3()),
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    if args.adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(args.adapter_path), is_trainable=True)
    target_modules = [value.strip() for value in args.lora_target_modules.split(",") if value.strip()]
    peft = LoraConfig(r=args.lora_rank, lora_alpha=2 * args.lora_rank, lora_dropout=0.05, target_modules=target_modules, task_type="CAUSAL_LM")
    trainer_peft = None if args.adapter_path else peft
    common = dict(
        output_dir=args.output, max_steps=args.max_steps,
        learning_rate=args.learning_rate, per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=args.precision == "bf16", use_cpu=args.use_cpu, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False, logging_steps=args.logging_steps,
        save_steps=args.save_steps, save_total_limit=2,
        optim="adamw_torch" if args.use_cpu else "adamw_torch_fused", report_to="none", seed=args.seed,
        deepspeed=deepspeed_config,
        ddp_timeout=43200,
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
        processed = []
        for row in rows:
            prompt_ids = tokenizer.encode(row["prompt"], add_special_tokens=False)
            completion_ids = tokenizer.encode(row["completion"], add_special_tokens=False)
            if not completion_ids or completion_ids[-1] != tokenizer.eos_token_id:
                completion_ids.append(tokenizer.eos_token_id)
            processed.append({"input_ids": prompt_ids + completion_ids,
                              "completion_mask": [0] * len(prompt_ids) + [1] * len(completion_ids)})
        dataset = Dataset.from_list(processed)
        config_kwargs = dict(common, max_length=args.max_length)
        if "completion_only_loss" in inspect.signature(SFTConfig).parameters:
            config_kwargs["completion_only_loss"] = True
        trainer = SFTTrainer(
            model=model, train_dataset=dataset, peft_config=trainer_peft,
            args=SFTConfig(**config_kwargs), **processing_kwargs(SFTTrainer),
        )
    elif args.stage == "dpo":
        from trl import DPOConfig, DPOTrainer
        dataset = Dataset.from_list([{key: row[key] for key in ("prompt", "chosen", "rejected")} for row in rows])
        reference_kwargs = {}
        dpo_config = dict(common, max_length=args.max_length, beta=args.dpo_beta)
        # Newer TRL can omit prompt logits, reducing the large-vocabulary peak.
        # This does not remove completion logits; the full-model smoke remains mandatory.
        if "use_logits_to_keep" in inspect.signature(DPOConfig).parameters:
            dpo_config["use_logits_to_keep"] = True
        if args.dpo_sft_weight:
            parameters = inspect.signature(DPOConfig).parameters
            if "loss_weights" in parameters:
                dpo_config.update(loss_type=["sigmoid", "sft"], loss_weights=[1.0, args.dpo_sft_weight])
            elif "rpo_alpha" in parameters:
                dpo_config["rpo_alpha"] = args.dpo_sft_weight
            else:
                raise RuntimeError("Installed TRL cannot add chosen-answer supervision; refusing unregularized DPO")
        if args.adapter_path and "ref_adapter_name" in inspect.signature(DPOTrainer.__init__).parameters:
            model.load_adapter(str(args.adapter_path), adapter_name="ref", is_trainable=False)
            reference_kwargs = {"model_adapter_name": "default", "ref_adapter_name": "ref"}
        trainer = DPOTrainer(
            model=model, train_dataset=dataset, peft_config=trainer_peft,
            args=DPOConfig(**dpo_config),
            **processing_kwargs(DPOTrainer), **reference_kwargs,
        )
        if args.adapter_path and "ref" not in getattr(trainer.model, "peft_config", {}):
            raise RuntimeError("DPO did not create a frozen copy of the previous adapter; upgrade TRL/PEFT before training")
    else:
        from trl import GRPOConfig, GRPOTrainer
        gold_lookup = {r["prompt"]: r["gold_completion"] for r in rows}
        task_lookup = {r["prompt"]: r.get("task", "constraint") for r in rows}
        score_lookup = {r["prompt"]: float(r.get("recorded_reward", 0.0)) for r in rows}
        context_lookup = {r["prompt"]: r.get("reward_context") for r in rows}
        prompt_dataset = Dataset.from_list([
            {"prompt": r["prompt"], "task": r.get("task", "constraint"),
             "gold_completion": r["gold_completion"], "recorded_reward": float(r.get("recorded_reward", 0)),
             "reward_context": json.dumps(r.get("reward_context"), ensure_ascii=False)}
            for r in rows
        ])
        def recorded_reward(prompts, completions, task=None, gold_completion=None, reward_context=None, **kwargs):
            rewards = []
            for idx, (prompt, completion) in enumerate(zip(prompts, completions)):
                try:
                    import sys
                    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
                    from qihc.problems.cvrp.json_output import extract_answer_json
                    predicted = extract_answer_json(completion)
                    gold = json.loads(gold_completion[idx] if gold_completion is not None else gold_lookup[prompt])
                    current_task = task[idx] if task is not None else task_lookup[prompt]
                    if current_task == "constraint":
                        import sys
                        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
                        from qihc.problems.cvrp.training_reward import constraint_reward
                        rewards.append(constraint_reward(predicted, gold))
                        continue
                    elif args.reward_mode == "rollout":
                        import sys
                        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
                        from qihc.problems.cvrp.training_reward import rollout_reward
                        context = reward_context[idx] if reward_context is not None else context_lookup[prompt]
                        if isinstance(context, str):
                            context = json.loads(context)
                        rewards.append(rollout_reward(predicted, context))
                        continue
                    else:
                        schema = isinstance(predicted.get("destroy_customers"), list) and isinstance(predicted.get("candidate_routes"), dict)
                        exact = predicted == gold
                    rewards.append((2.0 if exact else (0.25 if schema else -2.0)) + 0.05 * score_lookup[prompt])
                except Exception:
                    rewards.append(-2.0)
            return rewards
        grpo_kwargs = dict(common, max_completion_length=args.max_completion_length, num_generations=args.num_generations)
        grpo_parameters = inspect.signature(GRPOConfig).parameters
        if "generation_batch_size" in grpo_parameters:
            grpo_kwargs["generation_batch_size"] = resolve_generation_batch(
                args.generation_batch_size, int(os.environ.get("WORLD_SIZE", "1")),
                args.num_generations, args.gradient_accumulation_steps)
        elif args.generation_batch_size:
            raise RuntimeError("This TRL version does not support explicit generation_batch_size")
        if "max_prompt_length" in grpo_parameters:
            grpo_kwargs["max_prompt_length"] = args.max_length - args.max_completion_length
        trainer = GRPOTrainer(
            model=model, train_dataset=prompt_dataset, reward_funcs=recorded_reward,
            peft_config=trainer_peft,
            args=GRPOConfig(**grpo_kwargs),
            **processing_kwargs(GRPOTrainer),
        )
    checkpoint = latest_checkpoint(Path(args.output)) if args.resume else None
    trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
    if args.stage == "grpo" and args.require_reward_variance:
        variances = [float(row["reward_std"]) for row in trainer.state.log_history if "reward_std" in row]
        if not variances or max(variances) <= 1e-12:
            raise RuntimeError("GRPO had no measured reward variation; refusing to mark ineffective training as complete")
    trainer.save_model(args.output)
    if trainer.is_world_process_zero():
        (Path(args.output) / "qihc_training_manifest.json").write_text(
            json.dumps({
                "stage": args.stage,
                "dpo_beta": args.dpo_beta if args.stage == "dpo" else None,
                "dpo_sft_weight": args.dpo_sft_weight if args.stage == "dpo" else None,
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
                "reward_mode": args.reward_mode if args.stage == "grpo" else None,
                "generation_batch_size": getattr(trainer.args, "generation_batch_size", None) if args.stage == "grpo" else None,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "world_size": int(os.environ.get("WORLD_SIZE", "1")),
                "training_metrics": trainer.state.log_history,
                "precision": args.precision,
                "reference_policy": "previous_adapter" if args.stage == "dpo" and args.adapter_path else "base_or_unused",
            }, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__": raise SystemExit(main())
