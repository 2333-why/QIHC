"""LLM sampling for CR protocol (cached model, multi-GPU friendly)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from qihc.orchestrator.cr_protocol import (
    CRReasonSample,
    extract_answer_index,
    samples_from_completions,
)


@dataclass
class SamplerStats:
    n_completions: int = 0
    n_prompt_tokens: int = 0
    n_completion_tokens: int = 0
    wall_time_s: float = 0.0


@dataclass
class LLMSamplerConfig:
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    device: str | None = None
    dtype: str = "auto"
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_p: float = 0.95
    batch_size: int = 8
    system_prompt: str = (
        "You are a careful reasoning assistant. "
        "Think step by step, then give the final answer."
    )


class LLMSampler:
    """Sample chain-of-thought completions for CR-style reason aggregation."""

    def __init__(self, config: LLMSamplerConfig | None = None):
        self.config = config or LLMSamplerConfig()
        self._model = None
        self._tokenizer = None
        self.stats = SamplerStats()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError('LLM sampling requires: pip install -e ".[llm]"') from exc

        cfg = self.config
        device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        load_kwargs: dict = {"trust_remote_code": True}
        if cfg.dtype == "auto":
            load_kwargs["torch_dtype"] = "auto"
            load_kwargs["device_map"] = "auto" if device == "cuda" else None
        else:
            load_kwargs["torch_dtype"] = getattr(torch, cfg.dtype, torch.float16)

        self._model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **load_kwargs)
        if load_kwargs.get("device_map") is None:
            self._model.to(device)
        self._model.eval()
        self._device = device

    def _format_prompt(self, question: str, candidates: list[str]) -> str:
        opts = "\n".join(f"({chr(65+i)}) {c}" for i, c in enumerate(candidates))
        user = (
            f"{question.strip()}\n\nOptions:\n{opts}\n\n"
            "Let's think step by step, then state the final answer as (A), (B), etc."
        )
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": user},
        ]
        tok = self._tokenizer
        if hasattr(tok, "apply_chat_template"):
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return f"{self.config.system_prompt}\n\nUser: {user}\nAssistant:"

    def sample_completions(
        self,
        question: str,
        candidates: list[str],
        n_samples: int,
        seed: int = 0,
    ) -> list[str]:
        import time

        import torch

        self._ensure_loaded()
        t0 = time.perf_counter()
        prompt = self._format_prompt(question, candidates)
        tok = self._tokenizer
        model = self._model

        completions: list[str] = []
        torch.manual_seed(seed)
        enc = tok(prompt, return_tensors="pt")
        if hasattr(model, "device"):
            enc = {k: v.to(model.device) for k, v in enc.items()}
        elif self._device != "cpu":
            enc = {k: v.to(self._device) for k, v in enc.items()}

        prompt_len = enc["input_ids"].shape[1]
        self.stats.n_prompt_tokens += prompt_len * n_samples

        remaining = n_samples
        while remaining > 0:
            bs = min(self.config.batch_size, remaining)
            input_ids = enc["input_ids"].repeat(bs, 1)
            attn = enc.get("attention_mask", torch.ones_like(input_ids)).repeat(bs, 1)
            with torch.no_grad():
                out = model.generate(
                    input_ids,
                    attention_mask=attn,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=True,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    pad_token_id=tok.pad_token_id,
                )
            for row in out:
                new_tokens = row[prompt_len:]
                text = tok.decode(new_tokens, skip_special_tokens=True)
                completions.append(text.strip())
                self.stats.n_completion_tokens += int(new_tokens.numel())
            remaining -= bs

        self.stats.n_completions += len(completions)
        self.stats.wall_time_s += time.perf_counter() - t0
        return completions

    def sample_reasons(
        self,
        question: str,
        candidates: list[str],
        n_samples: int,
        seed: int = 0,
    ) -> list[CRReasonSample]:
        completions = self.sample_completions(question, candidates, n_samples, seed=seed)
        return samples_from_completions(completions, candidates)

    def score_candidates(
        self,
        question: str,
        candidates: list[str],
    ) -> np.ndarray:
        """Log-probability scores (same as bbh_llm) for zero-shot / greedy baseline."""
        from qihc.orchestrator.bbh_llm import score_candidates_causal_lm

        return score_candidates_causal_lm(
            question,
            candidates,
            model_name=self.config.model_name,
            device=self.config.device,
        )

    def zeroshot_answer(
        self,
        question: str,
        candidates: list[str],
    ) -> int:
        logits = self.score_candidates(question, candidates)
        return int(np.argmax(logits))
