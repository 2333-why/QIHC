"""Score BBH multiple-choice candidates with a causal LM (real logits)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from qihc.orchestrator.bbh import BBHTask
    from qihc.orchestrator.reasoning import SubsetProblem


def enrich_problems_with_llm_logits(
    problems: list["SubsetProblem"],
    model_name: str = "distilgpt2",
    device: str | None = None,
) -> list["SubsetProblem"]:
    """Replace pseudo logits in SubsetProblem list with LM scores."""
    from qihc.orchestrator.reasoning import SubsetProblem

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError('LLM scoring requires: pip install -e ".[llm]"') from exc

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    model.to(device)

    enriched: list[SubsetProblem] = []
    for p in problems:
        cands = list(p.metadata.get("candidates", [])) or _candidates_from_meta(p)
        logits = _score_with_model(model, tokenizer, p.text, cands, device)
        meta = dict(p.metadata)
        meta["logits_source"] = "llm"
        meta["model_name"] = model_name
        enriched.append(
            SubsetProblem(
                text=p.text,
                logits=logits,
                top_k=p.top_k,
                exclusion_pairs=p.exclusion_pairs,
                metadata=meta,
            )
        )
    return enriched


def _score_with_model(model, tokenizer, question: str, candidates: list[str], device: str) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    answer_prefix = "\nAnswer: "
    prefix = question.strip() + answer_prefix
    prefix_ids = tokenizer(prefix, return_tensors="pt").input_ids.to(device)
    prefix_len = prefix_ids.shape[1]

    scores: list[float] = []
    with torch.no_grad():
        for cand in candidates:
            full = prefix + cand.strip()
            enc = tokenizer(full, return_tensors="pt").to(device)
            ids = enc.input_ids
            out = model(ids)
            log_probs = F.log_softmax(out.logits, dim=-1)
            total = 0.0
            for pos in range(prefix_len, ids.shape[1]):
                token_id = ids[0, pos].item()
                total += log_probs[0, pos - 1, token_id].item()
            scores.append(total)
    return np.asarray(scores, dtype=float)


def score_candidates_causal_lm(
    question: str,
    candidates: list[str],
    model_name: str = "distilgpt2",
    device: str | None = None,
    answer_prefix: str = "\nAnswer: ",
) -> np.ndarray:
    """
    Score each candidate by summed token log-probability of the answer span.

    Higher score = model assigns higher likelihood to that completion.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            'LLM scoring requires: pip install -e ".[llm]"'
        ) from exc

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    model.to(device)

    return _score_with_model(model, tokenizer, question, candidates, device)


def _candidates_from_meta(problem: "SubsetProblem") -> list[str]:
    cands = problem.metadata.get("candidates")
    if cands:
        return list(cands)
    raise ValueError("Problem metadata missing candidates for LLM scoring")
