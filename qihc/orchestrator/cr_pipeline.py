"""
CR paper-aligned end-to-end pipeline (arXiv:2407.00071).

Modes follow the original paper — NOT logits-greedy:
  zeroshot   : LLM direct answer, temperature=0, no reason sampling
  linear     : sample N completions (T=1) → majority vote on answers
  random     : sample → random κ reasons → enhanced prompt → LLM T=0
  quadratic  : sample → dedup → QUBO reason selection → enhanced prompt → LLM T=0
  vci-1      : CR-encoded logits + one-way p-bit on constrained answer mask (CR limit)
  vci-2      : CR-encoded logits + full VCI q↔s loop on constrained answer mask

Legacy logits-greedy selection lives in cr_protocol.select_mask_cr (deprecated for paper tables).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np

from qihc.orchestrator.bbh import BBHTask
from qihc.orchestrator.cr_protocol import (
    CRParams,
    CRReasonSample,
    build_cr_qubo,
    build_reason_cooccurrence,
    cr_logits_from_samples,
    deduplicate_reasons,
    samples_from_completions,
)
from qihc.orchestrator.encoder import refine_mask_to_top_k
from qihc.orchestrator.reasoning import SubsetProblem, is_feasible

CRPaperMode = Literal["zeroshot", "random", "linear", "quadratic", "vci-1", "vci-2"]

CR_SYSTEM_PROMPT = (
    "You are a careful reasoning assistant. Answer multiple-choice questions "
    "by thinking step by step when helpful, then state the final option as (A), (B), etc."
)

CR_REASON_PROMPT_SUFFIX = (
    "\n\nLet's think step by step, then state the final answer as (A), (B), etc."
)


class CRFrontend(Protocol):
    """LLM or mock frontend for CR paper pipeline."""

    def sample_completions(
        self, question: str, candidates: list[str], n_samples: int, seed: int = 0
    ) -> list[str]: ...

    def generate_answer_direct(
        self, question: str, candidates: list[str], seed: int = 0
    ) -> tuple[int | None, str]: ...

    def generate_answer_with_reasons(
        self,
        question: str,
        candidates: list[str],
        reasons: list[tuple[str, float]],
        seed: int = 0,
    ) -> tuple[int | None, str]: ...


@dataclass
class MockCRFrontend:
    """CPU smoke frontend: simulates LLM sampling from candidate logits."""

    logits: np.ndarray
    seed: int = 0

    def sample_completions(
        self, question: str, candidates: list[str], n_samples: int, seed: int = 0
    ) -> list[str]:
        rng = np.random.default_rng(seed)
        logits = np.asarray(self.logits, dtype=float).ravel()
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        out: list[str] = []
        for _ in range(n_samples):
            idx = int(rng.choice(len(candidates), p=probs))
            letter = chr(65 + idx)
            out.append(
                f"Step by step: option ({letter}) {candidates[idx]} seems correct. "
                f"Final answer: ({letter})"
            )
        return out

    def generate_answer_direct(
        self, question: str, candidates: list[str], seed: int = 0
    ) -> tuple[int | None, str]:
        idx = int(np.argmax(self.logits))
        letter = chr(65 + idx)
        return idx, f"Final answer: ({letter})"

    def generate_answer_with_reasons(
        self,
        question: str,
        candidates: list[str],
        reasons: list[tuple[str, float]],
        seed: int = 0,
    ) -> tuple[int | None, str]:
        from qihc.orchestrator.cr_protocol import extract_answer_index

        votes: dict[int, float] = {}
        for text, w in reasons:
            idx = extract_answer_index(text, candidates)
            if idx is not None:
                votes[idx] = votes.get(idx, 0.0) + max(w, 1e-6)
        if votes:
            best = max(votes.items(), key=lambda kv: kv[1])[0]
            return best, f"Based on selected reasons, answer ({chr(65 + best)})"
        return self.generate_answer_direct(question, candidates, seed)


@dataclass
class CRPipelineResult:
    task_id: str
    mode: str
    correct: bool
    feasible: bool
    exact_match: bool
    gold_hit: bool
    pred_indices: list[int]
    gold_indices: list[int]
    n_samples: int
    n_distinct_reasons: int = 0
    n_selected_reasons: int = 0
    pbit_steps: int = 0
    llm_calls: int = 0
    free_energy_trace: list[float] = field(default_factory=list)


def majority_vote_index(samples: list[CRReasonSample], n_candidates: int) -> int | None:
    counts = np.zeros(n_candidates, dtype=float)
    for s in samples:
        if s.answer_index is not None and 0 <= s.answer_index < n_candidates:
            counts[s.answer_index] += 1.0
    if counts.sum() <= 0:
        return None
    return int(np.argmax(counts))


def _reason_weights(counts: np.ndarray, params: CRParams) -> np.ndarray:
    n = counts.size
    total = float(counts.sum()) + 1e-9
    freq = counts / total
    return params.mu * (freq - 0.5 / max(n, 1))


def select_reasons_via_qubo(
    distinct: list[CRReasonSample],
    mapping: list[int],
    params: CRParams,
    backend,
    mode: Literal["linear", "quadratic", "random"],
    rng: np.random.Generator,
) -> list[tuple[str, float]]:
    n = len(distinct)
    if n == 0:
        return []

    counts = np.zeros(n, dtype=float)
    for d_idx in mapping:
        counts[int(d_idx)] += 1.0

    kappa = min(int(params.kappa), n)
    if mode == "random":
        sel = rng.choice(n, size=kappa, replace=False)
        mask = np.zeros(n, dtype=bool)
        mask[sel] = True
    elif mode == "linear":
        top = np.argsort(-counts)[:kappa]
        mask = np.zeros(n, dtype=bool)
        mask[top] = True
    else:
        co = build_reason_cooccurrence(distinct, mapping)
        weight, field = build_cr_qubo(n, counts, co, params, mode="quadratic")
        mask, _energy, _elapsed = backend.solve(weight, field)
        mask = refine_mask_to_top_k(counts, mask, kappa)

    w_vals = _reason_weights(counts, params)
    selected: list[tuple[str, float]] = []
    for i in range(n):
        if mask[i]:
            selected.append((distinct[i].text, float(w_vals[i])))
    selected.sort(key=lambda x: (-x[1], x[0]))
    return selected


def _mask_from_indices(indices: list[int], n: int) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    for i in indices:
        if 0 <= int(i) < n:
            mask[int(i)] = True
    return mask


def _indices_from_single(idx: int | None, top_k: int, n: int) -> list[int]:
    if idx is None:
        return []
    if top_k <= 1:
        return [int(idx)] if 0 <= idx < n else []
    mask = np.zeros(n, dtype=bool)
    mask[int(idx)] = True
    return list(np.flatnonzero(mask))


def evaluate_cr_task(
    task: BBHTask,
    mode: CRPaperMode,
    frontend: CRFrontend,
    backend,
    params: CRParams,
    config,
    n_samples: int,
    seed: int,
) -> CRPipelineResult:
    """Run one CR-paper mode on a BBHTask."""
    from qihc.orchestrator.free_energy import compute_free_energy
    from qihc.orchestrator.vci_scheduler import VCIOrchestrator

    problem = task.to_subset_problem(seed=seed)
    n_cand = len(task.candidates)
    gold = [int(i) for i in task.gold_indices]
    rng = np.random.default_rng(seed)

    llm_calls = 0
    pbit_steps = 0
    n_distinct = 0
    n_selected = 0
    f_trace: list[float] = []
    pred_indices: list[int] = []

    if mode == "zeroshot":
        pred_idx, _text = frontend.generate_answer_direct(task.text, task.candidates, seed=seed)
        pred_indices = _indices_from_single(pred_idx, task.top_k, n_cand)
        llm_calls = 1

    elif mode == "linear":
        completions = frontend.sample_completions(
            task.text, task.candidates, n_samples=n_samples, seed=seed
        )
        samples = samples_from_completions(completions, task.candidates)
        llm_calls = n_samples
        pred_idx = majority_vote_index(samples, n_cand)
        pred_indices = _indices_from_single(pred_idx, task.top_k, n_cand)

    elif mode in ("random", "quadratic"):
        completions = frontend.sample_completions(
            task.text, task.candidates, n_samples=n_samples, seed=seed
        )
        samples = samples_from_completions(completions, task.candidates)
        distinct, mapping = deduplicate_reasons(samples, zeta=params.similarity_zeta)
        n_distinct = len(distinct)
        qmode = "random" if mode == "random" else "quadratic"
        reasons = select_reasons_via_qubo(distinct, mapping, params, backend, qmode, rng)
        n_selected = len(reasons)
        if mode == "quadratic":
            pbit_steps = int(getattr(backend.config, "sampling_steps", 0))
        pred_idx, _text = frontend.generate_answer_with_reasons(
            task.text, task.candidates, reasons, seed=seed + 1
        )
        pred_indices = _indices_from_single(pred_idx, task.top_k, n_cand)
        llm_calls = n_samples + 1

    elif mode in ("vci-1", "vci-2"):
        completions = frontend.sample_completions(
            task.text, task.candidates, n_samples=n_samples, seed=seed
        )
        samples = samples_from_completions(completions, task.candidates)
        llm_calls = n_samples
        cr_logits = cr_logits_from_samples(samples, n_cand, params, mode="quadratic")
        prob = SubsetProblem(
            text=problem.text,
            logits=cr_logits,
            top_k=problem.top_k,
            exclusion_pairs=problem.exclusion_pairs,
            metadata=dict(problem.metadata),
        )
        orch = VCIOrchestrator(config)
        res = orch.solve_subset(prob, mode=mode)  # type: ignore[arg-type]
        pred_indices = list(np.flatnonzero(res.final_mask))
        f_trace = [r.free_energy.total for r in res.rounds]
        pbit_steps = int(config.sampling_steps * res.n_rounds)

    else:
        raise ValueError(f"Unknown CR paper mode: {mode}")

    mask = _mask_from_indices(pred_indices, n_cand)
    if mode in ("vci-1", "vci-2") and not pred_indices:
        pred_indices = list(np.flatnonzero(mask))

    feasible = is_feasible(mask, task.top_k, task.exclusion_pairs)
    pred_set = set(pred_indices) if pred_indices else set(np.flatnonzero(mask))
    gold_set = set(gold)
    exact = pred_set == gold_set
    if mode in ("zeroshot", "random", "linear", "quadratic") and task.top_k > 1 and len(pred_set) == 1:
        correct = bool(pred_set & gold_set)
        gold_hit = correct
    else:
        correct = exact
        gold_hit = exact

    if not f_trace and mode not in ("vci-1", "vci-2"):
        fe = compute_free_energy(
            np.asarray(problem.logits, dtype=float),
            np.asarray(problem.logits, dtype=float),
            mask,
            exclusion_pairs=task.exclusion_pairs,
        )
        f_trace = [fe.total]

    return CRPipelineResult(
        task_id=task.task_id,
        mode=mode,
        correct=correct,
        feasible=feasible,
        exact_match=exact,
        gold_hit=bool(pred_set & gold_set) if pred_set else False,
        pred_indices=pred_indices or list(np.flatnonzero(mask)),
        gold_indices=gold,
        n_samples=n_samples if mode != "zeroshot" else 0,
        n_distinct_reasons=n_distinct,
        n_selected_reasons=n_selected,
        pbit_steps=pbit_steps,
        llm_calls=llm_calls,
        free_energy_trace=f_trace,
    )


def run_cr_paper_benchmark(
    tasks: list[BBHTask],
    modes: list[CRPaperMode],
    budget_steps: int,
    n_samples: int,
    seed: int,
    use_llm: bool,
    model_name: str,
) -> dict:
    """Benchmark CR paper modes on a task list."""
    from qihc.orchestrator.backend import PBitBackend
    from qihc.orchestrator.llm_sampler import LLMSampler, LLMSamplerConfig
    from qihc.orchestrator.vci_scheduler import VCIConfig

    cfg = VCIConfig.tier_a(sampling_steps=budget_steps, seed=seed)
    backend = PBitBackend(cfg)
    params = CRParams()

    shared_sampler: LLMSampler | None = None
    if use_llm:
        shared_sampler = LLMSampler(
            LLMSamplerConfig(
                model_name=model_name,
                system_prompt=CR_SYSTEM_PROMPT,
                batch_size=8,
            )
        )

    all_results: dict[str, list[dict]] = {m: [] for m in modes}
    llm_stats = None

    for ti, task in enumerate(tasks):
        if use_llm and shared_sampler is not None:
            frontend: CRFrontend = shared_sampler
        else:
            problem = task.to_subset_problem(seed=seed + ti)
            frontend = MockCRFrontend(logits=problem.logits, seed=seed + ti)

        for mode in modes:
            r = evaluate_cr_task(
                task,
                mode,
                frontend,
                backend,
                params,
                cfg,
                n_samples=n_samples,
                seed=seed + ti * 31,
            )
            all_results[mode].append(
                {
                    "task_id": r.task_id,
                    "correct": bool(r.correct),
                    "gold_hit": bool(r.gold_hit),
                    "feasible": bool(r.feasible),
                    "exact_match": bool(r.exact_match),
                    "top_k": int(task.top_k),
                    "pred_indices": [int(x) for x in r.pred_indices],
                    "gold_indices": [int(x) for x in r.gold_indices],
                    "n_samples": int(r.n_samples),
                    "n_distinct_reasons": int(r.n_distinct_reasons),
                    "n_selected_reasons": int(r.n_selected_reasons),
                    "pbit_steps": int(r.pbit_steps),
                    "llm_calls": int(r.llm_calls),
                    "F_trace": [float(x) for x in r.free_energy_trace],
                }
            )

        if (ti + 1) % 10 == 0:
            print(f"  [{ti + 1}/{len(tasks)}] tasks done")

    if shared_sampler is not None:
        llm_stats = {
            "n_completions": shared_sampler.stats.n_completions,
            "n_prompt_tokens": shared_sampler.stats.n_prompt_tokens,
            "n_completion_tokens": shared_sampler.stats.n_completion_tokens,
            "wall_time_s": round(shared_sampler.stats.wall_time_s, 2),
        }

    mean_top_k = float(np.mean([t.top_k for t in tasks])) if tasks else 1.0
    is_constrained_multiselect = mean_top_k > 1.5

    summary: dict[str, dict] = {}
    for mode, rows in all_results.items():
        acc = float(np.mean([r["correct"] for r in rows]))
        exact = float(np.mean([r["exact_match"] for r in rows]))
        gold_hit = float(np.mean([r["gold_hit"] for r in rows]))
        feas = float(np.mean([r["feasible"] for r in rows]))
        summary[mode] = {
            "accuracy": acc,
            "exact_match_rate": exact,
            "gold_hit_rate": gold_hit,
            "feasible_rate": feas,
            "mean_pbit_steps": float(np.mean([r["pbit_steps"] for r in rows])),
            "mean_llm_calls": float(np.mean([r["llm_calls"] for r in rows])),
            "n_tasks": len(rows),
        }

    zs = summary.get("zeroshot", {})
    zs_acc = zs.get("accuracy", 0.0)
    zs_exact = zs.get("exact_match_rate", 0.0)
    zs_feas = zs.get("feasible_rate", 0.0)
    for mode in summary:
        summary[mode]["gain_over_zeroshot"] = float(summary[mode]["accuracy"] - zs_acc)
        summary[mode]["gain_over_zeroshot_exact"] = float(
            summary[mode]["exact_match_rate"] - zs_exact
        )
        summary[mode]["feasible_gain_over_zeroshot"] = float(
            summary[mode]["feasible_rate"] - zs_feas
        )

    track = "constrained_multiselect" if is_constrained_multiselect else "paper_single_answer"
    primary_metric = "feasible_rate" if is_constrained_multiselect else "accuracy"

    return {
        "summary": summary,
        "per_task": all_results,
        "llm_stats": llm_stats,
        "n_samples": n_samples,
        "budget_steps": budget_steps,
        "protocol": "cr_paper_arxiv_2407_00071",
        "track": track,
        "mean_top_k": mean_top_k,
        "primary_metric": primary_metric,
        "use_llm": use_llm,
        "metric_warning": (
            "constrained_multiselect + mock LLM: CR modes output 1 answer → feas≈0; "
            "compare feasible_gain_over_zeroshot, NOT gain_over_zeroshot (gold-hit vs exact)."
            if is_constrained_multiselect and not use_llm
            else None
        ),
        "note": (
            "zeroshot=LLM T=0 direct answer; linear=majority vote; "
            "quadratic=QUBO reason select + enhanced prompt + LLM T=0; "
            "vci-1/2=CR-encoded constrained cooperative inference. "
            f"Track={track}, primary={primary_metric}."
        ),
    }
