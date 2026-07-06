# -*- coding: utf-8 -*-
"""
CR-protocol aligned BBH experiment (arXiv:2407.00071).

Modes: zeroshot | linear | quadratic | vci-1 | vci-2
With --use-llm: sample N completions per question (CR-style), then QUBO select.

Usage:
    # Fast mock (bundled, pseudo logits)
    python experiments/nsfc_evidence/run_cr_bbh.py

    # Server: real LLM + HF BBH
    python experiments/nsfc_evidence/run_cr_bbh.py \\
        --source hf --use-llm --model-name Qwen/Qwen2.5-7B-Instruct \\
        --n-samples 50 --limit-per-task 20 --hf-tasks logical_deduction_seven_objects
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from qihc.orchestrator.backend import PBitBackend  # noqa: E402
from qihc.orchestrator.bbh import load_bbh_tasks  # noqa: E402
from qihc.orchestrator.bbh_parser import DEFAULT_BBH_HF_TASKS  # noqa: E402
from qihc.orchestrator.cr_protocol import CRParams, CRMode, cr_logits_from_samples, evaluate_cr_on_problem  # noqa: E402
from qihc.orchestrator.llm_sampler import LLMSampler, LLMSamplerConfig  # noqa: E402
from qihc.orchestrator.reasoning import SubsetProblem  # noqa: E402
from qihc.orchestrator.vci_scheduler import VCIConfig  # noqa: E402


def _mock_samples_from_logits(logits: np.ndarray, n_samples: int, seed: int) -> list:
    from qihc.orchestrator.cr_protocol import CRReasonSample

    rng = np.random.default_rng(seed)
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    samples: list[CRReasonSample] = []
    for _ in range(n_samples):
        idx = int(rng.choice(len(logits), p=probs))
        samples.append(CRReasonSample(text=f"reason_for_{idx}", answer_index=idx))
    return samples


def run_cr_benchmark(
    tasks,
    modes: list[CRMode],
    budget_steps: int,
    n_samples: int,
    seed: int,
    use_llm: bool,
    model_name: str,
) -> dict[str, Any]:
    cfg = VCIConfig.tier_a(sampling_steps=budget_steps, seed=seed)
    backend = PBitBackend(cfg)
    params = CRParams()
    sampler = None
    if use_llm:
        sampler = LLMSampler(LLMSamplerConfig(model_name=model_name, batch_size=8))

    all_results: dict[str, list[dict]] = {m: [] for m in modes}
    llm_stats = None

    for ti, task in enumerate(tasks):
        problem = task.to_subset_problem(seed=seed + ti)
        logits = np.asarray(problem.logits, dtype=float)

        if use_llm and sampler is not None:
            samples = sampler.sample_reasons(
                task.text,
                task.candidates,
                n_samples=n_samples,
                seed=seed + ti * 17,
            )
            logits = np.asarray(
                cr_logits_from_samples(samples, len(task.candidates), params, "quadratic"),
                dtype=float,
            )
            problem = SubsetProblem(
                text=problem.text,
                logits=logits,
                top_k=problem.top_k,
                exclusion_pairs=problem.exclusion_pairs,
                metadata=dict(problem.metadata),
            )
        else:
            samples = _mock_samples_from_logits(logits, n_samples, seed + ti)

        for mode in modes:
            if mode == "zeroshot":
                r = evaluate_cr_on_problem(problem, mode, backend, None, None, params, cfg)
            elif mode == "linear":
                r = evaluate_cr_on_problem(problem, mode, backend, None, samples, params, cfg)
            else:
                r = evaluate_cr_on_problem(problem, mode, backend, None, samples, params, cfg)
            all_results[mode].append(
                {
                    "task_id": r.task_id,
                    "correct": r.correct,
                    "feasible": r.feasible,
                    "exact_match": r.exact_match,
                    "n_samples": r.n_samples,
                    "pbit_steps": r.pbit_steps,
                    "F_trace": r.free_energy_trace,
                }
            )

        if (ti + 1) % 10 == 0:
            print(f"  [{ti+1}/{len(tasks)}] tasks done")

    if sampler is not None:
        llm_stats = {
            "n_completions": sampler.stats.n_completions,
            "n_prompt_tokens": sampler.stats.n_prompt_tokens,
            "n_completion_tokens": sampler.stats.n_completion_tokens,
            "wall_time_s": round(sampler.stats.wall_time_s, 2),
        }

    summary = {}
    for mode, rows in all_results.items():
        summary[mode] = {
            "accuracy": float(np.mean([r["correct"] for r in rows])),
            "feasible_rate": float(np.mean([r["feasible"] for r in rows])),
            "mean_pbit_steps": float(np.mean([r["pbit_steps"] for r in rows])),
            "n_tasks": len(rows),
        }

    return {
        "summary": summary,
        "per_task": all_results,
        "llm_stats": llm_stats,
        "n_samples": n_samples,
        "budget_steps": budget_steps,
    }


def plot_cr_summary(summary: dict, out_path: str, title: str) -> None:
    modes = list(summary.keys())
    acc = [summary[m]["accuracy"] for m in modes]
    feas = [summary[m]["feasible_rate"] for m in modes]
    x = np.arange(len(modes))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=120)
    ax.bar(x - w / 2, acc, w, label="Accuracy / exact match", color="#4c78a8")
    ax.bar(x + w / 2, feas, w, label="Feasible rate", color="#72b7b2")
    ax.set_xticks(x)
    ax.set_xticklabels(modes, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="CR-protocol BBH benchmark")
    parser.add_argument("--source", choices=["bundled", "hf"], default="bundled")
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--n-samples", type=int, default=50, help="CR reason samples per question")
    parser.add_argument("--budget-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--limit-per-task", type=int, default=25)
    parser.add_argument("--hf-tasks", nargs="*", default=None)
    parser.add_argument("--modes", nargs="*", default=["zeroshot", "linear", "quadratic", "vci-1", "vci-2"])
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    out_dir = args.output_dir or os.path.join(REPO_ROOT, "experiments", "outputs", "nsfc_evidence", "cr_protocol")
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(REPO_ROOT, out_dir)

    tasks = load_bbh_tasks(
        source=args.source,
        hf_tasks=args.hf_tasks or (DEFAULT_BBH_HF_TASKS[:5] if args.source == "hf" else None),
        limit_per_task=args.limit_per_task if args.source == "hf" else None,
    )
    if args.limit:
        tasks = tasks[: args.limit]

    print(f"CR benchmark: {len(tasks)} tasks, n_samples={args.n_samples}, use_llm={args.use_llm}")
    data = run_cr_benchmark(
        tasks,
        modes=args.modes,  # type: ignore[arg-type]
        budget_steps=args.budget_steps,
        n_samples=args.n_samples,
        seed=args.seed,
        use_llm=args.use_llm,
        model_name=args.model_name,
    )

    payload = {
        "experiment": "cr_protocol_bbh",
        "source": args.source,
        "use_llm": args.use_llm,
        "model_name": args.model_name if args.use_llm else None,
        "modes": args.modes,
        **data,
        "reference": "Combinatorial Reasoning arXiv:2407.00071",
    }

    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "cr_protocol.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    plot_cr_summary(
        data["summary"],
        os.path.join(out_dir, "cr_protocol.png"),
        f"CR protocol ({args.source}, n={len(tasks)})",
    )

    print(f"\n=== CR protocol summary (n={len(tasks)}) ===")
    for mode, s in data["summary"].items():
        print(f"  {mode:12s} acc={s['accuracy']:.2%} feas={s['feasible_rate']:.2%}")
    if data.get("llm_stats"):
        print(f"  LLM stats: {data['llm_stats']}")
    print(f"Saved: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
