# -*- coding: utf-8 -*-
"""
Dual-axis evidence table: bundled constraint set (main) + optional HF/LLM arm.

Produces Table-1 style metrics: feasible rate, exact match, mean F, p-bit steps, LLM calls.

Usage:
    python experiments/nsfc_evidence/run_dual_evidence.py
    python experiments/nsfc_evidence/run_dual_evidence.py --output-dir experiments/outputs/nsfc_evidence/dual
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

from qihc.orchestrator.bbh import evaluate_prediction, load_bbh_problems  # noqa: E402
from qihc.orchestrator.bbh_llm import enrich_problems_with_llm_logits  # noqa: E402
from qihc.orchestrator.vci_scheduler import VCIConfig, VCIOrchestrator  # noqa: E402


def run_mode(
    problems,
    mode: str,
    budget_steps: int,
    seed: int,
) -> dict[str, Any]:
    if mode == "greedy":
        steps, max_rounds = 0, 1
    elif mode == "vci-1":
        steps, max_rounds = budget_steps, 1
    elif mode == "vci-2":
        steps, max_rounds = max(budget_steps // 2, 50), 2
    else:
        raise ValueError(mode)

    cfg = VCIConfig.tier_a(sampling_steps=steps, max_rounds=max_rounds, seed=seed)
    orch = VCIOrchestrator(cfg)

    feasible, exact, jaccard, f_vals, rounds, times = [], [], [], [], [], []
    per_task: list[dict] = []

    for p in problems:
        res = orch.solve_subset(p, mode=mode)  # type: ignore[arg-type]
        ev = evaluate_prediction(p, res.final_mask)
        feasible.append(ev["feasible"])
        exact.append(ev.get("exact_match", False))
        jaccard.append(ev.get("jaccard", 0.0))
        f_vals.append(res.final_free_energy)
        rounds.append(res.n_rounds)
        times.append(res.total_elapsed_s)
        per_task.append(
            {
                "task_id": p.metadata.get("task_id"),
                "feasible": ev["feasible"],
                "exact_match": ev.get("exact_match"),
                "jaccard": ev.get("jaccard"),
                "F_final": res.final_free_energy,
                "F_trace": [r.free_energy.total for r in res.rounds],
                "n_rounds": res.n_rounds,
                "pbit_steps": steps * res.n_rounds if mode != "greedy" else 0,
            }
        )

    llm_calls = 1
    mean_pbit = steps * float(np.mean(rounds)) if mode != "greedy" else 0.0
    return {
        "mode": mode,
        "feasible_rate": float(np.mean(feasible)),
        "exact_match_rate": float(np.mean(exact)),
        "mean_jaccard": float(np.mean(jaccard)),
        "mean_F": float(np.mean(f_vals)),
        "mean_time_s": float(np.mean(times)),
        "mean_rounds": float(np.mean(rounds)),
        "steps_per_round": steps,
        "mean_pbit_steps": mean_pbit,
        "llm_calls": llm_calls,
        "per_task": per_task,
    }


def plot_dual_axis(summary: dict, out_path: str, title: str) -> None:
    modes = list(summary.keys())
    x = np.arange(len(modes))
    w = 0.2
    fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
    ax.bar(x - 1.5 * w, [summary[m]["feasible_rate"] for m in modes], w, label="Feasible", color="#72b7b2")
    ax.bar(x - 0.5 * w, [summary[m]["exact_match_rate"] for m in modes], w, label="Exact match", color="#4c78a8")
    ax.bar(x + 0.5 * w, [summary[m]["mean_jaccard"] for m in modes], w, label="Jaccard", color="#e45756")
    f_norm = [summary[m]["mean_F"] for m in modes]
    f_min, f_max = min(f_norm), max(f_norm)
    span = f_max - f_min + 1e-9
    f_scaled = [(v - f_min) / span for v in f_norm]
    ax.bar(x + 1.5 * w, f_scaled, w, label="F (scaled)", color="#b279a2")
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Rate / scaled F")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Dual-axis evidence (bundled + optional HF)")
    parser.add_argument("--source", choices=["bundled", "hf"], default="bundled")
    parser.add_argument("--logits", choices=["pseudo", "llm"], default="pseudo")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--budget-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--limit-per-task", type=int, default=30)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    out_dir = args.output_dir or os.path.join(
        REPO_ROOT, "experiments", "outputs", "nsfc_evidence", f"dual_{args.source}"
    )
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(REPO_ROOT, out_dir)

    problems = load_bbh_problems(
        source=args.source,
        seed=args.seed,
        limit=args.limit,
        limit_per_task=args.limit_per_task if args.source == "hf" else None,
    )
    if args.logits == "llm":
        problems = enrich_problems_with_llm_logits(problems, model_name=args.model_name)

    modes = ["greedy", "vci-1", "vci-2"]
    summary = {m: run_mode(problems, m, args.budget_steps, args.seed) for m in modes}

    for m in summary:
        summary[m].pop("per_task", None)

    payload = {
        "experiment": "dual_axis_evidence",
        "source": args.source,
        "logits": args.logits,
        "model_name": args.model_name if args.logits == "llm" else None,
        "n_tasks": len(problems),
        "budget_steps": args.budget_steps,
        "same_compute_note": "vci-2: steps/2 per round × 2 rounds ≈ vci-1 total p-bit steps",
        "summary": summary,
    }

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "dual_axis.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    title = f"Dual-axis evidence ({args.source}, n={len(problems)})"
    plot_dual_axis(summary, os.path.join(out_dir, "dual_axis.png"), title)

    print(f"\n=== {title} ===")
    for mode, s in summary.items():
        print(
            f"  {mode:8s} feas={s['feasible_rate']:.2%} exact={s['exact_match_rate']:.2%} "
            f"F={s['mean_F']:.2f} pbit≈{s['mean_pbit_steps']:.0f}"
        )
    print(f"Saved: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
