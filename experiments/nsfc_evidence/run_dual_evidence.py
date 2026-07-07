# -*- coding: utf-8 -*-
"""
Dual-axis evidence: CR paper baselines + VCI on bundled / HF.

Modes: zeroshot | linear | quadratic | vci-1 | vci-2  (NOT logits-greedy)

Usage:
    python experiments/nsfc_evidence/run_dual_evidence.py
    python experiments/nsfc_evidence/run_dual_evidence.py --source hf --use-llm
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments.nsfc_evidence.common import (  # noqa: E402
    run_cr_paper_modes,
)
from qihc.orchestrator.bbh import load_bbh_problems  # noqa: E402
from qihc.orchestrator.bbh_llm import enrich_problems_with_llm_logits  # noqa: E402


def plot_dual_axis(summary: dict, out_path: str, title: str) -> None:
    modes = list(summary.keys())
    x = range(len(modes))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=120)
    ax.bar(
        [i - w / 2 for i in x],
        [summary[m]["feasible_rate"] for m in modes],
        w,
        label="Feasible",
        color="#72b7b2",
    )
    ax.bar(
        [i + w / 2 for i in x],
        [summary[m].get("accuracy", summary[m].get("exact_match_rate", 0)) for m in modes],
        w,
        label="Accuracy",
        color="#4c78a8",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(modes, rotation=15, ha="right")
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
    parser = argparse.ArgumentParser(description="Dual-axis CR paper + VCI evidence")
    parser.add_argument("--source", choices=["bundled", "hf"], default="bundled")
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--n-samples", type=int, default=50)
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
    if args.use_llm:
        problems = enrich_problems_with_llm_logits(problems, model_name=args.model_name)

    modes = ["zeroshot", "linear", "quadratic", "vci-1", "vci-2"]
    summary = run_cr_paper_modes(
        problems,
        modes,
        args.budget_steps,
        args.n_samples,
        args.seed,
        use_llm=args.use_llm,
        model_name=args.model_name,
    )

    payload = {
        "experiment": "dual_axis_evidence",
        "source": args.source,
        "use_llm": args.use_llm,
        "model_name": args.model_name if args.use_llm else None,
        "n_tasks": len(problems),
        "budget_steps": args.budget_steps,
        "baseline": "zeroshot (CR paper LLM T=0)",
        "same_compute_note": "vci-2: steps/2 per round × 2 rounds ≈ vci-1 total p-bit steps",
        "summary": summary,
        "reference": "Combinatorial Reasoning arXiv:2407.00071",
    }

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "dual_axis.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    title = f"CR paper + VCI ({args.source}, n={len(problems)})"
    plot_dual_axis(summary, os.path.join(out_dir, "dual_axis.png"), title)

    print(f"\n=== {title} ===")
    for mode, s in summary.items():
        print(
            f"  {mode:12s} acc={s['accuracy']:.2%} feas={s['feasible_rate']:.2%} "
            f"gain={s['gain_over_zeroshot']:+.2%} pbit≈{s['mean_pbit_steps']:.0f}"
        )
    print(f"Saved: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
