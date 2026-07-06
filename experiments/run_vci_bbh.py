# -*- coding: utf-8 -*-
"""
Case A BBH: Greedy / VCI-1 / VCI-2 with equal p-bit step budget.

Usage:
    # bundled synthetic mini-set (default)
    python experiments/run_vci_bbh.py

    # Hugging Face BIG-Bench Hard (real BBH)
    pip install -e ".[hf]"
    python experiments/run_vci_bbh.py --source hf --limit-per-task 30

    # Hugging Face + real LLM logits (DistilGPT-2)
    pip install -e ".[hf,llm]"
    python experiments/run_vci_bbh.py --source hf --logits llm --limit 50
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from qihc.orchestrator.bbh import evaluate_prediction, load_bbh_problems  # noqa: E402
from qihc.orchestrator.bbh_hf import DEFAULT_HF_REPO  # noqa: E402
from qihc.orchestrator.bbh_llm import enrich_problems_with_llm_logits  # noqa: E402
from qihc.orchestrator.bbh_parser import DEFAULT_BBH_HF_TASKS  # noqa: E402
from qihc.orchestrator.vci_scheduler import VCIConfig, VCIOrchestrator  # noqa: E402


def run_mode(problems, mode: str, budget_steps: int, seed: int) -> dict:
    if mode == "greedy":
        steps = 0
        max_rounds = 1
    elif mode == "vci-1":
        steps = budget_steps
        max_rounds = 1
    elif mode == "vci-2":
        steps = max(budget_steps // 2, 50)
        max_rounds = 2
    else:
        raise ValueError(mode)

    cfg = VCIConfig.tier_a(sampling_steps=steps, max_rounds=max_rounds, seed=seed)
    orch = VCIOrchestrator(cfg)

    feasible, exact, jaccard, times, rounds = [], [], [], [], []
    for p in problems:
        res = orch.solve_subset(p, mode=mode)  # type: ignore[arg-type]
        ev = evaluate_prediction(p, res.final_mask)
        feasible.append(ev["feasible"])
        exact.append(ev.get("exact_match", False))
        jaccard.append(ev.get("jaccard", 0.0))
        times.append(res.total_elapsed_s)
        rounds.append(res.n_rounds)

    pbit_steps_used = steps * float(np.mean(rounds)) if mode != "greedy" else 0.0
    return {
        "mode": mode,
        "feasible_rate": float(np.mean(feasible)),
        "exact_match_rate": float(np.mean(exact)),
        "mean_jaccard": float(np.mean(jaccard)),
        "mean_time_s": float(np.mean(times)),
        "mean_rounds": float(np.mean(rounds)),
        "steps_per_round": steps,
        "mean_pbit_steps": pbit_steps_used,
    }


def plot_bbh_results(summary: dict, out_path: str, title: str) -> None:
    modes = list(summary.keys())
    x = np.arange(len(modes))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=120)
    ax.bar(x - width, [summary[m]["feasible_rate"] for m in modes], width, label="Feasible", color="#72b7b2")
    ax.bar(x, [summary[m]["exact_match_rate"] for m in modes], width, label="Exact match", color="#4c78a8")
    ax.bar(x + width, [summary[m]["mean_jaccard"] for m in modes], width, label="Jaccard", color="#e45756")
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate / score")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="VCI on BBH (bundled or HuggingFace)")
    parser.add_argument("--source", choices=["bundled", "hf"], default="bundled")
    parser.add_argument("--budget-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--limit-per-task", type=int, default=None, help="HF only: cap per subtask")
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    parser.add_argument(
        "--hf-tasks",
        nargs="*",
        default=None,
        help=f"HF subtasks (default: {len(DEFAULT_BBH_HF_TASKS)} reasoning tasks)",
    )
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--logits",
        choices=["pseudo", "llm"],
        default="pseudo",
        help="pseudo = hash+gold bias; llm = causal LM answer scoring",
    )
    parser.add_argument("--model-name", default="distilgpt2", help="LM for --logits llm")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Default: experiments/outputs/vci_bbh[_hf][_llm]",
    )
    args = parser.parse_args()

    out_dir = args.output_dir
    if out_dir is None:
        sub = "vci_bbh"
        if args.source == "hf":
            sub = "vci_bbh_hf"
        if args.logits == "llm":
            sub += "_llm"
        out_dir = os.path.join("experiments", "outputs", sub)
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(REPO_ROOT, out_dir)

    problems = load_bbh_problems(
        source=args.source,
        seed=args.seed,
        limit=args.limit,
        hf_repo=args.hf_repo,
        hf_tasks=args.hf_tasks,
        limit_per_task=args.limit_per_task,
        use_cache=not args.no_cache,
        refresh_cache=args.refresh_cache,
    )

    if args.logits == "llm":
        print(f"Scoring {len(problems)} problems with {args.model_name} ...")
        problems = enrich_problems_with_llm_logits(
            problems, model_name=args.model_name
        )

    modes = ["greedy", "vci-1", "vci-2"]
    summary = {m: run_mode(problems, m, args.budget_steps, args.seed) for m in modes}

    payload = {
        "source": args.source,
        "logits": args.logits,
        "model_name": args.model_name if args.logits == "llm" else None,
        "hf_repo": args.hf_repo if args.source == "hf" else None,
        "hf_tasks": args.hf_tasks or (DEFAULT_BBH_HF_TASKS if args.source == "hf" else None),
        "n_tasks": len(problems),
        "budget_steps": args.budget_steps,
        "summary": summary,
        "note": "vci-2 uses budget_steps/2 per round × up to 2 rounds ≈ equal budget",
    }

    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Saved: {json_path}")

    label = "HF BBH" if args.source == "hf" else "bundled BBH"
    if args.logits == "llm":
        label += f" + {args.model_name}"
    print(f"\n=== {label} (n={len(problems)}, budget={args.budget_steps}) ===")
    for mode, s in summary.items():
        print(
            f"  {mode:8s} feas={s['feasible_rate']:.2%}  exact={s['exact_match_rate']:.2%}  "
            f"jacc={s['mean_jaccard']:.3f}  pbit_steps≈{s['mean_pbit_steps']:.0f}"
        )

    title = f"{label}: same p-bit budget comparison"
    plot_bbh_results(summary, os.path.join(out_dir, "bbh_comparison.png"), title)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
