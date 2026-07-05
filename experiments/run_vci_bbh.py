# -*- coding: utf-8 -*-
"""
Case A BBH mini-set: Greedy / VCI-1 / VCI-2 with equal p-bit step budget.

Usage:
    python experiments/run_vci_bbh.py
    python experiments/run_vci_bbh.py --budget-steps 300 --seed 0
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
from qihc.orchestrator.vci_scheduler import VCIConfig, VCIOrchestrator  # noqa: E402


def run_mode(
    problems,
    mode: str,
    budget_steps: int,
    seed: int,
) -> dict:
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


def plot_bbh_results(summary: dict, out_path: str) -> None:
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
    ax.set_title("BBH mini-set: same p-bit budget comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="VCI on BBH-style mini-set")
    parser.add_argument("--budget-steps", type=int, default=250, help="Total p-bit steps budget")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        default=os.path.join("experiments", "outputs", "vci_bbh"),
    )
    args = parser.parse_args()

    out_dir = args.output_dir
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(REPO_ROOT, out_dir)

    problems = load_bbh_problems(seed=args.seed, limit=args.limit)
    modes = ["greedy", "vci-1", "vci-2"]
    summary = {
        m: run_mode(problems, m, args.budget_steps, args.seed) for m in modes
    }

    payload = {
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

    print(f"\n=== BBH mini-set (n={len(problems)}, budget={args.budget_steps}) ===")
    for mode, s in summary.items():
        print(
            f"  {mode:8s} feas={s['feasible_rate']:.2%}  exact={s['exact_match_rate']:.2%}  "
            f"jacc={s['mean_jaccard']:.3f}  pbit_steps≈{s['mean_pbit_steps']:.0f}"
        )

    plot_bbh_results(summary, os.path.join(out_dir, "bbh_comparison.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
