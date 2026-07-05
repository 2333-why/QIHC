# -*- coding: utf-8 -*-
"""
Case A toy experiment: Greedy vs VCI-1 (CR) vs VCI-2.

Usage (from QIHC repo root):
    python experiments/run_vci_reasoning.py
    python experiments/run_vci_reasoning.py --problems 48 --steps 300 --seed 1
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

from qihc.orchestrator import (  # noqa: E402
    VCIConfig,
    VCIOrchestrator,
    demo_problem,
    generate_toy_problems,
)


def plot_free_energy_trajectories(out_path: str, sampling_steps: int = 200) -> None:
    plt.figure(figsize=(8, 4.5), dpi=120)
    problem = demo_problem()
    cfg = VCIConfig.tier_a(sampling_steps=sampling_steps, seed=0)
    orch = VCIOrchestrator(cfg)

    # Forced 2-round trace: greedy (violating) → q-refine → p-bit s-step
    trace = orch.trace_refine_demo(problem)
    rounds = [r.round_index for r in trace]
    f_trace = [r.free_energy.total for r in trace]
    viol = [r.free_energy.violations for r in trace]
    plt.plot(rounds, f_trace, "o-", color="#4c78a8", label="VCI-2 trace (greedy→refine→p-bit)", linewidth=2)
    for r, f, v in zip(rounds, f_trace, viol):
        plt.annotate(
            f"viol={v}, feas={trace[r].feasible}",
            (r, f),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
            color="#4c78a8",
        )

    # CR limit: single s-step without q-refine
    vci1 = orch.solve_subset(problem, mode="vci-1")
    plt.scatter(
        [0],
        [vci1.rounds[0].free_energy.total],
        color="#e45756",
        s=80,
        zorder=5,
        label=f"VCI-1 (CR) feas={vci1.final_feasible}",
    )

    plt.xlabel("VCI round")
    plt.ylabel("Free energy proxy F(q, s)")
    plt.title("Demo problem: mutual exclusion (2,5) — F descent & IF repair")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_comparison_bars(summary: dict, out_path: str) -> None:
    modes = list(summary.keys())
    feasible = [summary[m]["feasible_rate"] for m in modes]
    semantic = [summary[m]["mean_semantic_score"] for m in modes]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), dpi=120)
    x = np.arange(len(modes))
    axes[0].bar(x, feasible, color=["#72b7b2", "#e45756", "#4c78a8"][: len(modes)])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(modes)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Feasible rate")
    axes[0].set_title("Constraint satisfaction")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x, semantic, color=["#72b7b2", "#e45756", "#4c78a8"][: len(modes)])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(modes)
    axes[1].set_ylabel("Mean semantic score")
    axes[1].set_title("Preference quality")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="VCI Case A toy benchmark")
    parser.add_argument("--problems", type=int, default=32)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default=os.path.join("experiments", "outputs", "vci_reasoning"),
    )
    args = parser.parse_args()

    out_dir = args.output_dir
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(REPO_ROOT, out_dir)

    cfg = VCIConfig.tier_a(sampling_steps=args.steps, seed=args.seed)
    orch = VCIOrchestrator(cfg)
    problems = generate_toy_problems(
        n_problems=args.problems,
        n_candidates=6,
        top_k=3,
        n_exclusion_pairs=1,
        seed=args.seed,
    )
    modes = ["greedy", "vci-1", "vci-2"]
    summary = orch.compare_modes(problems, modes=modes)

    demo = demo_problem()
    demo_runs = {m: orch.solve_subset(demo, mode=m) for m in ["vci-1", "vci-2"]}
    demo_trace = orch.trace_refine_demo(demo)

    payload = {
        "summary": summary,
        "demo": {
            mode: {
                "feasible": res.final_feasible,
                "rounds": res.n_rounds,
                "semantic_score": res.final_semantic_score,
                "free_energy_trace": [r.free_energy.total for r in res.rounds],
                "violations_trace": [r.free_energy.violations for r in res.rounds],
                "final_indices": [int(i) for i in np.flatnonzero(res.final_mask)],
            }
            for mode, res in demo_runs.items()
        },
        "demo_trace": {
            "rounds": len(demo_trace),
            "free_energy_trace": [r.free_energy.total for r in demo_trace],
            "violations_trace": [r.free_energy.violations for r in demo_trace],
            "feasible_trace": [r.feasible for r in demo_trace],
        },
        "config": cfg.to_summary(),
    }

    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Saved: {metrics_path}")

    print("\n=== VCI Case A summary ===")
    for mode, stats in summary.items():
        print(
            f"  {mode:8s} feasible={stats['feasible_rate']:.2%}  "
            f"semantic={stats['mean_semantic_score']:.3f}  "
            f"rounds={stats['mean_rounds']:.2f}  "
            f"time={stats['mean_time_s']:.3f}s"
        )

    plot_comparison_bars(summary, os.path.join(out_dir, "comparison_bars.png"))
    plot_free_energy_trajectories(
        os.path.join(out_dir, "free_energy_demo.png"),
        sampling_steps=args.steps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
