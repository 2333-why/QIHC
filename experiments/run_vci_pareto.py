# -*- coding: utf-8 -*-
"""
Pareto frontier: feasible rate / exact match / F(q,s) vs p-bit budget.

Uses bundled BBH (mutual-exclusion constraints) where VCI gains are visible.

Usage:
    python experiments/run_vci_pareto.py
    python experiments/run_vci_pareto.py --budgets 50 100 150 200 300
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


def run_point(problems, mode: str, budget_steps: int, seed: int) -> dict:
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

    feasible, exact, times, f_vals = [], [], [], []
    for p in problems:
        res = orch.solve_subset(p, mode=mode)  # type: ignore[arg-type]
        ev = evaluate_prediction(p, res.final_mask)
        feasible.append(ev["feasible"])
        exact.append(ev.get("exact_match", False))
        times.append(res.total_elapsed_s)
        f_vals.append(res.final_free_energy)

    mean_rounds = 1.0 if mode == "greedy" else (1.0 if mode == "vci-1" else 2.0)
    pbit_steps = steps * mean_rounds if mode != "greedy" else 0.0

    return {
        "mode": mode,
        "budget_steps": budget_steps,
        "feasible_rate": float(np.mean(feasible)),
        "exact_match_rate": float(np.mean(exact)),
        "mean_time_s": float(np.mean(times)),
        "mean_free_energy": float(np.mean(f_vals)),
        "mean_pbit_steps": pbit_steps,
    }


def plot_pareto(rows: list[dict], out_path: str) -> None:
    colors = {"greedy": "#e45756", "vci-1": "#f58518", "vci-2": "#4c78a8"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=120)

    for mode in ("greedy", "vci-1", "vci-2"):
        pts = [r for r in rows if r["mode"] == mode]
        pts.sort(key=lambda r: r["mean_pbit_steps"])
        x = [r["mean_pbit_steps"] for r in pts]
        axes[0].plot(x, [r["feasible_rate"] for r in pts], "o-", label=mode, color=colors[mode])
        axes[1].plot(x, [r["exact_match_rate"] for r in pts], "o-", label=mode, color=colors[mode])
        axes[2].plot(x, [r["mean_free_energy"] for r in pts], "o-", label=mode, color=colors[mode])

    for ax, ylab in zip(
        axes,
        ["Feasible rate", "Exact match rate", "Mean F(q,s)"],
    ):
        ax.set_xlabel("Mean p-bit steps")
        ax.set_ylabel(ylab)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    axes[0].set_title("Feasible vs budget")
    axes[1].set_title("Accuracy vs budget")
    axes[2].set_title("Free energy vs budget")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_tradeoff_scatter(rows: list[dict], out_path: str) -> None:
    """Quality–cost scatter (Pareto-style)."""
    colors = {"greedy": "#e45756", "vci-1": "#f58518", "vci-2": "#4c78a8"}
    markers = {"greedy": "x", "vci-1": "s", "vci-2": "o"}

    plt.figure(figsize=(6.5, 5), dpi=120)
    for r in rows:
        mode = r["mode"]
        score = 0.5 * r["feasible_rate"] + 0.5 * r["exact_match_rate"]
        plt.scatter(
            r["mean_time_s"] * 1000,
            score,
            c=colors[mode],
            marker=markers[mode],
            s=60,
            alpha=0.85,
        )
    for mode, c in colors.items():
        plt.scatter([], [], c=c, marker=markers[mode], label=mode, s=60)
    plt.xlabel("Mean latency (ms)")
    plt.ylabel("Combined score (0.5·feas + 0.5·exact)")
    plt.title("Pareto probe: quality vs latency (bundled BBH)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="VCI Pareto frontier (bundled BBH)")
    parser.add_argument("--source", choices=["bundled"], default="bundled")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=[50, 100, 150, 200, 300],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    out_dir = args.output_dir or os.path.join("experiments", "outputs", "vci_pareto")
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(REPO_ROOT, out_dir)

    problems = load_bbh_problems(source=args.source, seed=args.seed, limit=args.limit)
    rows: list[dict] = []
    for budget in args.budgets:
        for mode in ("greedy", "vci-1", "vci-2"):
            rows.append(run_point(problems, mode, budget, args.seed))
            r = rows[-1]
            print(
                f"  budget={budget:3d} {mode:6s}  "
                f"feas={r['feasible_rate']:.2%} exact={r['exact_match_rate']:.2%}  "
                f"F={r['mean_free_energy']:.3f}  steps={r['mean_pbit_steps']:.0f}"
            )

    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "pareto.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source": args.source,
                "n_problems": len(problems),
                "budgets": args.budgets,
                "rows": rows,
            },
            f,
            indent=2,
        )
    print(f"Saved: {json_path}")

    plot_pareto(rows, os.path.join(out_dir, "pareto_curves.png"))
    plot_tradeoff_scatter(rows, os.path.join(out_dir, "pareto_tradeoff.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
