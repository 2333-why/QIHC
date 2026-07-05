# -*- coding: utf-8 -*-
"""
TTS (time-to-solution) scaling benchmark on Max-Cut.

Measures median steps-to-target success rate vs graph size N.

Usage:
    python experiments/run_sampler_scaling.py
    python experiments/run_sampler_scaling.py --nodes 12 16 20 24 --trials 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from qihc import IsingModel  # noqa: E402
from qihc.ising import maxcut  # noqa: E402


def run_trial(n_nodes: int, steps: int, seed: int, sampler: str = "parallel_tempering"):
    np.random.seed(seed)
    G = nx.erdos_renyi_graph(n_nodes, 0.5, seed=seed)
    J = maxcut.max_cut_to_ising(G)
    opt_cut, _ = maxcut.brute_force_max_cut(G) if n_nodes <= 22 else (None, None)
    if opt_cut is None:
        return None

    target = 0.90 * opt_cut
    common = dict(J=J, steps=steps, T_start=10.0, T_end=0.01, k=1.0)
    model = IsingModel(size=n_nodes)

    if sampler == "gibbs":
        spins, e_trace, _ = model.gibbs_sampling_Maxcut(**common, sequential=True)
    elif sampler == "parallel_tempering":
        spins, e_trace, _ = model.parallel_tempering_Maxcut(
            **common, n_replicas=6, swap_interval=20, sequential=True
        )
    else:
        raise ValueError(sampler)

    cut = maxcut.calculate_cut_value(G, maxcut.convert_spins_to_cut(spins))
    success = cut >= target
    return {
        "n_nodes": n_nodes,
        "seed": seed,
        "opt_cut": int(opt_cut),
        "cut": int(cut),
        "ratio": float(cut / opt_cut),
        "success": bool(success),
        "steps": steps,
    }


def scaling_success_rate(
    node_list: list[int],
    steps: int,
    trials: int,
    seed: int,
    sampler: str,
) -> dict:
    results = []
    rng = np.random.default_rng(seed)
    for n in node_list:
        trial_outcomes = []
        for _ in range(trials):
            trial_seed = int(rng.integers(0, 1_000_000))
            out = run_trial(n, steps, trial_seed, sampler=sampler)
            if out is not None:
                trial_outcomes.append(out)
        if trial_outcomes:
            rate = float(np.mean([t["success"] for t in trial_outcomes]))
            mean_ratio = float(np.mean([t["ratio"] for t in trial_outcomes]))
        else:
            rate, mean_ratio = float("nan"), float("nan")
        results.append(
            {
                "n_nodes": n,
                "success_rate": rate,
                "mean_cut_ratio": mean_ratio,
                "trials": len(trial_outcomes),
            }
        )
    return {"sampler": sampler, "steps": steps, "scaling": results}


def plot_scaling(data: dict, out_path: str) -> None:
    scaling = data["scaling"]
    ns = [r["n_nodes"] for r in scaling]
    rates = [r["success_rate"] for r in scaling]
    ratios = [r["mean_cut_ratio"] for r in scaling]

    fig, ax1 = plt.subplots(figsize=(8, 4.5), dpi=120)
    ax1.plot(ns, rates, "o-", color="#4c78a8", linewidth=2, label="Success rate @95% opt")
    ax1.set_xlabel("Graph size N")
    ax1.set_ylabel("Success rate")
    ax1.set_ylim(0, 1.05)
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(ns, ratios, "s--", color="#e45756", linewidth=1.5, label="Mean cut ratio")
    ax2.set_ylabel("Mean cut / optimal")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left")

    plt.title(f"TTS proxy scaling ({data['sampler']}, steps={data['steps']})")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Max-Cut success-rate scaling (TTS proxy)")
    parser.add_argument("--nodes", type=int, nargs="+", default=[12, 14, 16, 18, 20])
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sampler", default="parallel_tempering")
    parser.add_argument(
        "--output-dir",
        default=os.path.join("experiments", "outputs", "sampler_scaling"),
    )
    args = parser.parse_args()

    out_dir = args.output_dir
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(REPO_ROOT, out_dir)

    data = scaling_success_rate(args.nodes, args.steps, args.trials, args.seed, args.sampler)
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "scaling.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {json_path}")

    for row in data["scaling"]:
        print(
            f"  N={row['n_nodes']:2d}  success={row['success_rate']:.2%}  "
            f"ratio={row['mean_cut_ratio']:.3f}  trials={row['trials']}"
        )

    plot_scaling(data, os.path.join(out_dir, "tts_scaling.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
