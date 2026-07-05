# -*- coding: utf-8 -*-
"""
Compare p-bit / Ising samplers on Max-Cut (Gibbs, PT, SQA, SA).

Usage (from repository root):
    python experiments/run_sampler_benchmark.py
    python experiments/run_sampler_benchmark.py --nodes 20 --steps 800 --seed 0
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from qihc import IsingModel
from qihc.ising import maxcut


def run_benchmark(n_nodes: int = 20, edge_p: float = 0.5, steps: int = 800, seed: int = 0):
    np.random.seed(seed)
    G = nx.erdos_renyi_graph(n_nodes, edge_p, seed=seed)
    J = maxcut.max_cut_to_ising(G)

    opt_cut = None
    if n_nodes <= 22:
        opt_cut, _ = maxcut.brute_force_max_cut(G)

    common = dict(J=J, steps=steps, T_start=10.0, T_end=0.01, k=1.0)
    results: dict[str, dict] = {}

    model = IsingModel(size=n_nodes)
    spins, e_trace, _ = model.gibbs_sampling_Maxcut(**common, sequential=True)
    cut = maxcut.calculate_cut_value(G, maxcut.convert_spins_to_cut(spins))
    results["Gibbs"] = {"cut": cut, "energy": e_trace}

    model = IsingModel(size=n_nodes)
    spins, e_trace, _ = model.parallel_tempering_Maxcut(
        **common, n_replicas=6, swap_interval=20, sequential=True
    )
    cut = maxcut.calculate_cut_value(G, maxcut.convert_spins_to_cut(spins))
    results["Parallel Tempering"] = {"cut": cut, "energy": e_trace}

    model = IsingModel(size=n_nodes)
    spins, e_trace, _ = model.simulated_quantum_annealing_Maxcut(
        **common, Gamma_start=3.0, Gamma_end=0.01, m_slices=6
    )
    cut = maxcut.calculate_cut_value(G, maxcut.convert_spins_to_cut(spins))
    results["SQA"] = {"cut": cut, "energy": e_trace}

    model = IsingModel(size=n_nodes)
    spins, e_trace, _ = model.ising_simulated_annealing_Maxcut_Syn(**common)
    cut = maxcut.calculate_cut_value(G, maxcut.convert_spins_to_cut(spins))
    results["SA (sync)"] = {"cut": cut, "energy": e_trace}

    model = IsingModel(size=n_nodes)
    spins, e_trace, _ = model.ising_simulated_annealing_Maxcut_Asyn(**common)
    cut = maxcut.calculate_cut_value(G, maxcut.convert_spins_to_cut(spins))
    results["SA (async)"] = {"cut": cut, "energy": e_trace}

    print(f"=== Max-Cut benchmark (n={n_nodes}, steps={steps}, seed={seed}) ===")
    if opt_cut is not None:
        print(f"Brute-force optimal cut: {opt_cut}")
    for name, info in results.items():
        ratio = info["cut"] / opt_cut if opt_cut else float("nan")
        print(f"  {name:22s} cut={info['cut']:4d}  ratio={ratio:.3f}")

    return results, opt_cut


def plot_results(results: dict, out_path: str) -> None:
    plt.figure(figsize=(10, 5), dpi=120)
    for name, info in results.items():
        plt.plot(info["energy"], label=name, linewidth=1.5)
    plt.xlabel("Iteration / sweep")
    plt.ylabel("Ising energy")
    plt.title("Energy convergence: Gibbs / PT / SQA / SA")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="QIHC sampler benchmark on Max-Cut")
    parser.add_argument("--nodes", type=int, default=20)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        default=os.path.join("experiments", "outputs", "sampler_energy_convergence.png"),
    )
    args = parser.parse_args()

    # Allow running as `python experiments/run_sampler_benchmark.py` from repo root
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    results, _ = run_benchmark(n_nodes=args.nodes, steps=args.steps, seed=args.seed)
    out_path = args.output
    if not os.path.isabs(out_path):
        out_path = os.path.join(repo_root, out_path)
    plot_results(results, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
