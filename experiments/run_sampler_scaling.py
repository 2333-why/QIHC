# -*- coding: utf-8 -*-
"""
TTS (time-to-solution) scaling benchmark on Max-Cut.

Supports n=12–500 via exact (n<=22) or greedy heuristic reference cuts.
Optional --measure-tts records median steps to reach 90% of reference cut.

Usage:
    python experiments/run_sampler_scaling.py
    python experiments/run_sampler_scaling.py --nodes 50 100 200 500 --measure-tts
    python experiments/run_sampler_scaling.py --nodes 12 16 20 --trials 10
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

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

TTS_STEP_LADDER = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]


def _sample_cut(G, J, n_nodes: int, steps: int, sampler: str):
    common = dict(J=J, steps=steps, T_start=10.0, T_end=0.01, k=1.0)
    model = IsingModel(size=n_nodes)
    if sampler == "gibbs":
        spins, _, _ = model.gibbs_sampling_Maxcut(**common, sequential=True)
    elif sampler == "parallel_tempering":
        spins, _, _ = model.parallel_tempering_Maxcut(
            **common, n_replicas=6, swap_interval=20, sequential=True
        )
    else:
        raise ValueError(sampler)
    cut = maxcut.calculate_cut_value(G, maxcut.convert_spins_to_cut(spins))
    return int(cut)


def run_trial_fixed_steps(
    n_nodes: int,
    steps: int,
    seed: int,
    sampler: str,
    target_ratio: float = 0.90,
) -> dict | None:
    np.random.seed(seed)
    G = nx.erdos_renyi_graph(n_nodes, 0.5, seed=seed)
    J = maxcut.max_cut_to_ising(G)
    opt_cut, ref_kind = maxcut.reference_max_cut(G, seed=seed)
    if opt_cut <= 0:
        return None
    target = target_ratio * opt_cut
    cut = _sample_cut(G, J, n_nodes, steps, sampler)
    return {
        "n_nodes": n_nodes,
        "seed": seed,
        "ref_cut": int(opt_cut),
        "ref_kind": ref_kind,
        "cut": cut,
        "ratio": float(cut / opt_cut),
        "success": bool(cut >= target),
        "steps": steps,
        "target": float(target),
    }


def run_tts_trial(
    n_nodes: int,
    seed: int,
    sampler: str,
    step_ladder: list[int],
    target_ratio: float = 0.90,
) -> dict | None:
    np.random.seed(seed)
    G = nx.erdos_renyi_graph(n_nodes, 0.5, seed=seed)
    J = maxcut.max_cut_to_ising(G)
    opt_cut, ref_kind = maxcut.reference_max_cut(G, seed=seed)
    if opt_cut <= 0:
        return None
    target = target_ratio * opt_cut
    tts = None
    t0 = time.perf_counter()
    for steps in step_ladder:
        cut = _sample_cut(G, J, n_nodes, steps, sampler)
        if cut >= target:
            tts = steps
            break
    elapsed = time.perf_counter() - t0
    if tts is None:
        tts = step_ladder[-1]
    return {
        "n_nodes": n_nodes,
        "seed": seed,
        "ref_cut": int(opt_cut),
        "ref_kind": ref_kind,
        "median_tts_steps": int(tts),
        "success": bool(tts < step_ladder[-1]),
        "wall_time_s": float(elapsed),
        "target_ratio": target_ratio,
    }


def scaling_success_rate(
    node_list: list[int],
    steps: int,
    trials: int,
    seed: int,
    sampler: str,
    measure_tts: bool,
    step_ladder: list[int],
) -> dict:
    results = []
    rng = np.random.default_rng(seed)
    for n in node_list:
        trial_outcomes = []
        tts_values = []
        for _ in range(trials):
            trial_seed = int(rng.integers(0, 1_000_000))
            if measure_tts:
                out = run_tts_trial(n, trial_seed, sampler, step_ladder)
                if out is not None:
                    tts_values.append(out)
                    trial_outcomes.append({"success": out["success"], "ratio": 1.0})
            else:
                out = run_trial_fixed_steps(n, steps, trial_seed, sampler)
                if out is not None:
                    trial_outcomes.append(out)
        if trial_outcomes:
            rate = float(np.mean([t["success"] for t in trial_outcomes]))
            mean_ratio = float(np.mean([t.get("ratio", 0) for t in trial_outcomes]))
        else:
            rate, mean_ratio = float("nan"), float("nan")
        row = {
            "n_nodes": n,
            "success_rate": rate,
            "mean_cut_ratio": mean_ratio,
            "trials": len(trial_outcomes),
            "ref_kind": trial_outcomes[0].get("ref_kind", "unknown") if trial_outcomes else None,
        }
        if tts_values:
            row["median_tts_steps"] = float(np.median([t["median_tts_steps"] for t in tts_values]))
            row["tts_success_rate"] = float(np.mean([t["success"] for t in tts_values]))
        results.append(row)
    return {
        "sampler": sampler,
        "steps": steps if not measure_tts else None,
        "measure_tts": measure_tts,
        "step_ladder": step_ladder if measure_tts else None,
        "scaling": results,
    }


def plot_scaling(data: dict, out_path: str) -> None:
    scaling = data["scaling"]
    ns = [r["n_nodes"] for r in scaling]
    rates = [r["success_rate"] for r in scaling]
    ratios = [r["mean_cut_ratio"] for r in scaling]

    fig, ax1 = plt.subplots(figsize=(8, 4.5), dpi=120)
    ax1.plot(ns, rates, "o-", color="#4c78a8", linewidth=2, label="Success rate @90% ref")
    ax1.set_xlabel("Graph size N")
    ax1.set_ylabel("Success rate")
    ax1.set_ylim(0, 1.05)
    ax1.grid(alpha=0.3)

    if data.get("measure_tts") and scaling[0].get("median_tts_steps") is not None:
        tts = [r.get("median_tts_steps", float("nan")) for r in scaling]
        ax2 = ax1.twinx()
        ax2.plot(ns, tts, "s--", color="#e45756", linewidth=1.5, label="Median TTS steps")
        ax2.set_ylabel("Median steps to target")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    else:
        ax2 = ax1.twinx()
        ax2.plot(ns, ratios, "s--", color="#e45756", linewidth=1.5, label="Mean cut ratio")
        ax2.set_ylabel("Mean cut / reference")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left")

    title = f"Max-Cut scaling ({data['sampler']}"
    if data.get("measure_tts"):
        title += ", TTS)"
    else:
        title += f", steps={data['steps']})"
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Max-Cut scaling / TTS benchmark")
    parser.add_argument(
        "--nodes",
        type=int,
        nargs="+",
        default=[12, 14, 16, 18, 20, 22, 50, 100, 200, 500],
    )
    parser.add_argument("--steps", type=int, default=800, help="Fixed steps when not measuring TTS")
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sampler", default="parallel_tempering")
    parser.add_argument(
        "--measure-tts",
        action="store_true",
        help="Measure median steps-to-target via step ladder",
    )
    parser.add_argument(
        "--tts-steps",
        type=int,
        nargs="+",
        default=None,
        help="Custom TTS step ladder",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("experiments", "outputs", "sampler_scaling"),
    )
    args = parser.parse_args()

    out_dir = args.output_dir
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(REPO_ROOT, out_dir)

    ladder = args.tts_steps or TTS_STEP_LADDER
    data = scaling_success_rate(
        args.nodes,
        args.steps,
        args.trials,
        args.seed,
        args.sampler,
        args.measure_tts,
        ladder,
    )
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "scaling.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {json_path}")

    for row in data["scaling"]:
        line = (
            f"  N={row['n_nodes']:3d}  success={row['success_rate']:.2%}  "
            f"ratio={row['mean_cut_ratio']:.3f}  trials={row['trials']}"
        )
        if row.get("median_tts_steps") is not None:
            line += f"  median_tts={row['median_tts_steps']:.0f}"
        print(line)

    plot_scaling(data, os.path.join(out_dir, "tts_scaling.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
