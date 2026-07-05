# -*- coding: utf-8 -*-
"""
Handoff / temperature-mapping probe for VCI.

Sweeps semantic noise (proxy for LLM temperature / entropy H(q)) and
measures when VCI-2 q-refine helps over VCI-1.

Usage:
    python experiments/run_vci_handoff.py
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

from qihc.orchestrator.bbh import load_bbh_problems  # noqa: E402
from qihc.orchestrator.free_energy import softmax_entropy  # noqa: E402
from qihc.orchestrator.reasoning import SubsetProblem  # noqa: E402
from qihc.orchestrator.vci_scheduler import VCIConfig, VCIOrchestrator  # noqa: E402


def perturb_logits(problem: SubsetProblem, noise_scale: float, seed: int) -> SubsetProblem:
    rng = np.random.default_rng(seed)
    logits = problem.logits + rng.normal(0.0, noise_scale, size=problem.logits.shape)
    return SubsetProblem(
        text=problem.text,
        logits=logits,
        top_k=problem.top_k,
        exclusion_pairs=problem.exclusion_pairs,
        metadata=dict(problem.metadata),
    )


def run_sweep(
    base_problems: list[SubsetProblem],
    noise_scales: list[float],
    budget_steps: int,
    seed: int,
) -> list[dict]:
    rows: list[dict] = []
    for ni, sigma in enumerate(noise_scales):
        problems = [
            perturb_logits(p, sigma, seed=seed + ni * 100 + i)
            for i, p in enumerate(base_problems)
        ]
        mean_h = float(np.mean([softmax_entropy(p.logits) for p in problems]))
        t_lm = 1.0 / (mean_h + 1e-6)
        t_ising = 0.5 * (10.0 + 0.01)  # default T_start/T_end mid

        cfg1 = VCIConfig.tier_a(sampling_steps=budget_steps, seed=seed)
        cfg2 = VCIConfig.tier_a(sampling_steps=max(budget_steps // 2, 50), max_rounds=2, seed=seed)
        o1, o2 = VCIOrchestrator(cfg1), VCIOrchestrator(cfg2)

        feas1, feas2, em1, em2 = [], [], [], []
        for p in problems:
            r1 = o1.solve_subset(p, mode="vci-1")
            r2 = o2.solve_subset(p, mode="vci-2")
            feas1.append(r1.final_feasible)
            feas2.append(r2.final_feasible)
            gold = p.metadata.get("gold_mask")
            if gold is not None:
                gold = np.asarray(gold, dtype=bool)
                em1.append(bool(np.array_equal(r1.final_mask, gold)))
                em2.append(bool(np.array_equal(r2.final_mask, gold)))

        rows.append(
            {
                "noise_scale": sigma,
                "mean_entropy_Hq": mean_h,
                "T_lm_proxy": t_lm,
                "T_ising_mid": t_ising,
                "vci1_feasible_rate": float(np.mean(feas1)),
                "vci2_feasible_rate": float(np.mean(feas2)),
                "feasible_gain": float(np.mean(feas2) - np.mean(feas1)),
                "vci1_exact_match": float(np.mean(em1)) if em1 else None,
                "vci2_exact_match": float(np.mean(em2)) if em2 else None,
            }
        )
    return rows


def plot_handoff(rows: list[dict], out_path: str) -> None:
    h = [r["mean_entropy_Hq"] for r in rows]
    gain = [r["feasible_gain"] for r in rows]
    v1 = [r["vci1_feasible_rate"] for r in rows]
    v2 = [r["vci2_feasible_rate"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=120)
    axes[0].plot(h, v1, "o-", label="VCI-1", color="#e45756")
    axes[0].plot(h, v2, "s-", label="VCI-2", color="#4c78a8")
    axes[0].set_xlabel("Semantic entropy H(q)")
    axes[0].set_ylabel("Feasible rate")
    axes[0].set_title("Handoff: IF satisfaction vs H(q)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].bar(range(len(rows)), gain, color="#72b7b2")
    axes[1].set_xticks(range(len(rows)))
    axes[1].set_xticklabels([f"σ={r['noise_scale']:.2f}" for r in rows], rotation=30)
    axes[1].set_ylabel("VCI-2 − VCI-1 feasible gain")
    axes[1].set_title("Co-inference gain vs noise")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_temperature_map(rows: list[dict], out_path: str) -> None:
    plt.figure(figsize=(6, 4.5), dpi=120)
    for r in rows:
        plt.scatter(
            r["T_lm_proxy"],
            r["T_ising_mid"],
            s=80 + 400 * max(r["feasible_gain"], 0),
            alpha=0.7,
            label=f"σ={r['noise_scale']:.2f}",
        )
    plt.xlabel("T_lm proxy ∝ 1/H(q)")
    plt.ylabel("T_ising mid (schedule)")
    plt.title("Temperature mapping probe (marker ∝ VCI-2 gain)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="VCI handoff / temperature mapping")
    parser.add_argument("--budget-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default=os.path.join("experiments", "outputs", "vci_handoff"),
    )
    args = parser.parse_args()

    out_dir = args.output_dir
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(REPO_ROOT, out_dir)

    noise_scales = [0.0, 0.15, 0.3, 0.5, 0.8, 1.2]
    base = load_bbh_problems(seed=args.seed)
    rows = run_sweep(base, noise_scales, args.budget_steps, args.seed)

    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "handoff.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "budget_steps": args.budget_steps}, f, indent=2)
    print(f"Saved: {json_path}")

    for r in rows:
        print(
            f"  σ={r['noise_scale']:.2f} H={r['mean_entropy_Hq']:.2f}  "
            f"vci1={r['vci1_feasible_rate']:.2%} vci2={r['vci2_feasible_rate']:.2%}  "
            f"gain={r['feasible_gain']:+.2%}"
        )

    plot_handoff(rows, os.path.join(out_dir, "handoff_curve.png"))
    plot_temperature_map(rows, os.path.join(out_dir, "temperature_map.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
