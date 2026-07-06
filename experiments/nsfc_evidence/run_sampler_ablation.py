# -*- coding: utf-8 -*-
"""PT vs Gibbs vs SA on BBH-derived QUBO instances (sampler ablation, TAPT-aligned)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from qihc.orchestrator.backend import PBitBackend  # noqa: E402
from qihc.orchestrator.bbh import load_bbh_problems  # noqa: E402
from qihc.orchestrator.encoder import refine_mask_to_top_k  # noqa: E402
from qihc.orchestrator.reasoning import is_feasible, subset_to_ising  # noqa: E402
from qihc.orchestrator.vci_scheduler import VCIConfig  # noqa: E402


def solve_one(problem, sampler: str, steps: int, seed: int) -> dict:
    np.random.seed(seed)
    logits = problem.logits
    weight, field = subset_to_ising(
        logits,
        top_k=problem.top_k,
        exclusion_pairs=problem.exclusion_pairs,
    )
    cfg = VCIConfig.tier_a(sampling_steps=steps, sampler=sampler, seed=seed)  # type: ignore[arg-type]
    backend = PBitBackend(cfg)
    t0 = time.perf_counter()
    mask, energy, _ = backend.solve(weight, field)
    elapsed = time.perf_counter() - t0
    mask = refine_mask_to_top_k(logits, mask, problem.top_k)
    return {
        "feasible": is_feasible(mask, problem.top_k, problem.exclusion_pairs),
        "energy": float(energy),
        "elapsed_s": elapsed,
        "sampler": sampler,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Sampler ablation on bundled QUBO")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--samplers",
        nargs="*",
        default=["gibbs", "parallel_tempering", "sa_sync"],
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    out_dir = args.output_dir or os.path.join(REPO_ROOT, "experiments", "outputs", "nsfc_evidence", "sampler_ablation")
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(REPO_ROOT, out_dir)

    problems = load_bbh_problems(source="bundled", seed=args.seed)
    summary = {}
    for sampler in args.samplers:
        rows = [solve_one(p, sampler, args.steps, args.seed + i) for i, p in enumerate(problems)]
        summary[sampler] = {
            "feasible_rate": float(np.mean([r["feasible"] for r in rows])),
            "mean_energy": float(np.mean([r["energy"] for r in rows])),
            "mean_time_s": float(np.mean([r["elapsed_s"] for r in rows])),
            "win_rate_vs_gibbs": None,
        }

    if "gibbs" in summary and "parallel_tempering" in summary:
        g_feas = [solve_one(p, "gibbs", args.steps, args.seed + i)["feasible"] for i, p in enumerate(problems)]
        pt_feas = [
            solve_one(p, "parallel_tempering", args.steps, args.seed + i)["feasible"] for i, p in enumerate(problems)
        ]
        wins = sum(1 for g, pt in zip(g_feas, pt_feas) if pt and not g) + sum(
            1 for g, pt in zip(g_feas, pt_feas) if pt == g
        )
        summary["parallel_tempering"]["win_rate_vs_gibbs"] = wins / len(problems)

    modes = list(summary.keys())
    feas = [summary[m]["feasible_rate"] for m in modes]
    fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
    ax.bar(modes, feas, color=["#e45756", "#4c78a8", "#72b7b2"][: len(modes)])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Feasible rate")
    ax.set_title(f"Sampler ablation (bundled, steps={args.steps})")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "sampler_ablation.png"), bbox_inches="tight")
    plt.close()

    payload = {"experiment": "sampler_ablation", "steps": args.steps, "summary": summary}
    with open(os.path.join(out_dir, "sampler_ablation.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved: {out_dir}")
    for m, s in summary.items():
        print(f"  {m:22s} feas={s['feasible_rate']:.2%} time={s['mean_time_s']:.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
