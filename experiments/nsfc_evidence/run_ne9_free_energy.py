# -*- coding: utf-8 -*-
"""NE9: Joint free-energy monotone descent under three-law conditions (Theorem 4).

Usage:
  python experiments/nsfc_evidence/run_ne9_free_energy.py --profile smoke
  python experiments/nsfc_evidence/run_ne9_free_energy.py --profile full
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments.nsfc_evidence.three_law_common import (  # noqa: E402
    load_problems,
    resolve_out_dir,
    save_fig,
    save_json,
)
from qihc.orchestrator.free_energy import compute_free_energy  # noqa: E402
from qihc.orchestrator.vci_scheduler import VCIConfig, VCIOrchestrator  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="NE9 free-energy descent")
    parser.add_argument("--profile", choices=["smoke", "full"], default="full")
    parser.add_argument("--n-tasks", type=int, default=None)
    parser.add_argument("--budget-steps", type=int, default=None)
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    n_tasks = args.n_tasks or (10 if args.profile == "smoke" else 40)
    steps = args.budget_steps or (100 if args.profile == "smoke" else 200)
    out_dir = resolve_out_dir("ne9_free_energy", args.output_dir)

    problems = load_problems("bundled", n_tasks=n_tasks, seed=args.seed)
    cfg = VCIConfig.tier_a(
        sampling_steps=steps,
        max_rounds=args.max_rounds,
        seed=args.seed,
        sampler="parallel_tempering",
    )
    orch = VCIOrchestrator(cfg)

    trajectories = []
    mono_flags = []
    for p in problems:
        # Force multi-round by using vci-full even if early feasible
        res = orch.solve_subset(p, mode="vci-full")
        f_vals = [r.free_energy.total for r in res.rounds]
        # Also record explicit refine demo when only 1 round
        if len(f_vals) < 2:
            demo = orch.trace_refine_demo(p)
            f_vals = [r.free_energy.total for r in demo]
        trajectories.append(f_vals)
        diffs = np.diff(f_vals)
        mono_flags.append(bool(np.all(diffs <= 1e-6)))

    # pad and plot mean ± std
    max_t = max(len(t) for t in trajectories)
    mat = np.full((len(trajectories), max_t), np.nan)
    for i, t in enumerate(trajectories):
        mat[i, : len(t)] = t
    mean = np.nanmean(mat, axis=0)
    std = np.nanstd(mat, axis=0)
    xs = np.arange(mean.size)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=130)
    ax.plot(xs, mean, color="#4c78a8", lw=2, label="mean F")
    ax.fill_between(xs, mean - std, mean + std, color="#4c78a8", alpha=0.2)
    for t in trajectories[: min(8, len(trajectories))]:
        ax.plot(range(len(t)), t, color="gray", alpha=0.25, lw=0.8)
    ax.set_xlabel("Round")
    ax.set_ylabel(r"$\mathcal{F}_\beta(q,m)$")
    ax.set_title("NE9: Free-energy descent (Theorem 4)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_fig(os.path.join(out_dir, "ne9_free_energy_descent.png"))

    payload = {
        "profile": args.profile,
        "n_tasks": n_tasks,
        "budget_steps": steps,
        "monotone_fraction": float(np.mean(mono_flags)),
        "mean_final_F": float(np.nanmean(mat[:, -1])),
        "mean_delta_F": float(np.nanmean(mat[:, 0] - mat[:, min(1, mat.shape[1] - 1)])),
        "trajectories": trajectories,
    }
    save_json(os.path.join(out_dir, "summary.json"), payload)
    print(f"NE9 done. monotone_fraction={payload['monotone_fraction']:.3f} -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
