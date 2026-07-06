# -*- coding: utf-8 -*-
"""Record F(q,s) trajectories across bundled constraint problems."""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from qihc.orchestrator.bbh import load_bbh_problems  # noqa: E402
from qihc.orchestrator.vci_scheduler import VCIConfig, VCIOrchestrator  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="F trajectory evidence")
    parser.add_argument("--budget-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    out_dir = args.output_dir or os.path.join(REPO_ROOT, "experiments", "outputs", "nsfc_evidence", "f_trajectories")
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(REPO_ROOT, out_dir)

    problems = load_bbh_problems(source="bundled", seed=args.seed)
    cfg = VCIConfig.tier_a(sampling_steps=max(args.budget_steps // 2, 50), max_rounds=2, seed=args.seed)
    orch = VCIOrchestrator(cfg)

    traces: list[dict] = []
    all_f_vci2: list[list[float]] = []

    for p in problems:
        r1 = orch.solve_subset(p, mode="vci-1")
        r2 = orch.solve_subset(p, mode="vci-2")
        traces.append(
            {
                "task_id": p.metadata.get("task_id"),
                "vci1_F": [x.free_energy.total for x in r1.rounds],
                "vci2_F": [x.free_energy.total for x in r2.rounds],
                "vci1_feasible": r1.final_feasible,
                "vci2_feasible": r2.final_feasible,
            }
        )
        all_f_vci2.append([x.free_energy.total for x in r2.rounds])

    max_len = max(len(t) for t in all_f_vci2)
    mat = np.full((len(all_f_vci2), max_len), np.nan)
    for i, t in enumerate(all_f_vci2):
        for j, v in enumerate(t):
            mat[i, j] = v
    mean_f = np.nanmean(mat, axis=0)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    rounds = np.arange(max_len)
    ax.plot(rounds, mean_f, "o-", color="#4c78a8", linewidth=2, label="Mean F (VCI-2)")
    ax.set_xlabel("VCI round")
    ax.set_ylabel("F(q, s)")
    ax.set_title("Free-energy descent (bundled constraint set, n=40)")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "f_descent_mean.png"), bbox_inches="tight")
    plt.close()

    payload = {
        "experiment": "f_trajectories",
        "n_tasks": len(problems),
        "mean_F_trace_vci2": [float(x) for x in mean_f],
        "traces": traces,
    }
    with open(os.path.join(out_dir, "f_trajectories.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Saved F trajectories: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
