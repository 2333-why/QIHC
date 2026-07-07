# -*- coding: utf-8 -*-
"""
Case F (P1): CR paper degradation chain + VCI on synthetic 200.

zeroshot → linear → quadratic → vci-1 → vci-2 → vci-full
(vci-full = extended VCI rounds; baselines are CR paper modes, NOT greedy)
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.nsfc_evidence.common import (  # noqa: E402
    load_synthetic_problems,
    plot_dual_bars,
    resolve_out_dir,
    run_cr_paper_modes,
    save_json,
)
from qihc.orchestrator.bbh import evaluate_prediction  # noqa: E402
from qihc.orchestrator.vci_scheduler import VCIConfig, VCIOrchestrator  # noqa: E402


def run_vci_full(problems, budget_steps: int, seed: int) -> dict:
    steps = max(budget_steps // 2, 50)
    cfg = VCIConfig.tier_a(sampling_steps=steps, max_rounds=4, seed=seed)
    orch = VCIOrchestrator(cfg)
    feas, exact = [], []
    for p in problems:
        res = orch.solve_subset(p, mode="vci-full")
        ev = evaluate_prediction(p, res.final_mask)
        feas.append(ev["feasible"])
        exact.append(ev.get("exact_match", False))
    return {
        "mode": "vci-full",
        "accuracy": float(np.mean(exact)),
        "feasible_rate": float(np.mean(feas)),
        "exact_match_rate": float(np.mean(exact)),
        "mean_pbit_steps": steps * 4,
        "llm_calls": 50,
        "family": "vci",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Case F: CR paper chain + VCI")
    parser.add_argument("--n-tasks", type=int, default=200)
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--budget-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    out_dir = resolve_out_dir("case_f_vci_ablation", args.output_dir)
    problems = load_synthetic_problems(n_tasks=args.n_tasks, seed=args.seed)
    cr_modes = ["zeroshot", "linear", "quadratic", "vci-1", "vci-2"]
    summary = run_cr_paper_modes(
        problems, cr_modes, args.budget_steps, args.n_samples, args.seed
    )
    summary["vci-full"] = run_vci_full(problems, args.budget_steps, args.seed)
    zs = summary.get("zeroshot", {}).get("accuracy", 0.0)
    for m in summary:
        summary[m]["gain_over_zeroshot"] = float(summary[m]["accuracy"] - zs)

    payload = {
        "experiment": "case_f_vci_ablation",
        "case": "P1-F",
        "description": "CR paper chain + VCI-full on synthetic constrained set",
        "n_tasks": len(problems),
        "budget_steps": args.budget_steps,
        "baseline": "zeroshot (CR paper)",
        "summary": summary,
    }
    save_json(f"{out_dir}/vci_ablation.json", payload)
    plot_dual_bars(summary, f"{out_dir}/vci_ablation.png", f"Case F: CR→VCI chain (n={len(problems)})")

    print(f"\n=== Case F CR→VCI chain (n={len(problems)}) ===")
    for mode, s in summary.items():
        print(
            f"  {mode:12s} acc={s['accuracy']:.2%} feas={s['feasible_rate']:.2%} "
            f"gain={s.get('gain_over_zeroshot', 0):+.2%}"
        )
    print(f"Saved: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
