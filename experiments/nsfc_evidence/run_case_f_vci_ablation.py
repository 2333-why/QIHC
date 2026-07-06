# -*- coding: utf-8 -*-
"""
Case F (P1): VCI degradation chain — vci-0 | vci-1 | vci-2 | vci-full on synthetic 200.
"""
from __future__ import annotations

import argparse
import sys

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.nsfc_evidence.common import (  # noqa: E402
    load_synthetic_problems,
    plot_dual_bars,
    resolve_out_dir,
    run_vci_modes,
    save_json,
    strip_per_task,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Case F: VCI ablation chain")
    parser.add_argument("--n-tasks", type=int, default=200)
    parser.add_argument("--budget-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    out_dir = resolve_out_dir("case_f_vci_ablation", args.output_dir)
    problems = load_synthetic_problems(n_tasks=args.n_tasks, seed=args.seed)
    modes = ["vci-0", "vci-1", "vci-2", "vci-full"]
    summary = run_vci_modes(problems, modes, args.budget_steps, args.seed)

    payload = {
        "experiment": "case_f_vci_ablation",
        "case": "P1-F",
        "description": "VCI degradation chain on synthetic constrained set",
        "n_tasks": len(problems),
        "budget_steps": args.budget_steps,
        "summary": strip_per_task(summary),
    }
    save_json(f"{out_dir}/vci_ablation.json", payload)
    plot_dual_bars(
        strip_per_task(summary),
        f"{out_dir}/vci_ablation.png",
        f"Case F: VCI chain (n={len(problems)})",
    )

    print(f"\n=== Case F VCI ablation (n={len(problems)}) ===")
    for mode, s in payload["summary"].items():
        print(f"  {mode:10s} feas={s['feasible_rate']:.2%} exact={s['exact_match_rate']:.2%}")
    print(f"Saved: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
