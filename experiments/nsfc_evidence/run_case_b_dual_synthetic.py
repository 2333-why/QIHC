# -*- coding: utf-8 -*-
"""
Case B' (P0): CR+IF extended constraint set — dual-axis on synthetic n=200.

Modes: greedy | vci-1 | vci-2
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
    parser = argparse.ArgumentParser(description="Case B': dual-axis on synthetic 200")
    parser.add_argument("--n-tasks", type=int, default=200)
    parser.add_argument("--budget-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    out_dir = resolve_out_dir("case_b_dual_synthetic", args.output_dir)
    problems = load_synthetic_problems(n_tasks=args.n_tasks, seed=args.seed, regenerate=args.regenerate)
    modes = ["greedy", "vci-1", "vci-2"]
    summary = run_vci_modes(problems, modes, args.budget_steps, args.seed)

    payload = {
        "experiment": "case_b_dual_synthetic",
        "case": "P0-B",
        "description": "CR+IF extended constraint set (synthetic 200)",
        "n_tasks": len(problems),
        "budget_steps": args.budget_steps,
        "summary": strip_per_task(summary),
    }
    save_json(f"{out_dir}/dual_axis.json", payload)
    plot_dual_bars(
        strip_per_task(summary),
        f"{out_dir}/dual_axis.png",
        f"Case B': dual-axis synthetic (n={len(problems)})",
    )

    print(f"\n=== Case B' dual synthetic (n={len(problems)}) ===")
    for mode, s in payload["summary"].items():
        print(f"  {mode:8s} feas={s['feasible_rate']:.2%} exact={s['exact_match_rate']:.2%}")
    print(f"Saved: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
