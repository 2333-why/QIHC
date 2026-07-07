# -*- coding: utf-8 -*-
"""
Case B' (P0): CR paper baselines + VCI on synthetic constrained set (n=200).

Modes: zeroshot | linear | quadratic | vci-1 | vci-2  (NOT logits-greedy)
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
    run_cr_paper_modes,
    save_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Case B': CR paper + VCI on synthetic 200")
    parser.add_argument("--n-tasks", type=int, default=200)
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--budget-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    out_dir = resolve_out_dir("case_b_dual_synthetic", args.output_dir)
    problems = load_synthetic_problems(n_tasks=args.n_tasks, seed=args.seed, regenerate=args.regenerate)
    modes = ["zeroshot", "linear", "quadratic", "vci-1", "vci-2"]
    summary = run_cr_paper_modes(
        problems,
        modes,
        args.budget_steps,
        args.n_samples,
        args.seed,
        use_llm=args.use_llm,
        model_name=args.model_name,
    )

    payload = {
        "experiment": "case_b_dual_synthetic",
        "case": "P0-B",
        "description": "CR paper baselines + VCI on synthetic constrained set",
        "n_tasks": len(problems),
        "budget_steps": args.budget_steps,
        "n_samples": args.n_samples,
        "baseline": "zeroshot (CR paper)",
        "summary": summary,
        "reference": "Combinatorial Reasoning arXiv:2407.00071",
    }
    save_json(f"{out_dir}/dual_axis.json", payload)
    plot_dual_bars(
        summary,
        f"{out_dir}/dual_axis.png",
        f"Case B': CR+VCI synthetic (n={len(problems)})",
    )

    print(f"\n=== Case B' CR+VCI synthetic (n={len(problems)}) ===")
    for mode, s in summary.items():
        print(
            f"  {mode:12s} acc={s['accuracy']:.2%} feas={s['feasible_rate']:.2%} "
            f"gain={s['gain_over_zeroshot']:+.2%}"
        )
    print(f"Saved: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
