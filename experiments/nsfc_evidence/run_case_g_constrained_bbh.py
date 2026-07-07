# -*- coding: utf-8 -*-
"""
Case G (P2): Constrained real BBH — CR paper baselines + VCI with IF constraints.

Modes: zeroshot | linear | quadratic | vci-1 | vci-2  (NOT greedy)
"""
from __future__ import annotations

import argparse
import os
import sys

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.nsfc_evidence.common import (  # noqa: E402
    plot_dual_bars,
    resolve_out_dir,
    run_cr_paper_modes,
    save_json,
)
from qihc.orchestrator.bbh_llm import enrich_problems_with_llm_logits  # noqa: E402
from qihc.orchestrator.constrained_data import load_constrained_problems  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Case G: constrained HF BBH + CR paper")
    parser.add_argument("--limit-per-task", type=int, default=50)
    parser.add_argument("--hf-tasks", nargs="*", default=None)
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--use-llm", action="store_true", help="Real LLM for CR paper pipeline")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--budget-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    model_name = args.model_name or os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
    out_dir = resolve_out_dir("case_g_constrained_bbh", args.output_dir)

    problems = load_constrained_problems(
        source="constrained_hf",
        seed=args.seed,
        limit_per_task=args.limit_per_task,
        hf_tasks=args.hf_tasks,
    )
    if args.use_llm:
        problems = enrich_problems_with_llm_logits(problems, model_name=model_name)

    modes = ["zeroshot", "linear", "quadratic", "vci-1", "vci-2"]
    summary = run_cr_paper_modes(
        problems,
        modes,
        args.budget_steps,
        args.n_samples,
        args.seed,
        use_llm=args.use_llm,
        model_name=model_name,
    )

    payload = {
        "experiment": "case_g_constrained_bbh",
        "case": "P2-G",
        "description": "Constrained HF BBH: CR paper baselines vs VCI (IF satisfaction)",
        "n_tasks": len(problems),
        "use_llm": args.use_llm,
        "model_name": model_name if args.use_llm else None,
        "budget_steps": args.budget_steps,
        "baseline": "zeroshot (CR paper)",
        "summary": summary,
    }
    save_json(f"{out_dir}/dual_axis.json", payload)
    plot_dual_bars(
        summary,
        f"{out_dir}/dual_axis.png",
        f"Case G: constrained HF (n={len(problems)})",
    )

    print(f"\n=== Case G constrained HF (n={len(problems)}, llm={args.use_llm}) ===")
    for mode, s in summary.items():
        print(
            f"  {mode:12s} acc={s['accuracy']:.2%} feas={s['feasible_rate']:.2%} "
            f"gain={s['gain_over_zeroshot']:+.2%}"
        )
    print(f"Saved: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
