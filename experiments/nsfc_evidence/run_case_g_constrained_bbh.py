# -*- coding: utf-8 -*-
"""
Case G (P2): Constrained real BBH — HF questions with top_k=2/3 + synthetic exclusions.

Uses 7B LLM logits when --logits llm.
"""
from __future__ import annotations

import argparse
import sys

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.nsfc_evidence.common import (  # noqa: E402
    plot_dual_bars,
    resolve_out_dir,
    run_vci_modes,
    save_json,
    strip_per_task,
)
from qihc.orchestrator.bbh_llm import enrich_problems_with_llm_logits  # noqa: E402
from qihc.orchestrator.constrained_data import load_constrained_problems  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Case G: constrained HF BBH")
    parser.add_argument("--limit-per-task", type=int, default=50)
    parser.add_argument("--hf-tasks", nargs="*", default=None)
    parser.add_argument("--logits", choices=["pseudo", "llm"], default="llm")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--budget-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    import os

    model_name = args.model_name or os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
    out_dir = resolve_out_dir("case_g_constrained_bbh", args.output_dir)

    problems = load_constrained_problems(
        source="constrained_hf",
        seed=args.seed,
        limit_per_task=args.limit_per_task,
        hf_tasks=args.hf_tasks,
    )
    if args.logits == "llm":
        problems = enrich_problems_with_llm_logits(problems, model_name=model_name)

    modes = ["greedy", "vci-1", "vci-2"]
    summary = run_vci_modes(problems, modes, args.budget_steps, args.seed)

    payload = {
        "experiment": "case_g_constrained_bbh",
        "case": "P2-G",
        "description": "Constrained HF BBH subset selection with IF",
        "n_tasks": len(problems),
        "logits": args.logits,
        "model_name": model_name if args.logits == "llm" else None,
        "budget_steps": args.budget_steps,
        "summary": strip_per_task(summary),
    }
    save_json(f"{out_dir}/dual_axis.json", payload)
    plot_dual_bars(
        strip_per_task(summary),
        f"{out_dir}/dual_axis.png",
        f"Case G: constrained HF BBH (n={len(problems)})",
    )

    print(f"\n=== Case G constrained HF BBH (n={len(problems)}, {args.logits}) ===")
    for mode, s in payload["summary"].items():
        print(f"  {mode:8s} feas={s['feasible_rate']:.2%} exact={s['exact_match_rate']:.2%}")
    print(f"Saved: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
