# -*- coding: utf-8 -*-
"""
Case A' (P0): CR-aligned subset selection on synthetic constrained set (n=200).

Modes: zeroshot | random | linear | quadratic | vci-1 | vci-2
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
    save_json,
)
from experiments.nsfc_evidence.run_cr_bbh import run_cr_benchmark  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Case A': CR-aligned subset (synthetic 200)")
    parser.add_argument("--n-tasks", type=int, default=200)
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--budget-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    out_dir = resolve_out_dir("case_a_cr_subset", args.output_dir)
    problems = load_synthetic_problems(n_tasks=args.n_tasks, seed=args.seed, regenerate=args.regenerate)

    from qihc.orchestrator.bbh import BBHTask

    tasks = []
    for p in problems:
        tasks.append(
            BBHTask(
                task_id=str(p.metadata["task_id"]),
                task_type=str(p.metadata.get("task_type", "synthetic")),
                text=p.text,
                candidates=list(p.metadata.get("candidates", [])),
                top_k=p.top_k,
                gold_indices=list(p.metadata.get("gold_indices", [])),
                exclusion_pairs=list(p.exclusion_pairs),
                logits=p.logits.copy(),
            )
        )

    modes = ["zeroshot", "random", "linear", "quadratic", "vci-1", "vci-2"]
    data = run_cr_benchmark(
        tasks,
        modes=modes,
        budget_steps=args.budget_steps,
        n_samples=args.n_samples,
        seed=args.seed,
        use_llm=False,
        model_name="",
    )

    payload = {
        "experiment": "case_a_cr_subset",
        "case": "P0-A",
        "description": "CR-aligned subset selection on synthetic constrained set",
        "n_tasks": len(tasks),
        "modes": modes,
        **data,
        "reference": "Combinatorial Reasoning arXiv:2407.00071",
    }
    save_json(f"{out_dir}/cr_protocol.json", payload)
    plot_dual_bars(
        {m: {"feasible_rate": s["feasible_rate"], "exact_match_rate": s["accuracy"]} for m, s in data["summary"].items()},
        f"{out_dir}/cr_protocol.png",
        f"Case A': CR subset (n={len(tasks)})",
    )

    print(f"\n=== Case A' CR subset (n={len(tasks)}) ===")
    for mode, s in data["summary"].items():
        print(f"  {mode:12s} acc={s['accuracy']:.2%} feas={s['feasible_rate']:.2%}")
    print(f"Saved: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
