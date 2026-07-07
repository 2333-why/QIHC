# -*- coding: utf-8 -*-
"""
Unified ablation aligned with CR paper (arXiv:2407.00071).

Compares paper baselines (zeroshot / linear / quadratic) against VCI-1/2.
Does NOT include logits-greedy (vci-0 / greedy).

Tracks:
  - accuracy + gain_over_zeroshot (CR paper primary)
  - feasible_rate (QIHC IF-constraint primary on bundled/synthetic)

Usage:
    python experiments/nsfc_evidence/run_unified_ablation.py --dataset bundled
    python experiments/nsfc_evidence/run_unified_ablation.py --dataset bundled --use-llm \\
        --model-name Qwen/Qwen2.5-7B-Instruct --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import sys
from typing import Any

import numpy as np

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
from qihc.orchestrator.bbh import BBHTask, load_bbh_tasks  # noqa: E402
from qihc.orchestrator.cr_pipeline import CRPaperMode  # noqa: E402

PAPER_MODES: list[CRPaperMode] = [
    "zeroshot",
    "linear",
    "quadratic",
    "vci-1",
    "vci-2",
]


def _load_tasks(dataset: str, n_tasks: int, seed: int) -> list[BBHTask]:
    if dataset == "bundled":
        return load_bbh_tasks(source="bundled")
    problems = load_synthetic_problems(n_tasks=n_tasks, seed=seed)
    tasks: list[BBHTask] = []
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
    return tasks


def _aggregate_seeds(seed_rows: list[dict[str, dict]]) -> dict[str, dict]:
    modes = seed_rows[0].keys()
    out: dict[str, dict] = {}
    for mode in modes:
        acc = [r[mode]["accuracy"] for r in seed_rows]
        feas = [r[mode]["feasible_rate"] for r in seed_rows]
        gain = [r[mode].get("gain_over_zeroshot", 0.0) for r in seed_rows]
        out[mode] = {
            "accuracy_mean": float(np.mean(acc)),
            "accuracy_std": float(np.std(acc)),
            "feasible_rate_mean": float(np.mean(feas)),
            "feasible_rate_std": float(np.std(feas)),
            "gain_over_zeroshot_mean": float(np.mean(gain)),
            "n_seeds": len(seed_rows),
        }
    return out


def run_one_seed(
    tasks: list[BBHTask],
    budget_steps: int,
    n_samples: int,
    seed: int,
    use_llm: bool,
    model_name: str,
    include_random: bool,
) -> dict[str, dict[str, Any]]:
    modes: list[CRPaperMode] = list(PAPER_MODES)
    if include_random:
        modes = ["zeroshot", "random", "linear", "quadratic", "vci-1", "vci-2"]

    data = run_cr_benchmark(
        tasks,
        modes=modes,
        budget_steps=budget_steps,
        n_samples=n_samples,
        seed=seed,
        use_llm=use_llm,
        model_name=model_name,
    )
    unified: dict[str, dict] = {}
    for m, s in data["summary"].items():
        unified[m] = {
            "accuracy": s["accuracy"],
            "feasible_rate": s["feasible_rate"],
            "gain_over_zeroshot": s.get("gain_over_zeroshot", 0.0),
            "mean_pbit_steps": s.get("mean_pbit_steps", 0),
            "mean_llm_calls": s.get("mean_llm_calls", 0),
            "family": "vci" if m.startswith("vci") else "cr",
        }
    return unified


def main() -> int:
    parser = argparse.ArgumentParser(description="CR paper-aligned unified ablation")
    parser.add_argument("--dataset", choices=["bundled", "synthetic"], default="bundled")
    parser.add_argument("--n-tasks", type=int, default=200)
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--budget-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--use-llm", action="store_true", help="Real LLM (required for paper-comparable acc)")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--include-random", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    seeds = args.seeds if args.seeds else [args.seed]
    out_dir = resolve_out_dir("unified_ablation", args.output_dir)

    per_seed: list[dict] = []
    tasks: list[BBHTask] = []
    for si, seed in enumerate(seeds):
        print(f"\n=== Seed {seed} ({si + 1}/{len(seeds)}) ===")
        tasks = _load_tasks(args.dataset, args.n_tasks, seed)
        summary = run_one_seed(
            tasks,
            args.budget_steps,
            args.n_samples,
            seed,
            args.use_llm,
            args.model_name,
            args.include_random,
        )
        per_seed.append(summary)
        for mode, s in sorted(summary.items()):
            print(
                f"  {mode:12s} acc={s['accuracy']:.2%} feas={s['feasible_rate']:.2%} "
                f"gain={s['gain_over_zeroshot']:+.2%}"
            )

    last = per_seed[-1]
    payload = {
        "experiment": "unified_ablation_cr_paper",
        "dataset": args.dataset,
        "use_llm": args.use_llm,
        "model_name": args.model_name if args.use_llm else None,
        "n_tasks": len(tasks),
        "budget_steps": args.budget_steps,
        "n_samples": args.n_samples,
        "seeds": seeds,
        "modes": list(last.keys()),
        "baseline": "zeroshot (CR paper LLM T=0, NOT logits-greedy)",
        "summary": last,
        "per_seed": per_seed,
        "aggregate": _aggregate_seeds(per_seed) if len(seeds) > 1 else None,
        "reference": "Combinatorial Reasoning arXiv:2407.00071",
    }
    save_json(f"{out_dir}/unified_ablation.json", payload)

    plot_rows = {
        k: {
            "feasible_rate": v["feasible_rate"],
            "exact_match_rate": v["accuracy"],
        }
        for k, v in last.items()
    }
    plot_dual_bars(
        plot_rows,
        f"{out_dir}/unified_ablation.png",
        f"CR paper ablation ({args.dataset}, n={len(tasks)}, seeds={seeds})",
    )
    print(f"\nSaved: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
