# -*- coding: utf-8 -*-
"""
Unified ablation: VCI-0/1/2 + CR protocol modes in one reproducible table.

Supports bundled (n=40) or synthetic (n=200), pseudo or Qwen LLM logits, multi-seed.

Usage:
    python experiments/nsfc_evidence/run_unified_ablation.py --dataset bundled
    python experiments/nsfc_evidence/run_unified_ablation.py --dataset synthetic --n-tasks 200
    python experiments/nsfc_evidence/run_unified_ablation.py --dataset bundled --logits llm \\
        --model-name Qwen/Qwen2.5-7B-Instruct --seeds 0 1 2 3 4
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
    run_vci_modes,
    save_json,
    strip_per_task,
)
from experiments.nsfc_evidence.run_cr_bbh import run_cr_benchmark  # noqa: E402
from qihc.orchestrator.bbh import BBHTask, load_bbh_tasks  # noqa: E402
from qihc.orchestrator.bbh_llm import enrich_problems_with_llm_logits  # noqa: E402

VCI_MODES = ["vci-0", "vci-1", "vci-2"]
CR_MODES = ["zeroshot", "random", "linear", "quadratic", "vci-1", "vci-2"]


def _load_problems(dataset: str, n_tasks: int, seed: int, logits: str, model_name: str):
    if dataset == "bundled":
        tasks = load_bbh_tasks(source="bundled")
        problems = [t.to_subset_problem(seed=seed + i) for i, t in enumerate(tasks)]
    else:
        problems = load_synthetic_problems(n_tasks=n_tasks, seed=seed)
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
    if logits == "llm":
        print(f"  Enriching {len(problems)} problems with {model_name} ...")
        problems = enrich_problems_with_llm_logits(problems, model_name=model_name)
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
    return problems, tasks


def _aggregate_seeds(seed_rows: list[dict[str, dict]]) -> dict[str, dict]:
    modes = seed_rows[0].keys()
    out: dict[str, dict] = {}
    for mode in modes:
        feas = [r[mode]["feasible_rate"] for r in seed_rows]
        exact = [r[mode].get("exact_match_rate", r[mode].get("accuracy", 0)) for r in seed_rows]
        out[mode] = {
            "feasible_rate_mean": float(np.mean(feas)),
            "feasible_rate_std": float(np.std(feas)),
            "exact_match_rate_mean": float(np.mean(exact)),
            "exact_match_rate_std": float(np.std(exact)),
            "n_seeds": len(seed_rows),
        }
    return out


def run_one_seed(
    problems,
    tasks,
    budget_steps: int,
    n_samples: int,
    seed: int,
    use_llm: bool,
    model_name: str,
) -> dict[str, dict[str, Any]]:
    vci_summary = strip_per_task(run_vci_modes(problems, VCI_MODES, budget_steps, seed))
    cr_data = run_cr_benchmark(
        tasks,
        modes=CR_MODES,
        budget_steps=budget_steps,
        n_samples=n_samples,
        seed=seed,
        use_llm=use_llm,
        model_name=model_name,
    )
    cr_summary = {
        m: {
            "feasible_rate": s["feasible_rate"],
            "exact_match_rate": s["accuracy"],
            "mean_pbit_steps": s.get("mean_pbit_steps", 0),
            "llm_calls": n_samples if m not in ("zeroshot", "random") else 1,
        }
        for m, s in cr_data["summary"].items()
    }

    unified: dict[str, dict] = {}
    for m, s in vci_summary.items():
        unified[m] = {**s, "family": "vci", "exact_match_rate": s["exact_match_rate"]}
    for m, s in cr_summary.items():
        key = f"cr:{m}" if m in VCI_MODES else m
        unified[key] = {**s, "family": "cr"}
    return unified


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified VCI + CR ablation")
    parser.add_argument("--dataset", choices=["bundled", "synthetic"], default="bundled")
    parser.add_argument("--n-tasks", type=int, default=200)
    parser.add_argument("--n-samples", type=int, default=50, help="CR reason samples per question")
    parser.add_argument("--budget-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0, help="Single seed (ignored if --seeds set)")
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--logits", choices=["pseudo", "llm"], default="pseudo")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    seeds = args.seeds if args.seeds else [args.seed]
    out_dir = resolve_out_dir("unified_ablation", args.output_dir)
    use_llm = args.logits == "llm"

    per_seed: list[dict] = []
    for si, seed in enumerate(seeds):
        print(f"\n=== Seed {seed} ({si + 1}/{len(seeds)}) ===")
        problems, tasks = _load_problems(
            args.dataset, args.n_tasks, seed, args.logits, args.model_name
        )
        summary = run_one_seed(
            problems, tasks, args.budget_steps, args.n_samples, seed, use_llm, args.model_name
        )
        per_seed.append(summary)
        for mode, s in sorted(summary.items()):
            fr = s["feasible_rate"]
            em = s.get("exact_match_rate", 0)
            print(f"  {mode:14s} feas={fr:.2%} exact={em:.2%}")

    last = per_seed[-1]
    payload = {
        "experiment": "unified_ablation",
        "dataset": args.dataset,
        "logits": args.logits,
        "model_name": args.model_name if use_llm else None,
        "n_tasks": len(problems),
        "budget_steps": args.budget_steps,
        "n_samples": args.n_samples,
        "seeds": seeds,
        "modes_vci": VCI_MODES,
        "modes_cr": CR_MODES,
        "summary": last,
        "per_seed": per_seed,
        "aggregate": _aggregate_seeds(per_seed) if len(seeds) > 1 else None,
    }
    save_json(f"{out_dir}/unified_ablation.json", payload)

    plot_rows = {
        k: {
            "feasible_rate": v["feasible_rate"],
            "exact_match_rate": v.get("exact_match_rate", 0),
        }
        for k, v in last.items()
    }
    plot_dual_bars(
        plot_rows,
        f"{out_dir}/unified_ablation.png",
        f"Unified ablation ({args.dataset}, n={len(problems)}, seeds={seeds})",
    )
    print(f"\nSaved: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
