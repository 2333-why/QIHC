# -*- coding: utf-8 -*-
"""
Case D (P1): CR subtask breakdown — quadratic vs vci-2 by task_type (CR Fig.2 style).
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.nsfc_evidence.common import load_synthetic_problems, resolve_out_dir, save_json  # noqa: E402
from experiments.nsfc_evidence.run_cr_bbh import run_cr_benchmark  # noqa: E402
from qihc.orchestrator.bbh import BBHTask  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Case D: CR breakdown by task type")
    parser.add_argument("--n-tasks", type=int, default=200)
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--budget-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    out_dir = resolve_out_dir("case_d_cr_by_task", args.output_dir)
    problems = load_synthetic_problems(n_tasks=args.n_tasks, seed=args.seed)
    tasks = [
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
        for p in problems
    ]

    modes = ["quadratic", "vci-2"]
    data = run_cr_benchmark(
        tasks,
        modes=modes,
        budget_steps=args.budget_steps,
        n_samples=args.n_samples,
        seed=args.seed,
        use_llm=False,
        model_name="",
    )

    by_type: dict[str, dict[str, list]] = defaultdict(lambda: {m: [] for m in modes})
    for mode in modes:
        for row in data["per_task"][mode]:
            tid = row["task_id"]
            task_type = next((t.task_type for t in tasks if t.task_id == tid), "unknown")
            by_type[task_type][mode].append(row)

    breakdown = {}
    for task_type, mode_rows in by_type.items():
        breakdown[task_type] = {}
        for mode in modes:
            rows = mode_rows[mode]
            breakdown[task_type][mode] = {
                "accuracy": float(np.mean([r["correct"] for r in rows])),
                "feasible_rate": float(np.mean([r["feasible"] for r in rows])),
                "n": len(rows),
            }
        q_acc = breakdown[task_type]["quadratic"]["accuracy"]
        v_acc = breakdown[task_type]["vci-2"]["accuracy"]
        breakdown[task_type]["gain_acc"] = v_acc - q_acc
        q_feas = breakdown[task_type]["quadratic"]["feasible_rate"]
        v_feas = breakdown[task_type]["vci-2"]["feasible_rate"]
        breakdown[task_type]["gain_feas"] = v_feas - q_feas

    payload = {
        "experiment": "case_d_cr_by_task",
        "case": "P1-D",
        "description": "CR subtask breakdown: quadratic vs vci-2",
        "n_tasks": len(tasks),
        "breakdown": breakdown,
    }
    save_json(f"{out_dir}/cr_by_task.json", payload)

    types = sorted(breakdown.keys())
    x = np.arange(len(types))
    w = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=120)
    for ax, metric, title in [
        (axes[0], "accuracy", "Accuracy by task type"),
        (axes[1], "feasible_rate", "Feasibility by task type"),
    ]:
        ax.bar(x - w / 2, [breakdown[t]["quadratic"][metric] for t in types], w, label="quadratic", color="#4c78a8")
        ax.bar(x + w / 2, [breakdown[t]["vci-2"][metric] for t in types], w, label="vci-2", color="#72b7b2")
        ax.set_xticks(x)
        ax.set_xticklabels(types, rotation=20, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/cr_by_task.png", bbox_inches="tight")
    plt.close()

    print(f"\n=== Case D CR by task type ===")
    for t in types:
        b = breakdown[t]
        print(
            f"  {t:28s} quad acc={b['quadratic']['accuracy']:.2%} "
            f"vci-2 acc={b['vci-2']['accuracy']:.2%} gain={b['gain_acc']:+.2%}"
        )
    print(f"Saved: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
