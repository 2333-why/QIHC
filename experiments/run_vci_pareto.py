# -*- coding: utf-8 -*-
"""
Pareto frontier: feasible rate / exact match / latency vs p-bit budget.

Includes VCI modes and CR linear/quadratic baselines at matched budget.

Usage:
    python experiments/run_vci_pareto.py
    python experiments/run_vci_pareto.py --budgets 100 200 400 600 --include-cr
    python experiments/run_vci_pareto.py --logits llm --model-name Qwen/Qwen2.5-7B-Instruct
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from qihc.orchestrator.bbh import BBHTask, evaluate_prediction, load_bbh_problems, load_bbh_tasks  # noqa: E402
from qihc.orchestrator.bbh_llm import enrich_problems_with_llm_logits  # noqa: E402
from qihc.orchestrator.vci_scheduler import VCIConfig, VCIOrchestrator  # noqa: E402
from experiments.nsfc_evidence.run_cr_bbh import run_cr_benchmark  # noqa: E402

VCI_MODES = ("vci-1", "vci-2")
CR_MODES = ("zeroshot", "linear", "quadratic")


def run_vci_point(problems, mode: str, budget_steps: int, seed: int) -> dict:
    if mode == "greedy":
        steps, max_rounds = 0, 1
    elif mode == "vci-1":
        steps, max_rounds = budget_steps, 1
    elif mode == "vci-2":
        steps, max_rounds = max(budget_steps // 2, 50), 2
    else:
        raise ValueError(mode)

    cfg = VCIConfig.tier_a(sampling_steps=steps, max_rounds=max_rounds, seed=seed)
    orch = VCIOrchestrator(cfg)

    feasible, exact, times, f_vals = [], [], [], []
    for p in problems:
        res = orch.solve_subset(p, mode=mode)  # type: ignore[arg-type]
        ev = evaluate_prediction(p, res.final_mask)
        feasible.append(ev["feasible"])
        exact.append(ev.get("exact_match", False))
        times.append(res.total_elapsed_s)
        f_vals.append(res.final_free_energy)

    mean_rounds = 1.0 if mode == "greedy" else (1.0 if mode == "vci-1" else 2.0)
    pbit_steps = steps * mean_rounds if mode != "greedy" else 0.0

    return {
        "mode": mode,
        "family": "vci",
        "budget_steps": budget_steps,
        "feasible_rate": float(np.mean(feasible)),
        "exact_match_rate": float(np.mean(exact)),
        "mean_time_s": float(np.mean(times)),
        "mean_free_energy": float(np.mean(f_vals)),
        "mean_pbit_steps": pbit_steps,
    }


def run_cr_point(
    tasks,
    mode: str,
    budget_steps: int,
    n_samples: int,
    seed: int,
    use_llm: bool = False,
    model_name: str = "",
) -> dict:
    data = run_cr_benchmark(
        tasks,
        modes=[mode],  # type: ignore[list-item]
        budget_steps=budget_steps,
        n_samples=n_samples,
        seed=seed,
        use_llm=use_llm,
        model_name=model_name,
    )
    s = data["summary"][mode]
    pbit = 0.0 if mode == "zeroshot" else s.get("mean_pbit_steps", float(budget_steps))
    return {
        "mode": mode if mode.startswith("cr") else mode,
        "family": "cr",
        "budget_steps": budget_steps,
        "feasible_rate": s["feasible_rate"],
        "exact_match_rate": s["accuracy"],
        "gain_over_zeroshot": s.get("gain_over_zeroshot", 0.0),
        "mean_time_s": float("nan"),
        "mean_free_energy": float("nan"),
        "mean_pbit_steps": pbit,
    }


def plot_pareto(rows: list[dict], out_path: str) -> None:
    colors = {
        "zeroshot": "#e45756",
        "vci-1": "#f58518",
        "vci-2": "#4c78a8",
        "linear": "#54a24b",
        "quadratic": "#b279a2",
    }
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=120)
    modes = sorted({r["mode"] for r in rows})

    for mode in modes:
        pts = [r for r in rows if r["mode"] == mode]
        pts.sort(key=lambda r: r["mean_pbit_steps"])
        x = [r["mean_pbit_steps"] for r in pts]
        c = colors.get(mode, "#888888")
        axes[0].plot(x, [r["feasible_rate"] for r in pts], "o-", label=mode, color=c)
        axes[1].plot(x, [r["exact_match_rate"] for r in pts], "o-", label=mode, color=c)
        valid_f = [r for r in pts if not np.isnan(r["mean_free_energy"])]
        if valid_f:
            axes[2].plot(
                [r["mean_pbit_steps"] for r in valid_f],
                [r["mean_free_energy"] for r in valid_f],
                "o-",
                label=mode,
                color=c,
            )

    for ax, ylab in zip(axes, ["Feasible rate", "Exact match rate", "Mean F(q,s)"]):
        ax.set_xlabel("Mean p-bit steps")
        ax.set_ylabel(ylab)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    axes[0].set_title("Feasible vs budget")
    axes[1].set_title("Accuracy vs budget")
    axes[2].set_title("Free energy vs budget")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_tradeoff_scatter(rows: list[dict], out_path: str) -> None:
    colors = {
        "zeroshot": "#e45756",
        "vci-1": "#f58518",
        "vci-2": "#4c78a8",
        "linear": "#54a24b",
        "quadratic": "#b279a2",
    }
    markers = {
        "zeroshot": "x",
        "vci-1": "s",
        "vci-2": "o",
        "linear": "^",
        "quadratic": "D",
    }

    plt.figure(figsize=(6.5, 5), dpi=120)
    for r in rows:
        mode = r["mode"]
        score = 0.5 * r["feasible_rate"] + 0.5 * r["exact_match_rate"]
        lat_ms = r["mean_time_s"] * 1000 if not np.isnan(r["mean_time_s"]) else r["mean_pbit_steps"]
        plt.scatter(
            lat_ms,
            score,
            c=colors.get(mode, "#888"),
            marker=markers.get(mode, "o"),
            s=60,
            alpha=0.85,
        )
    for mode in sorted({r["mode"] for r in rows}):
        plt.scatter([], [], c=colors.get(mode, "#888"), marker=markers.get(mode, "o"), label=mode, s=60)
    plt.xlabel("Latency proxy (ms or p-bit steps)")
    plt.ylabel("Combined score (0.5·feas + 0.5·exact)")
    plt.title("Pareto: quality vs cost (bundled BBH)")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="VCI + CR Pareto frontier")
    parser.add_argument("--source", choices=["bundled"], default="bundled")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--budgets", type=int, nargs="+", default=[50, 100, 150, 200, 300, 400, 600])
    parser.add_argument("--n-samples", type=int, default=50, help="CR mock samples per question")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-cr", action="store_true", default=True)
    parser.add_argument("--no-cr", action="store_true", help="Skip CR baselines")
    parser.add_argument("--use-llm", action="store_true", help="Real LLM for CR paper modes")
    parser.add_argument("--logits", choices=["pseudo", "llm"], default="pseudo")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    include_cr = args.include_cr and not args.no_cr
    use_llm = args.use_llm or args.logits == "llm"

    out_dir = args.output_dir or os.path.join("experiments", "outputs", "vci_pareto")
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(REPO_ROOT, out_dir)

    problems = load_bbh_problems(source=args.source, seed=args.seed, limit=args.limit)
    if use_llm:
        problems = enrich_problems_with_llm_logits(problems, model_name=args.model_name)
    tasks = load_bbh_tasks(source="bundled")[: len(problems)]
    tasks = [
        BBHTask(
            task_id=str(p.metadata["task_id"]),
            task_type=str(p.metadata.get("task_type", "bundled")),
            text=p.text,
            candidates=list(p.metadata.get("candidates", [])),
            top_k=p.top_k,
            gold_indices=list(p.metadata.get("gold_indices", [])),
            exclusion_pairs=list(p.exclusion_pairs),
            logits=p.logits.copy(),
        )
        for p in problems
    ]

    rows: list[dict] = []
    for budget in args.budgets:
        for mode in VCI_MODES:
            rows.append(run_vci_point(problems, mode, budget, args.seed))
            r = rows[-1]
            print(
                f"  budget={budget:3d} {mode:8s}  "
                f"feas={r['feasible_rate']:.2%} exact={r['exact_match_rate']:.2%}  "
                f"steps={r['mean_pbit_steps']:.0f}"
            )
        if include_cr:
            for mode in CR_MODES:
                budget = 0 if mode == "zeroshot" else budget
                rows.append(
                    run_cr_point(
                        tasks,
                        mode,
                        budget,
                        args.n_samples,
                        args.seed,
                        use_llm=use_llm,
                        model_name=args.model_name,
                    )
                )
                r = rows[-1]
                print(
                    f"  budget={budget:3d} {r['mode']:12s}  "
                    f"feas={r['feasible_rate']:.2%} exact={r['exact_match_rate']:.2%}"
                )

    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "pareto.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source": args.source,
                "logits": args.logits,
                "use_llm": use_llm,
                "model_name": args.model_name if use_llm else None,
                "baseline": "zeroshot (CR paper)",
                "n_problems": len(problems),
                "budgets": args.budgets,
                "include_cr": include_cr,
                "rows": rows,
            },
            f,
            indent=2,
        )
    print(f"Saved: {json_path}")

    plot_pareto(rows, os.path.join(out_dir, "pareto_curves.png"))
    plot_tradeoff_scatter(rows, os.path.join(out_dir, "pareto_tradeoff.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
