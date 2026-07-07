"""Shared helpers for NSFC P0–P2 evidence experiments."""

from __future__ import annotations

import json
import os
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qihc.orchestrator.bbh import evaluate_prediction
from qihc.orchestrator.constrained_data import load_constrained_problems, load_synthetic_tasks
from qihc.orchestrator.vci_scheduler import VCIConfig, VCIOrchestrator


def repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def resolve_out_dir(default_rel: str, output_dir: str | None) -> str:
    root = repo_root()
    out = output_dir or os.path.join(root, "experiments", "outputs", "nsfc_evidence", default_rel)
    if not os.path.isabs(out):
        out = os.path.join(root, out)
    os.makedirs(out, exist_ok=True)
    return out


def save_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def problem_to_bbh_task(problem) -> "BBHTask":
    """Convert SubsetProblem to BBHTask for CR paper pipeline."""
    from qihc.orchestrator.bbh import BBHTask

    gold = list(problem.metadata.get("gold_indices", []))
    if not gold and problem.metadata.get("gold_mask") is not None:
        gold = list(np.flatnonzero(np.asarray(problem.metadata["gold_mask"], dtype=bool)))
    return BBHTask(
        task_id=str(problem.metadata.get("task_id", "unknown")),
        task_type=str(problem.metadata.get("task_type", "subset")),
        text=problem.text,
        candidates=list(problem.metadata.get("candidates", [])),
        top_k=problem.top_k,
        gold_indices=gold,
        exclusion_pairs=list(problem.exclusion_pairs),
        logits=np.asarray(problem.logits, dtype=float).copy(),
    )


def run_cr_paper_modes(
    problems,
    modes: list[str],
    budget_steps: int,
    n_samples: int,
    seed: int,
    use_llm: bool = False,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
) -> dict[str, dict[str, Any]]:
    """Run CR paper-aligned modes on SubsetProblem list."""
    from experiments.nsfc_evidence.run_cr_bbh import run_cr_benchmark

    tasks = [problem_to_bbh_task(p) for p in problems]
    data = run_cr_benchmark(
        tasks,
        modes=modes,  # type: ignore[arg-type]
        budget_steps=budget_steps,
        n_samples=n_samples,
        seed=seed,
        use_llm=use_llm,
        model_name=model_name,
    )
    summary: dict[str, dict] = {}
    for m, s in data["summary"].items():
        summary[m] = {
            "mode": m,
            "accuracy": s["accuracy"],
            "feasible_rate": s["feasible_rate"],
            "exact_match_rate": s["accuracy"],
            "gain_over_zeroshot": s.get("gain_over_zeroshot", 0.0),
            "mean_pbit_steps": s.get("mean_pbit_steps", 0),
            "mean_llm_calls": s.get("mean_llm_calls", 0),
            "llm_calls": int(round(s.get("mean_llm_calls", 0))),
        }
    return summary


def run_vci_modes(
    problems,
    modes: list[str],
    budget_steps: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for mode in modes:
        if mode in ("greedy", "vci-0"):
            steps, max_rounds = 0, 1
        elif mode == "vci-1":
            steps, max_rounds = budget_steps, 1
        elif mode == "vci-2":
            steps, max_rounds = max(budget_steps // 2, 50), 2
        elif mode == "vci-full":
            steps, max_rounds = max(budget_steps // 2, 50), 4
        else:
            raise ValueError(mode)

        cfg = VCIConfig.tier_a(sampling_steps=steps, max_rounds=max_rounds, seed=seed)
        orch = VCIOrchestrator(cfg)
        feasible, exact, jaccard, f_vals, rounds, times = [], [], [], [], [], []
        per_task: list[dict] = []

        for p in problems:
            res = orch.solve_subset(p, mode=mode)  # type: ignore[arg-type]
            ev = evaluate_prediction(p, res.final_mask)
            feasible.append(ev["feasible"])
            exact.append(ev.get("exact_match", False))
            jaccard.append(ev.get("jaccard", 0.0))
            f_vals.append(res.final_free_energy)
            rounds.append(res.n_rounds)
            times.append(res.total_elapsed_s)
            per_task.append(
                {
                    "task_id": p.metadata.get("task_id"),
                    "task_type": p.metadata.get("task_type"),
                    "feasible": ev["feasible"],
                    "exact_match": ev.get("exact_match"),
                    "jaccard": ev.get("jaccard"),
                    "F_final": res.final_free_energy,
                    "n_rounds": res.n_rounds,
                    "pbit_steps": steps * res.n_rounds if mode not in ("greedy", "vci-0") else 0,
                }
            )

        llm_calls = 1
        mean_pbit = steps * float(np.mean(rounds)) if mode not in ("greedy", "vci-0") else 0.0
        summary[mode] = {
            "mode": mode,
            "feasible_rate": float(np.mean(feasible)),
            "exact_match_rate": float(np.mean(exact)),
            "mean_jaccard": float(np.mean(jaccard)),
            "mean_F": float(np.mean(f_vals)),
            "mean_time_s": float(np.mean(times)),
            "mean_rounds": float(np.mean(rounds)),
            "steps_per_round": steps,
            "mean_pbit_steps": mean_pbit,
            "llm_calls": llm_calls,
            "per_task": per_task,
        }
    return summary


def strip_per_task(summary: dict[str, dict]) -> dict[str, dict]:
    return {k: {kk: vv for kk, vv in v.items() if kk != "per_task"} for k, v in summary.items()}


def plot_dual_bars(summary: dict, out_path: str, title: str) -> None:
    modes = list(summary.keys())
    x = np.arange(len(modes))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=120)
    ax.bar(x - w / 2, [summary[m]["feasible_rate"] for m in modes], w, label="Feasible", color="#72b7b2")
    ax.bar(
        x + w / 2,
        [summary[m].get("exact_match_rate", summary[m].get("accuracy", 0)) for m in modes],
        w,
        label="Exact/Acc",
        color="#4c78a8",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(modes, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def load_synthetic_problems(n_tasks: int = 200, seed: int = 0, regenerate: bool = False):
    tasks = load_synthetic_tasks(n_tasks=n_tasks, seed=seed, regenerate=regenerate)
    return [t.to_subset_problem(seed=seed + i) for i, t in enumerate(tasks)]
