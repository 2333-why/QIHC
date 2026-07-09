"""Shared helpers for QIHC three-law (NE1–NE9) experiments."""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from qihc.orchestrator.bbh import evaluate_prediction, load_bbh_problems  # noqa: E402
from qihc.orchestrator.constrained_data import load_synthetic_tasks  # noqa: E402
from qihc.orchestrator.encoder import refine_mask_to_top_k  # noqa: E402
from qihc.orchestrator.reasoning import is_feasible, subset_to_ising  # noqa: E402
from qihc.orchestrator.backend import PBitBackend  # noqa: E402
from qihc.orchestrator.vci_scheduler import VCIConfig, VCIOrchestrator  # noqa: E402


def resolve_out_dir(default_rel: str, output_dir: str | None) -> str:
    out = output_dir or os.path.join(REPO_ROOT, "experiments", "outputs", "nsfc_evidence", default_rel)
    if not os.path.isabs(out):
        out = os.path.join(REPO_ROOT, out)
    os.makedirs(out, exist_ok=True)
    return out


def save_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_problems(source: str = "bundled", n_tasks: int = 40, seed: int = 0):
    if source == "bundled":
        return load_bbh_problems(source="bundled", seed=seed)[:n_tasks]
    tasks = load_synthetic_tasks(n_tasks=n_tasks, seed=seed, regenerate=False)
    return [t.to_subset_problem(seed=seed + i) for i, t in enumerate(tasks)]


def solve_with_logits(
    problem,
    logits: np.ndarray,
    steps: int,
    seed: int,
    sampler: str = "parallel_tempering",
) -> dict[str, Any]:
    np.random.seed(seed)
    logits = np.asarray(logits, dtype=float).ravel()
    weight, field = subset_to_ising(
        logits,
        top_k=problem.top_k,
        exclusion_pairs=problem.exclusion_pairs,
    )
    cfg = VCIConfig.tier_a(sampling_steps=steps, sampler=sampler, seed=seed)  # type: ignore[arg-type]
    backend = PBitBackend(cfg)
    mask, energy, elapsed = backend.solve(weight, field)
    mask = refine_mask_to_top_k(logits, mask, problem.top_k)
    feasible = is_feasible(mask, problem.top_k, problem.exclusion_pairs)
    ev = evaluate_prediction(problem, mask)
    return {
        "feasible": bool(feasible),
        "exact_match": bool(ev.get("exact_match", False)),
        "jaccard": float(ev.get("jaccard", 0.0)),
        "energy": float(energy),
        "elapsed_s": float(elapsed),
        "mask": mask,
    }


def bootstrap_ci(values: list[float], n_boot: int = 500, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    means = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=arr.size, replace=True)
        means.append(float(sample.mean()))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(arr.mean()), float(lo), float(hi)


def save_fig(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
