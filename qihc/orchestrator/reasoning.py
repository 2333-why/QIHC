"""Subset-selection / reasoning tasks for VCI (Case A)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from qihc.orchestrator.encoder import moe_logits_to_ising


@dataclass
class SubsetProblem:
    """Pick top-k candidates with optional mutual-exclusion constraints."""

    text: str
    logits: np.ndarray
    top_k: int
    exclusion_pairs: list[tuple[int, int]] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def subset_to_ising(
    logits: np.ndarray,
    top_k: int,
    cardinality_penalty: float = 2.0,
    exclusion_pairs: list[tuple[int, int]] | None = None,
    exclusion_penalty: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Encode subset selection as Ising model.

    Adds soft mutual-exclusion coupling on ``exclusion_pairs``:
    penalizes selecting both endpoints simultaneously.
    """
    weight, field = moe_logits_to_ising(logits, top_k=top_k, penalty=cardinality_penalty)
    if not exclusion_pairs:
        return weight, field

    logits = np.asarray(logits, dtype=float).ravel()
    n = logits.size
    pair_coupling = -0.5 * float(exclusion_penalty)
    for i, j in exclusion_pairs:
        if 0 <= i < n and 0 <= j < n and i != j:
            weight[i, j] += pair_coupling
            weight[j, i] += pair_coupling
    return weight, field


def count_exclusion_violations(
    mask: np.ndarray,
    exclusion_pairs: list[tuple[int, int]],
) -> tuple[int, list[tuple[int, int]]]:
    """Return violation count and list of violated pairs."""
    mask = np.asarray(mask, dtype=bool).ravel()
    violated: list[tuple[int, int]] = []
    for i, j in exclusion_pairs:
        if mask[i] and mask[j]:
            violated.append((int(i), int(j)))
    return len(violated), violated


def is_feasible(mask: np.ndarray, top_k: int, exclusion_pairs: list[tuple[int, int]]) -> bool:
    mask = np.asarray(mask, dtype=bool).ravel()
    if int(mask.sum()) != top_k:
        return False
    return count_exclusion_violations(mask, exclusion_pairs)[0] == 0


def generate_toy_problems(
    n_problems: int = 32,
    n_candidates: int = 6,
    top_k: int = 3,
    n_exclusion_pairs: int = 1,
    seed: int = 0,
) -> list[SubsetProblem]:
    """Synthetic Case-A instances with random logits and exclusion constraints."""
    rng = np.random.default_rng(seed)
    problems: list[SubsetProblem] = []
    for idx in range(n_problems):
        logits = rng.normal(loc=0.0, scale=1.0, size=n_candidates)
        pairs: list[tuple[int, int]] = []
        for _ in range(n_exclusion_pairs):
            i, j = rng.choice(n_candidates, size=2, replace=False)
            if i > j:
                i, j = j, i
            if (i, j) not in pairs:
                pairs.append((int(i), int(j)))
        problems.append(
            SubsetProblem(
                text=f"toy_reasoning_{idx}",
                logits=logits,
                top_k=top_k,
                exclusion_pairs=pairs,
                metadata={"index": idx},
            )
        )
    return problems


def demo_problem() -> SubsetProblem:
    """Fixed 6-reason / k=3 example from project outline (pair 2,5 exclusive)."""
    logits = np.array([2.1, 1.8, 2.5, 1.2, 0.9, 2.4], dtype=float)
    return SubsetProblem(
        text="demo_select_reasons",
        logits=logits,
        top_k=3,
        exclusion_pairs=[(2, 5)],
        metadata={"demo": True},
    )
