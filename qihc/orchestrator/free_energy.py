"""Variational free-energy proxy for VCI (F(q, s))."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qihc.orchestrator.encoder import routing_score
from qihc.orchestrator.reasoning import count_exclusion_violations


@dataclass
class FreeEnergyResult:
    """Proxy for F(q, s) and diagnostic feedback for q-step refine."""

    total: float
    kl_term: float
    energy_term: float
    entropy_term: float
    violation_penalty: float
    semantic_score: float
    violations: int
    violated_pairs: list[tuple[int, int]]
    feedback: dict


def softmax_entropy(logits: np.ndarray) -> float:
    logits = np.asarray(logits, dtype=float).ravel()
    z = logits - logits.max()
    exp = np.exp(z)
    p = exp / exp.sum()
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


def kl_proxy(q_logits: np.ndarray, q0_logits: np.ndarray) -> float:
    """L2 drift from initial semantic prior (differentiable proxy for KL)."""
    q = np.asarray(q_logits, dtype=float).ravel()
    q0 = np.asarray(q0_logits, dtype=float).ravel()
    return float(np.mean((q - q0) ** 2))


def compute_free_energy(
    q_logits: np.ndarray,
    q0_logits: np.ndarray,
    mask: np.ndarray,
    ising_energy: float | None = None,
    exclusion_pairs: list[tuple[int, int]] | None = None,
    beta: float = 1.0,
    kl_weight: float = 0.5,
    violation_weight: float = 3.0,
    refine_penalty: float = 1.5,
) -> FreeEnergyResult:
    """
    Compute variational free-energy proxy:

        F ≈ kl_weight * KL_proxy(q, q0)
            + beta * E_Ising(s|q)
            - entropy(q)
            + violation_weight * (#violations)

    Lower is better. ``E_Ising`` uses backend energy if provided,
    otherwise falls back to negative routing score.
    """
    q_logits = np.asarray(q_logits, dtype=float).ravel()
    q0_logits = np.asarray(q0_logits, dtype=float).ravel()
    mask = np.asarray(mask, dtype=bool).ravel()

    kl = kl_proxy(q_logits, q0_logits)
    entropy = softmax_entropy(q_logits)
    semantic = routing_score(q_logits, mask)

    if ising_energy is not None:
        energy_term = float(ising_energy)
    else:
        energy_term = -semantic

    pairs = exclusion_pairs or []
    n_viol, violated = count_exclusion_violations(mask, pairs)
    viol_pen = float(violation_weight * n_viol)

    total = kl_weight * kl + beta * energy_term - entropy + viol_pen

    feedback = {
        "violation_pairs": violated,
        "violations": n_viol,
        "refine_penalty": refine_penalty,
        "semantic_score": semantic,
        "delta_kl": kl,
    }

    return FreeEnergyResult(
        total=total,
        kl_term=kl,
        energy_term=energy_term,
        entropy_term=entropy,
        violation_penalty=viol_pen,
        semantic_score=semantic,
        violations=n_viol,
        violated_pairs=violated,
        feedback=feedback,
    )
