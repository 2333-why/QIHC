"""Mean-field free energy and contraction utilities (Theorems 1c / 4)."""

from __future__ import annotations

import numpy as np


def binary_entropy(m: np.ndarray) -> float:
    """Sum of binary entropies for magnetizations in (-1, 1)."""
    m = np.clip(np.asarray(m, dtype=float).ravel(), -1.0 + 1e-8, 1.0 - 1e-8)
    p = 0.5 * (1.0 + m)
    q = 1.0 - p
    return float(-np.sum(p * np.log(p) + q * np.log(q)))


def free_energy_mean_field(
    m: np.ndarray,
    field: np.ndarray,
    weight: np.ndarray,
    beta: float = 1.0,
    higher_order: list[tuple[tuple[int, ...], float]] | None = None,
) -> float:
    """
    Mean-field free energy:

        F = -h·m - 0.5 m^T J m - Σ α_S ∏ m_i + (1/β) Σ H(m_i)
    """
    m = np.asarray(m, dtype=float).ravel()
    field = np.asarray(field, dtype=float).ravel()
    weight = np.asarray(weight, dtype=float)
    energy = -float(np.dot(field, m)) - 0.5 * float(m @ weight @ m)
    if higher_order:
        for idxs, alpha in higher_order:
            energy -= float(alpha) * float(np.prod(m[list(idxs)]))
    return energy + (1.0 / max(beta, 1e-12)) * binary_entropy(m)


def soft_g_term(
    m: np.ndarray,
    higher_order: list[tuple[tuple[int, ...], float]] | None,
) -> np.ndarray:
    """Mean-field contribution of higher-order terms to effective field."""
    m = np.asarray(m, dtype=float).ravel()
    g = np.zeros_like(m)
    if not higher_order:
        return g
    for idxs, alpha in higher_order:
        idxs = tuple(int(i) for i in idxs)
        for i in idxs:
            others = [j for j in idxs if j != i]
            g[i] += float(alpha) * float(np.prod(m[others])) if others else float(alpha)
    return g


def effective_field(
    m: np.ndarray,
    field: np.ndarray,
    weight: np.ndarray,
    higher_order: list[tuple[tuple[int, ...], float]] | None = None,
) -> np.ndarray:
    m = np.asarray(m, dtype=float).ravel()
    field = np.asarray(field, dtype=float).ravel()
    weight = np.asarray(weight, dtype=float)
    return field + weight @ m + soft_g_term(m, higher_order)


def mean_field_iterate(
    m: np.ndarray,
    field: np.ndarray,
    weight: np.ndarray,
    beta: float = 1.0,
    higher_order: list[tuple[tuple[int, ...], float]] | None = None,
) -> np.ndarray:
    """One mean-field update: m <- tanh(β h_eff(m))."""
    h = effective_field(m, field, weight, higher_order)
    return np.tanh(beta * h)


def mean_field_fixed_point(
    field: np.ndarray,
    weight: np.ndarray,
    beta: float = 1.0,
    higher_order: list[tuple[tuple[int, ...], float]] | None = None,
    m0: np.ndarray | None = None,
    max_iters: int = 200,
    tol: float = 1e-8,
) -> tuple[np.ndarray, list[float], list[float]]:
    """
    Iterate mean-field map to a fixed point.

    Returns
    -------
    m_star, residual_history, free_energy_history
    """
    n = np.asarray(field).size
    m = np.zeros(n) if m0 is None else np.asarray(m0, dtype=float).ravel().copy()
    residuals: list[float] = []
    energies: list[float] = []
    for _ in range(max_iters):
        m_next = mean_field_iterate(m, field, weight, beta=beta, higher_order=higher_order)
        res = float(np.max(np.abs(m_next - m)))
        residuals.append(res)
        energies.append(free_energy_mean_field(m_next, field, weight, beta=beta, higher_order=higher_order))
        m = m_next
        if res < tol:
            break
    return m, residuals, energies


def contraction_lipschitz(
    weight: np.ndarray,
    higher_order: list[tuple[tuple[int, ...], float]] | None = None,
) -> float:
    """
    Lipschitz constant L of h_eff (Theorem 1c):

        L = ||J||_∞ + max_i Σ_{S∋i} |α_S| (|S|-1)
    """
    weight = np.asarray(weight, dtype=float)
    j_inf = float(np.max(np.sum(np.abs(weight), axis=1))) if weight.size else 0.0
    g_bound = 0.0
    if higher_order:
        n = weight.shape[0]
        per_i = np.zeros(n)
        for idxs, alpha in higher_order:
            d = len(idxs)
            for i in idxs:
                per_i[i] += abs(float(alpha)) * max(d - 1, 0)
        g_bound = float(np.max(per_i)) if n else 0.0
    return j_inf + g_bound
