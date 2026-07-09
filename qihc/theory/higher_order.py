"""Higher-order soft division vs quadratization utilities (Theorems 1 / 1b)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from qihc.theory.mean_field import effective_field, mean_field_fixed_point, mean_field_iterate


@dataclass
class HigherOrderInstance:
    """Synthetic higher-order Ising / HUBO instance for division-law experiments."""

    n: int
    field: np.ndarray
    weight: np.ndarray
    higher_order: list[tuple[tuple[int, ...], float]]
    top_k: int = 3
    exclusion_pairs: list[tuple[int, int]] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def generate_higher_order_instance(
    n: int = 8,
    top_k: int = 3,
    n_ho_terms: int = 5,
    degree: int = 3,
    seed: int = 0,
    field_scale: float = 1.0,
    j_scale: float = 0.3,
    alpha_scale: float = 0.4,
) -> HigherOrderInstance:
    """Generate a small higher-order constrained subset instance."""
    rng = np.random.default_rng(seed)
    field = rng.normal(0.0, field_scale, size=n)
    # Prefer a planted gold set
    gold = sorted(rng.choice(n, size=top_k, replace=False).tolist())
    for i in gold:
        field[i] += 1.2

    weight = np.zeros((n, n), dtype=float)
    # Mild pairwise couplings + cardinality soft penalty encoded as all-pairs
    p = 1.5
    for i in range(n):
        for j in range(i + 1, n):
            w = -p + rng.normal(0.0, j_scale)
            weight[i, j] = w
            weight[j, i] = w

    higher_order: list[tuple[tuple[int, ...], float]] = []
    attempts = 0
    while len(higher_order) < n_ho_terms and attempts < 200:
        idxs = tuple(sorted(rng.choice(n, size=degree, replace=False).tolist()))
        if any(t[0] == idxs for t in higher_order):
            attempts += 1
            continue
        # Positive α encourages co-activation of the clique (coherence)
        alpha = float(rng.uniform(0.2, 1.0) * alpha_scale)
        # Prefer terms overlapping gold to create meaningful higher-order signal
        if len(set(idxs) & set(gold)) >= 2:
            alpha *= 1.5
        higher_order.append((idxs, alpha))
        attempts += 1

    # One exclusion pair outside gold if possible
    exclusion_pairs: list[tuple[int, int]] = []
    non_gold = [i for i in range(n) if i not in gold]
    if len(non_gold) >= 2:
        a, b = sorted(rng.choice(non_gold, size=2, replace=False).tolist())
        exclusion_pairs.append((int(a), int(b)))
        weight[a, b] -= 2.0
        weight[b, a] -= 2.0

    return HigherOrderInstance(
        n=n,
        field=field,
        weight=weight,
        higher_order=higher_order,
        top_k=top_k,
        exclusion_pairs=exclusion_pairs,
        metadata={"gold": gold, "degree": degree, "n_ho_terms": len(higher_order)},
    )


def soft_effective_field(m: np.ndarray, inst: HigherOrderInstance) -> np.ndarray:
    return effective_field(m, inst.field, inst.weight, inst.higher_order)


def soft_mean_field_iterate(
    inst: HigherOrderInstance,
    beta: float = 1.0,
    m0: np.ndarray | None = None,
    max_iters: int = 100,
) -> tuple[np.ndarray, list[float]]:
    m, residuals, _ = mean_field_fixed_point(
        inst.field,
        inst.weight,
        beta=beta,
        higher_order=inst.higher_order,
        m0=m0,
        max_iters=max_iters,
    )
    return m, residuals


def _reduce_cubic_to_quadratic(
    weight: np.ndarray,
    field: np.ndarray,
    i: int,
    j: int,
    k: int,
    alpha: float,
    aux_idx: int,
    M: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rosenberg-style reduction for α * x_i x_j x_k with x=(1+s)/2.

    Introduces auxiliary y ≈ x_i x_j, then α y x_k, with penalty
    enforcing y = x_i x_j. Implemented in spin variables approximately.
    """
    n = weight.shape[0]
    assert aux_idx < n
    # Soft penalty encouraging aux ≈ AND(i,j): reward aux when both i,j on
    # Using spin couplings (heuristic but standard for experiments):
    # Encourage s_aux ≈ s_i, s_aux ≈ s_j when both positive.
    weight[i, aux_idx] += 0.5 * M
    weight[aux_idx, i] += 0.5 * M
    weight[j, aux_idx] += 0.5 * M
    weight[aux_idx, j] += 0.5 * M
    weight[i, j] -= 0.25 * M
    weight[j, i] -= 0.25 * M
    field[aux_idx] -= 0.5 * M
    # Couple aux with k for the cubic term (α > 0 encourages co-activation)
    weight[aux_idx, k] += 0.5 * alpha
    weight[k, aux_idx] += 0.5 * alpha
    return weight, field


def quadratize_higher_order(
    inst: HigherOrderInstance,
    M: float = 4.0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Reduce higher-order terms to quadratic by introducing auxiliary spins.

    Returns (weight_ext, field_ext, n_aux).
    """
    n = inst.n
    terms = list(inst.higher_order)
    # Count auxiliaries: one per cubic; for degree 4 use two auxiliaries
    n_aux = 0
    for idxs, _ in terms:
        d = len(idxs)
        n_aux += max(d - 2, 0)

    n_ext = n + n_aux
    weight = np.zeros((n_ext, n_ext), dtype=float)
    weight[:n, :n] = inst.weight.copy()
    field = np.zeros(n_ext, dtype=float)
    field[:n] = inst.field.copy()

    aux = n
    for idxs, alpha in terms:
        idxs = list(idxs)
        while len(idxs) > 2:
            i, j = idxs[0], idxs[1]
            # Create aux for AND(i,j), replace with [aux] + rest
            k_placeholder = idxs[2] if len(idxs) > 2 else idxs[0]
            weight, field = _reduce_cubic_to_quadratic(
                weight, field, i, j, k_placeholder if len(idxs) == 3 else aux, alpha if len(idxs) == 3 else 0.0, aux, M=M
            )
            if len(idxs) == 3:
                # already coupled aux-k above
                idxs = []
            else:
                # For degree>3: replace (i,j) by aux and continue with remaining + aux
                idxs = [aux] + idxs[2:]
            aux += 1
            if not idxs:
                break
        if len(idxs) == 2:
            a, b = idxs
            weight[a, b] += 0.5 * alpha
            weight[b, a] += 0.5 * alpha

    return weight, field, n_aux


def hubo_energy(spins: np.ndarray, inst: HigherOrderInstance) -> float:
    """Evaluate full higher-order energy on ±1 spins (no auxiliaries)."""
    s = np.asarray(spins, dtype=float).ravel()
    e = -0.5 * float(s @ inst.weight @ s) - float(np.dot(inst.field, s))
    # Convert to x=(1+s)/2 for higher-order product terms
    x = 0.5 * (1.0 + s)
    for idxs, alpha in inst.higher_order:
        e -= float(alpha) * float(np.prod(x[list(idxs)]))
    return e


def quadratic_energy(spins: np.ndarray, weight: np.ndarray, field: np.ndarray) -> float:
    s = np.asarray(spins, dtype=float).ravel()
    return -0.5 * float(s @ weight @ s) - float(np.dot(field, s))


def magnetizations_to_mask(m: np.ndarray, top_k: int) -> np.ndarray:
    m = np.asarray(m, dtype=float).ravel()
    idx = np.argsort(-m)[:top_k]
    mask = np.zeros(m.size, dtype=bool)
    mask[idx] = True
    return mask


# Re-export iterate for callers that import from this module
__all__ = [
    "HigherOrderInstance",
    "generate_higher_order_instance",
    "soft_effective_field",
    "soft_mean_field_iterate",
    "quadratize_higher_order",
    "hubo_energy",
    "quadratic_energy",
    "magnetizations_to_mask",
    "mean_field_iterate",
]
