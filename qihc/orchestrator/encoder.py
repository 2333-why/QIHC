"""MoE routing logits -> Ising / QUBO encoding."""

from __future__ import annotations

import numpy as np


def moe_logits_to_ising(
    logits: np.ndarray,
    top_k: int,
    penalty: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Encode top-k expert routing as an Ising model (Weight, Field).

    Binary activation x_i = (1 + s_i) / 2, s_i in {-1, +1}.

    QUBO objective (minimize):
        -sum_i logits_i * x_i + penalty * (sum_i x_i - top_k)^2

    Returns
    -------
    Weight : (n, n) symmetric coupling matrix
    Field  : (n,) external field
    """
    logits = np.asarray(logits, dtype=float).ravel()
    n = logits.size
    if not 0 < top_k < n:
        raise ValueError(f"top_k must satisfy 0 < top_k < num_experts, got {top_k}/{n}")

    p = float(penalty)
    k = int(top_k)

    # Linear coefficients on x_i after expanding cardinality penalty.
    c = -logits + p - 2.0 * p * k
    field = -c / 2.0

    weight = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            weight[i, j] = -p
            weight[j, i] = -p

    return weight, field


def mask_to_expert_indices(mask: np.ndarray) -> list[int]:
    return [int(i) for i in np.flatnonzero(mask)]


def greedy_top_k(logits: np.ndarray, top_k: int) -> np.ndarray:
    """Deterministic top-k expert mask."""
    logits = np.asarray(logits, dtype=float).ravel()
    indices = np.argsort(-logits)[:top_k]
    mask = np.zeros(logits.size, dtype=bool)
    mask[indices] = True
    return mask


def refine_mask_to_top_k(logits: np.ndarray, mask: np.ndarray, top_k: int) -> np.ndarray:
    """Project a mask to exactly top_k experts, preferring higher logits."""
    logits = np.asarray(logits, dtype=float).ravel()
    mask = np.asarray(mask, dtype=bool).ravel()
    selected = list(np.flatnonzero(mask))
    if len(selected) > top_k:
        selected = sorted(selected, key=lambda i: -logits[i])[:top_k]
    elif len(selected) < top_k:
        remaining = [i for i in range(logits.size) if i not in selected]
        extra = sorted(remaining, key=lambda i: -logits[i])[: top_k - len(selected)]
        selected.extend(extra)
    out = np.zeros(logits.size, dtype=bool)
    out[selected] = True
    return out


def batch_moe_logits_to_ising(
    logits_batch: np.ndarray,
    top_k: int,
    expert_capacity: int,
    row_penalty: float = 2.0,
    col_penalty: float = 6.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Joint batch encoding: row cardinality + per-expert capacity.

    Variables x_{r,e} flattened as index r * E + e.
    Minimize routing loss plus soft row/column penalties.
    """
    logits_batch = np.asarray(logits_batch, dtype=float)
    if logits_batch.ndim != 2:
        raise ValueError("logits_batch must be 2-D (requests, experts)")
    n_requests, n_experts = logits_batch.shape
    if not 0 < top_k < n_experts:
        raise ValueError(f"top_k must satisfy 0 < top_k < num_experts, got {top_k}/{n_experts}")

    p_row = float(row_penalty)
    p_col = float(col_penalty)
    k = int(top_k)
    cap = int(expert_capacity)
    n = n_requests * n_experts

    def idx(r: int, e: int) -> int:
        return r * n_experts + e

    field = np.zeros(n, dtype=float)
    weight = np.zeros((n, n), dtype=float)

    for r in range(n_requests):
        for e in range(n_experts):
            i = idx(r, e)
            c = (
                -logits_batch[r, e]
                + p_row * (1.0 - 2.0 * k)
                + p_col * (1.0 - 2.0 * cap)
            )
            field[i] = -c / 2.0

    for r in range(n_requests):
        for e1 in range(n_experts):
            for e2 in range(e1 + 1, n_experts):
                i, j = idx(r, e1), idx(r, e2)
                weight[i, j] -= p_row
                weight[j, i] -= p_row

    for e in range(n_experts):
        for r1 in range(n_requests):
            for r2 in range(r1 + 1, n_requests):
                i, j = idx(r1, e), idx(r2, e)
                weight[i, j] -= p_col
                weight[j, i] -= p_col

    return weight, field


def spins_to_bias_matrix(
    spins: dict[int, int] | np.ndarray,
    n_requests: int,
    n_experts: int,
) -> np.ndarray:
    """Reshape flattened spins into (requests, experts) bias matrix."""
    if isinstance(spins, dict):
        size = n_requests * n_experts
        arr = np.ones(size, dtype=float)
        for i, s in spins.items():
            if int(i) < size:
                arr[int(i)] = float(s)
    else:
        arr = np.asarray(spins, dtype=float).ravel()
    return arr.reshape(n_requests, n_experts)


def sequential_capacity_decode(
    logits_batch: np.ndarray,
    top_k: int,
    expert_capacity: int,
    request_order: np.ndarray | None = None,
    spin_matrix: np.ndarray | None = None,
    spin_weight: float = 0.25,
) -> list[np.ndarray]:
    """
    Feasible top-k assignment under per-expert capacity.

    Uses request ordering and optional p-bit spin bias for tie-breaking.
    """
    logits_batch = np.asarray(logits_batch, dtype=float)
    n_requests, n_experts = logits_batch.shape
    k = int(top_k)
    cap = int(expert_capacity)

    if request_order is None:
        priority = logits_batch.max(axis=1)
        if spin_matrix is not None:
            priority = priority + spin_weight * spin_matrix.max(axis=1)
        request_order = np.argsort(-priority)

    usage = np.zeros(n_experts, dtype=int)
    masks: list[np.ndarray] = [np.zeros(n_experts, dtype=bool) for _ in range(n_requests)]

    for r in request_order:
        scores = logits_batch[r].copy()
        if spin_matrix is not None:
            scores = scores + spin_weight * spin_matrix[r]

        chosen: list[int] = []
        for e in np.argsort(-scores):
            if len(chosen) >= k:
                break
            if usage[e] < cap:
                chosen.append(e)
                usage[e] += 1

        for e in chosen:
            masks[r][e] = True

    return masks


def batch_spins_to_masks(
    spins: dict[int, int] | np.ndarray,
    logits_batch: np.ndarray,
    top_k: int,
    expert_capacity: int,
    spin_weight: float = 0.25,
) -> list[np.ndarray]:
    """Decode batch spins via capacity-feasible assignment guided by p-bit bias."""
    logits_batch = np.asarray(logits_batch, dtype=float)
    n_requests, n_experts = logits_batch.shape
    spin_matrix = spins_to_bias_matrix(spins, n_requests, n_experts)
    return sequential_capacity_decode(
        logits_batch,
        top_k=top_k,
        expert_capacity=expert_capacity,
        spin_matrix=spin_matrix,
        spin_weight=spin_weight,
    )


def capacity_aware_routing_score(
    logits_batch: np.ndarray,
    masks: list[np.ndarray],
    capacity: int,
    overflow_penalty: float = 0.0,
) -> float:
    """Mean per-request logits minus batch overflow penalty on hot experts."""
    logits_batch = np.asarray(logits_batch, dtype=float)
    per_req = [routing_score(logits_batch[r], masks[r]) for r in range(len(masks))]
    base = float(np.mean(per_req))
    if overflow_penalty <= 0.0:
        return base
    usage = np.sum(np.stack(masks, axis=0), axis=0).astype(float)
    overflow = np.maximum(usage - capacity, 0.0)
    penalty = overflow_penalty * float(overflow.sum())
    return base - penalty / max(len(masks), 1)


def spins_to_mask(spins: dict[int, int] | np.ndarray, top_k: int) -> np.ndarray:
    """Convert Ising spins (+1 active) to boolean mask; trim/pad to top_k."""
    if isinstance(spins, dict):
        size = max(spins.keys()) + 1 if spins else 0
        arr = np.ones(size, dtype=int)
        for i, s in spins.items():
            arr[int(i)] = int(s)
    else:
        arr = np.asarray(spins, dtype=int).ravel()

    active = np.where(arr > 0)[0]
    if active.size <= top_k:
        mask = np.zeros(arr.size, dtype=bool)
        mask[active] = True
        return mask

    mask = np.zeros(arr.size, dtype=bool)
    mask[active[:top_k]] = True
    return mask


def routing_score(logits: np.ndarray, mask: np.ndarray) -> float:
    """Higher is better: sum of selected expert logits."""
    return float(np.dot(logits, mask.astype(float)))


def load_balance_score(masks: list[np.ndarray]) -> float:
    """
    Load-balance metric in [0, 1]; 1 means perfectly uniform expert usage.
    """
    if not masks:
        return 0.0
    usage = np.sum(np.stack(masks, axis=0), axis=0).astype(float)
    usage /= max(usage.sum(), 1.0)
    ideal = 1.0 / usage.size
    deviation = np.abs(usage - ideal).sum()
    max_dev = 2.0 * (1.0 - ideal)  # worst case: one expert gets all
    return float(max(0.0, 1.0 - deviation / max(max_dev, 1e-12)))
