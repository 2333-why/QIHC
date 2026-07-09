"""Refresh-timing law utilities (Lemma 2 / Theorem 2)."""

from __future__ import annotations

import numpy as np


def kl_bound_stale_field(
    delta_h: np.ndarray,
    beta: float = 1.0,
    lambda_max: float | None = None,
    n: int | None = None,
) -> float:
    """
    KL(π_h || π_h') ≤ (β²/2) λ_max(Σ) ||Δh||²  (Lemma 2).

    Uses λ_max ≤ n as a safe upper bound when covariance is unknown.
    """
    delta_h = np.asarray(delta_h, dtype=float).ravel()
    if n is None:
        n = delta_h.size
    if lambda_max is None:
        lambda_max = float(n)
    return 0.5 * (beta**2) * float(lambda_max) * float(np.dot(delta_h, delta_h))


def tv_bound_stale_field(
    delta_h: np.ndarray,
    beta: float = 1.0,
    lambda_max: float | None = None,
    n: int | None = None,
) -> float:
    """TV ≤ sqrt(KL/2) via Pinsker."""
    kl = kl_bound_stale_field(delta_h, beta=beta, lambda_max=lambda_max, n=n)
    return float(np.sqrt(max(kl, 0.0) / 2.0))


def estimate_tv_from_samples(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
) -> float:
    """
    Empirical TV between two collections of ±1 spin configurations.

    Uses Hamming-histogram over unique states (exact for small n).
    """
    a = np.asarray(samples_a, dtype=int)
    b = np.asarray(samples_b, dtype=int)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)

    def _hist(arr: np.ndarray) -> dict[tuple, float]:
        counts: dict[tuple, int] = {}
        for row in arr:
            key = tuple(int(x) for x in row.tolist())
            counts[key] = counts.get(key, 0) + 1
        total = max(sum(counts.values()), 1)
        return {k: v / total for k, v in counts.items()}

    pa, pb = _hist(a), _hist(b)
    keys = set(pa) | set(pb)
    return 0.5 * float(sum(abs(pa.get(k, 0.0) - pb.get(k, 0.0)) for k in keys))


def residual_infeasibility_curve(
    feasible_flags_over_time: list[list[bool]],
) -> dict[str, list[float]]:
    """
    Convert per-budget feasibility traces into residual infeasibility r(t).

    Parameters
    ----------
    feasible_flags_over_time :
        Outer list = budget checkpoints; each inner list = per-problem feasible bools.
    """
    budgets = list(range(1, len(feasible_flags_over_time) + 1))
    r = [1.0 - float(np.mean(flags)) for flags in feasible_flags_over_time]
    return {"t": [float(b) for b in budgets], "r": r}


def fit_power_law(t: np.ndarray, r: np.ndarray) -> dict[str, float]:
    """Fit r ~ t^{-γ} in log-log space (positive r only)."""
    t = np.asarray(t, dtype=float)
    r = np.asarray(r, dtype=float)
    mask = (t > 0) & (r > 1e-8)
    if mask.sum() < 2:
        return {"gamma": 0.0, "log_c": 0.0, "r2": 0.0}
    x = np.log(t[mask])
    y = np.log(r[mask])
    coef = np.polyfit(x, y, 1)
    gamma = float(-coef[0])
    log_c = float(coef[1])
    y_hat = coef[0] * x + coef[1]
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-12
    return {"gamma": gamma, "log_c": log_c, "r2": 1.0 - ss_res / ss_tot}
