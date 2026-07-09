"""Trust-gating law utilities (Theorems 3a / 3b)."""

from __future__ import annotations

import numpy as np


def snr_from_noise(h_star: np.ndarray, sigma: float) -> float:
    """SNR = ||h*||² / (σ² n)."""
    h_star = np.asarray(h_star, dtype=float).ravel()
    n = max(h_star.size, 1)
    denom = (sigma**2) * n + 1e-12
    return float(np.dot(h_star, h_star) / denom)


def optimal_lambda(snr: float, c1: float = 1.0, c2: float = 1.0) -> float:
    """λ* = (c1/c2) * SNR / (1 + SNR)  (Theorem 3a, Wiener form)."""
    snr = max(float(snr), 0.0)
    return (c1 / max(c2, 1e-12)) * (snr / (1.0 + snr))


def usefulness_threshold_snr(lam: float, c1: float = 1.0, c2: float = 1.0) -> float:
    """
    SNR*(λ) such that G(λ)>0 iff SNR > SNR*(λ), for λ < 2 c1/c2.
    """
    lam = float(lam)
    denom = 2.0 * c1 - c2 * lam
    if denom <= 1e-12:
        return float("inf")
    return (c2 * lam) / denom


def expected_gain(lam: float, snr: float, n: int, h_norm2: float, c1: float = 1.0, c2: float = 1.0) -> float:
    """E[G(λ)] = λ c1 ||h*||² - 0.5 λ² c2 (||h*||² + σ² n), with σ² n = ||h*||² / SNR."""
    if snr <= 0:
        noise_energy = 1e12
    else:
        noise_energy = h_norm2 / snr
    return float(lam * c1 * h_norm2 - 0.5 * (lam**2) * c2 * (h_norm2 + noise_energy))


def gated_lambda(
    snr_hat: float,
    c1: float = 1.0,
    c2: float = 1.0,
    always_lambda: float = 1.0,
    mode: str = "gated",
) -> float:
    """
    Trust strategies:
      always : fixed large λ
      never  : λ = 0
      gated  : λ = λ*(SNR_hat) ∈ [0, 2λ*]  (do-no-harm)
    """
    if mode == "never":
        return 0.0
    if mode == "always":
        return float(always_lambda)
    # gated
    lam_star = optimal_lambda(snr_hat, c1=c1, c2=c2)
    return float(np.clip(lam_star, 0.0, 2.0 * lam_star + 1e-12))


def estimate_trust_proxies(
    logits: np.ndarray,
    n_consistency_samples: int = 5,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """
    Online trust proxies for SNR (NE7):
      - entropy of softmax(logits)
      - self-consistency under logit noise
      - top-k stability
    """
    rng = rng or np.random.default_rng(0)
    logits = np.asarray(logits, dtype=float).ravel()
    z = logits - logits.max()
    p = np.exp(z)
    p = p / p.sum()
    entropy = float(-np.sum(p * np.log(np.clip(p, 1e-12, 1.0))))
    max_ent = float(np.log(max(logits.size, 1)))
    inv_entropy = 1.0 - entropy / max(max_ent, 1e-12)

    top = int(np.argmax(p))
    agree = 0
    for _ in range(n_consistency_samples):
        noisy = logits + rng.normal(0.0, 0.5, size=logits.size)
        if int(np.argmax(noisy)) == top:
            agree += 1
    consistency = agree / max(n_consistency_samples, 1)

    # Softmax peakiness as third proxy
    peakiness = float(np.max(p))

    # Aggregate proxy in [0, 1] mapped later to SNR scale
    proxy = float(0.4 * inv_entropy + 0.4 * consistency + 0.2 * peakiness)
    return {
        "entropy": entropy,
        "inv_entropy": inv_entropy,
        "consistency": float(consistency),
        "peakiness": peakiness,
        "proxy": proxy,
    }


def proxy_to_snr(proxy: float, scale: float = 5.0) -> float:
    """Map [0,1] proxy to a positive SNR estimate."""
    proxy = float(np.clip(proxy, 0.0, 1.0))
    return scale * proxy / max(1.0 - proxy, 1e-3)


def inject_field_noise(h_star: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """ĥ = h* + ε, ε ~ N(0, σ²)."""
    h_star = np.asarray(h_star, dtype=float).ravel()
    return h_star + rng.normal(0.0, sigma, size=h_star.size)
