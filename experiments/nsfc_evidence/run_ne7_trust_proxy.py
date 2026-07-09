# -*- coding: utf-8 -*-
"""NE7: Trust proxy validity vs true SNR (supports NE6 gating).

Usage:
  python experiments/nsfc_evidence/run_ne7_trust_proxy.py --profile smoke
  python experiments/nsfc_evidence/run_ne7_trust_proxy.py --profile full
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def spearmanr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Spearman rank correlation without scipy dependency. Returns (rho, nan_pvalue)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3:
        return 0.0, float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = float(np.sqrt(np.sum(rx**2) * np.sum(ry**2))) + 1e-12
    rho = float(np.sum(rx * ry) / denom)
    return rho, float("nan")

from experiments.nsfc_evidence.three_law_common import (  # noqa: E402
    load_problems,
    resolve_out_dir,
    save_fig,
    save_json,
    solve_with_logits,
)
from qihc.theory.trust import (  # noqa: E402
    estimate_trust_proxies,
    gated_lambda,
    inject_field_noise,
    optimal_lambda,
    proxy_to_snr,
    snr_from_noise,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="NE7 trust proxy validity")
    parser.add_argument("--profile", choices=["smoke", "full"], default="full")
    parser.add_argument("--n-tasks", type=int, default=None)
    parser.add_argument("--budget-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    n_tasks = args.n_tasks or (10 if args.profile == "smoke" else 40)
    steps = args.budget_steps or (80 if args.profile == "smoke" else 200)
    sigmas = [0.0, 0.4, 0.8] if args.profile == "smoke" else [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    out_dir = resolve_out_dir("ne7_trust_proxy", args.output_dir)

    problems = load_problems("bundled", n_tasks=n_tasks, seed=args.seed)
    rng = np.random.default_rng(args.seed)

    records = []
    for sigma in sigmas:
        for i, p in enumerate(problems):
            h_star = np.asarray(p.logits, dtype=float).ravel()
            h = inject_field_noise(h_star, sigma, rng) if sigma > 0 else h_star.copy()
            snr_true = snr_from_noise(h_star, sigma) if sigma > 0 else 1e3
            proxies = estimate_trust_proxies(h, rng=rng)
            snr_hat = proxy_to_snr(proxies["proxy"])
            lam_true = optimal_lambda(snr_true)
            lam_hat = gated_lambda(snr_hat, mode="gated")
            records.append(
                {
                    "sigma": sigma,
                    "snr_true": float(snr_true),
                    "snr_hat": float(snr_hat),
                    "proxy": float(proxies["proxy"]),
                    "entropy": float(proxies["entropy"]),
                    "consistency": float(proxies["consistency"]),
                    "peakiness": float(proxies["peakiness"]),
                    "lam_true": float(lam_true),
                    "lam_hat": float(lam_hat),
                    "abs_lam_err": float(abs(lam_hat - lam_true)),
                }
            )

    snr_true_arr = np.array([r["snr_true"] for r in records])
    snr_hat_arr = np.array([r["snr_hat"] for r in records])
    proxy_arr = np.array([r["proxy"] for r in records])
    # Spearman on finite SNR
    valid = np.isfinite(snr_true_arr) & (snr_true_arr < 500)
    rho_snr, p_snr = spearmanr(snr_true_arr[valid], snr_hat_arr[valid]) if valid.sum() > 3 else (0.0, 1.0)
    rho_proxy, p_proxy = spearmanr(snr_true_arr[valid], proxy_arr[valid]) if valid.sum() > 3 else (0.0, 1.0)

    # Feasible loss: proxy-gated vs oracle-gated
    loss = []
    for sigma in sigmas:
        feas_proxy, feas_oracle = [], []
        for i, p in enumerate(problems):
            h_star = np.asarray(p.logits, dtype=float).ravel()
            h = inject_field_noise(h_star, sigma, rng) if sigma > 0 else h_star.copy()
            snr_t = snr_from_noise(h_star, sigma) if sigma > 0 else 1e3
            proxies = estimate_trust_proxies(h, rng=rng)
            lam_p = gated_lambda(proxy_to_snr(proxies["proxy"]), mode="gated")
            lam_o = optimal_lambda(snr_t)
            rp = solve_with_logits(p, lam_p * h, steps=steps, seed=args.seed + i)
            ro = solve_with_logits(p, lam_o * h, steps=steps, seed=args.seed + i)
            feas_proxy.append(rp["feasible"])
            feas_oracle.append(ro["feasible"])
        loss.append(
            {
                "sigma": sigma,
                "feas_proxy": float(np.mean(feas_proxy)),
                "feas_oracle": float(np.mean(feas_oracle)),
                "delta": float(np.mean(feas_proxy) - np.mean(feas_oracle)),
            }
        )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=130)
    axes[0].scatter(snr_true_arr[valid], snr_hat_arr[valid], alpha=0.5, c="#4c78a8")
    axes[0].set_xlabel("true SNR")
    axes[0].set_ylabel("proxy SNR")
    axes[0].set_title(f"NE7: SNR proxy (Spearman ρ={rho_snr:.2f})")
    axes[0].grid(True, alpha=0.3)

    xs = [d["sigma"] for d in loss]
    axes[1].plot(xs, [d["feas_oracle"] for d in loss], "s--", label="oracle λ*", color="#e45756")
    axes[1].plot(xs, [d["feas_proxy"] for d in loss], "o-", label="proxy λ̂", color="#4c78a8")
    axes[1].set_xlabel("σ")
    axes[1].set_ylabel("feasible rate")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("NE7: proxy vs oracle gating loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    save_fig(os.path.join(out_dir, "ne7_proxy_validity.png"))

    payload = {
        "profile": args.profile,
        "n_tasks": n_tasks,
        "spearman_snr": float(rho_snr),
        "spearman_snr_pvalue": float(p_snr),
        "spearman_proxy": float(rho_proxy),
        "mean_abs_lam_err": float(np.mean([r["abs_lam_err"] for r in records])),
        "mean_feas_delta_proxy_minus_oracle": float(np.mean([d["delta"] for d in loss])),
        "loss_by_sigma": loss,
        "n_records": len(records),
    }
    save_json(os.path.join(out_dir, "summary.json"), payload)
    print(
        f"NE7 done. Spearman(SNR)={rho_snr:.3f} meanΔfeas={payload['mean_feas_delta_proxy_minus_oracle']:.3f} -> {out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
