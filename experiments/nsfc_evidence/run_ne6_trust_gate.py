# -*- coding: utf-8 -*-
"""NE6: do-no-harm trust gating envelope (Theorem 3b).

Strategies: always / never / gated  ×  σ grid.

Usage:
  python experiments/nsfc_evidence/run_ne6_trust_gate.py --profile smoke
  python experiments/nsfc_evidence/run_ne6_trust_gate.py --profile full
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

from experiments.nsfc_evidence.three_law_common import (  # noqa: E402
    bootstrap_ci,
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
    proxy_to_snr,
    snr_from_noise,
)


def run_strategy(problems, sigma: float, mode: str, steps: int, seed: int, always_lambda: float = 1.0) -> dict:
    rng = np.random.default_rng(seed)
    feas, exact, jacc = [], [], []
    lambdas = []
    for i, p in enumerate(problems):
        h_star = np.asarray(p.logits, dtype=float).ravel()
        if mode == "never":
            # pure constraint p-bit: zero semantic field (or very weak)
            h = np.zeros_like(h_star)
            lam = 0.0
        else:
            h_noisy = inject_field_noise(h_star, sigma, rng)
            if mode == "always":
                lam = always_lambda
            else:
                proxies = estimate_trust_proxies(h_noisy, rng=rng)
                snr_hat = proxy_to_snr(proxies["proxy"])
                # also blend with known sigma when available (oracle-free online)
                snr_true = snr_from_noise(h_star, sigma) if sigma > 0 else 1e6
                # gated uses proxy; for stronger signal mix lightly with true SNR only in analysis — keep online
                lam = gated_lambda(snr_hat, mode="gated", always_lambda=always_lambda)
            h = lam * h_noisy
        lambdas.append(lam)
        res = solve_with_logits(p, h if mode != "never" else h_star * 0.0 + 0.01 * h_star, steps=steps, seed=seed + i)
        # never: use tiny prior so cardinality still works via encoding from near-zero logits
        if mode == "never":
            res = solve_with_logits(p, 0.05 * np.ones_like(h_star), steps=steps, seed=seed + i)
        feas.append(res["feasible"])
        exact.append(res["exact_match"])
        jacc.append(res["jaccard"])

    mean_f, lo, hi = bootstrap_ci([float(x) for x in feas], seed=seed)
    return {
        "mode": mode,
        "sigma": sigma,
        "feasible_rate": mean_f,
        "feasible_ci": [lo, hi],
        "exact_match_rate": float(np.mean(exact)),
        "mean_jaccard": float(np.mean(jacc)),
        "mean_lambda": float(np.mean(lambdas)),
        "n": len(problems),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="NE6 do-no-harm trust gate")
    parser.add_argument("--profile", choices=["smoke", "full"], default="full")
    parser.add_argument("--n-tasks", type=int, default=None)
    parser.add_argument("--budget-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--always-lambda", type=float, default=1.0)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    n_tasks = args.n_tasks or (10 if args.profile == "smoke" else 40)
    steps = args.budget_steps or (100 if args.profile == "smoke" else 300)
    sigmas = [0.0, 0.4, 0.8] if args.profile == "smoke" else [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    modes = ["always", "never", "gated"]
    out_dir = resolve_out_dir("ne6_trust_gate", args.output_dir)

    problems = load_problems("bundled", n_tasks=n_tasks, seed=args.seed)
    rows = []
    for sigma in sigmas:
        for mode in modes:
            row = run_strategy(problems, sigma, mode, steps, args.seed, always_lambda=args.always_lambda)
            rows.append(row)
            print(f"  σ={sigma:.1f} {mode:6s} feas={row['feasible_rate']:.3f} λ̄={row['mean_lambda']:.3f}")

    # Plot envelope
    fig, ax = plt.subplots(figsize=(8, 5), dpi=130)
    colors = {"always": "#e45756", "never": "#9e9ac8", "gated": "#4c78a8"}
    for mode in modes:
        sub = [r for r in rows if r["mode"] == mode]
        xs = [r["sigma"] for r in sub]
        ys = [r["feasible_rate"] for r in sub]
        lo = [r["feasible_ci"][0] for r in sub]
        hi = [r["feasible_ci"][1] for r in sub]
        ax.plot(xs, ys, "o-", color=colors[mode], label=mode, lw=2)
        ax.fill_between(xs, lo, hi, color=colors[mode], alpha=0.15)
    # non-inferiority shade: gated >= never
    never_y = np.array([r["feasible_rate"] for r in rows if r["mode"] == "never"])
    gated_y = np.array([r["feasible_rate"] for r in rows if r["mode"] == "gated"])
    xs0 = [r["sigma"] for r in rows if r["mode"] == "never"]
    ax.fill_between(xs0, never_y, np.maximum(never_y, gated_y), color="#72b7b2", alpha=0.25, label="do-no-harm region")
    ax.set_xlabel("noise scale σ")
    ax.set_ylabel("feasible rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("NE6: do-no-harm trust gating (Theorem 3b)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_fig(os.path.join(out_dir, "ne6_donoharm_envelope.png"))

    # Non-inferiority check
    nonneg = []
    for sigma in sigmas:
        g = next(r["feasible_rate"] for r in rows if r["mode"] == "gated" and r["sigma"] == sigma)
        n = next(r["feasible_rate"] for r in rows if r["mode"] == "never" and r["sigma"] == sigma)
        nonneg.append(g + 1e-9 >= n)

    payload = {
        "profile": args.profile,
        "n_tasks": n_tasks,
        "budget_steps": steps,
        "noninferior_all_sigma": bool(all(nonneg)),
        "noninferior_per_sigma": {str(s): bool(v) for s, v in zip(sigmas, nonneg)},
        "rows": rows,
    }
    save_json(os.path.join(out_dir, "summary.json"), payload)
    print(f"NE6 done. noninferior_all={payload['noninferior_all_sigma']} -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
