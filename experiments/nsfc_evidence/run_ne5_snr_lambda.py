# -*- coding: utf-8 -*-
"""NE5: SNR × λ grid — optimal trust Wiener curve (Theorem 3a).

Usage:
  python experiments/nsfc_evidence/run_ne5_snr_lambda.py --profile smoke
  python experiments/nsfc_evidence/run_ne5_snr_lambda.py --profile full
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
    load_problems,
    resolve_out_dir,
    save_fig,
    save_json,
    solve_with_logits,
)
from qihc.theory.trust import inject_field_noise, optimal_lambda, snr_from_noise  # noqa: E402


def gain_curve_for_snr(problems, sigma: float, lambdas: list[float], steps: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    # baseline λ=0
    base_feas = []
    for i, p in enumerate(problems):
        h_star = np.asarray(p.logits, dtype=float).ravel()
        res = solve_with_logits(p, 0.05 * np.ones_like(h_star), steps=steps, seed=seed + i)
        base_feas.append(res["feasible"])
    u0 = float(np.mean(base_feas))

    gains = []
    feas_list = []
    snr_vals = []
    for lam in lambdas:
        feas = []
        snrs = []
        for i, p in enumerate(problems):
            h_star = np.asarray(p.logits, dtype=float).ravel()
            snrs.append(snr_from_noise(h_star, sigma))
            h = inject_field_noise(h_star, sigma, rng)
            res = solve_with_logits(p, lam * h, steps=steps, seed=seed + 1000 + i)
            feas.append(res["feasible"])
        g = float(np.mean(feas)) - u0
        gains.append(g)
        feas_list.append(float(np.mean(feas)))
        snr_vals.append(float(np.mean(snrs)))

    snr_mean = float(np.mean(snr_vals))
    # empirical λ*: argmax gain
    best_idx = int(np.argmax(gains))
    lam_star_emp = float(lambdas[best_idx])
    lam_star_th = float(optimal_lambda(snr_mean))
    return {
        "sigma": sigma,
        "snr_mean": snr_mean,
        "lambdas": lambdas,
        "gains": gains,
        "feasible_rates": feas_list,
        "u0": u0,
        "lam_star_emp": lam_star_emp,
        "lam_star_theory": lam_star_th,
        "gain_at_emp_star": gains[best_idx],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="NE5 SNR×λ Wiener curve")
    parser.add_argument("--profile", choices=["smoke", "full"], default="full")
    parser.add_argument("--n-tasks", type=int, default=None)
    parser.add_argument("--budget-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    n_tasks = args.n_tasks or (8 if args.profile == "smoke" else 30)
    steps = args.budget_steps or (80 if args.profile == "smoke" else 250)
    sigmas = [0.2, 0.5, 1.0] if args.profile == "smoke" else [0.1, 0.3, 0.5, 0.8, 1.2]
    lambdas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0] if args.profile == "full" else [0.0, 0.5, 1.0, 1.5]
    out_dir = resolve_out_dir("ne5_snr_lambda", args.output_dir)

    problems = load_problems("bundled", n_tasks=n_tasks, seed=args.seed)
    rows = []
    fig, ax = plt.subplots(figsize=(8, 5), dpi=130)
    for sigma in sigmas:
        row = gain_curve_for_snr(problems, sigma, lambdas, steps, args.seed)
        rows.append(row)
        ax.plot(lambdas, row["gains"], "o-", label=f"σ={sigma} (SNR≈{row['snr_mean']:.2f})")
        ax.axvline(row["lam_star_emp"], color="gray", ls=":", alpha=0.5)
        print(
            f"  σ={sigma} SNR={row['snr_mean']:.2f} λ*_emp={row['lam_star_emp']:.2f} "
            f"λ*_th={row['lam_star_theory']:.2f} G*={row['gain_at_emp_star']:.3f}"
        )

    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xlabel("trust coefficient λ")
    ax.set_ylabel("feasible-rate gain G(λ)")
    ax.set_title("NE5: G(λ) parabolas (Theorem 3a)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    save_fig(os.path.join(out_dir, "ne5_gain_parabolas.png"))

    # Wiener curve: λ* vs SNR
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=130)
    snrs = [r["snr_mean"] for r in rows]
    emp = [r["lam_star_emp"] for r in rows]
    th = [r["lam_star_theory"] for r in rows]
    ax.plot(snrs, emp, "o-", color="#4c78a8", label="empirical λ*")
    ax.plot(snrs, th, "s--", color="#e45756", label="theory λ* (Wiener)")
    # dense theory curve
    snr_grid = np.linspace(0.01, max(snrs + [2.0]), 50)
    ax.plot(snr_grid, [optimal_lambda(s) for s in snr_grid], color="#e45756", alpha=0.3)
    ax.set_xlabel("SNR")
    ax.set_ylabel("λ*")
    ax.set_title("NE5: optimal trust Wiener curve")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_fig(os.path.join(out_dir, "ne5_wiener_curve.png"))

    payload = {
        "profile": args.profile,
        "n_tasks": n_tasks,
        "budget_steps": steps,
        "lambdas": lambdas,
        "rows": rows,
        "mean_abs_err_lam_star": float(np.mean([abs(r["lam_star_emp"] - r["lam_star_theory"]) for r in rows])),
    }
    save_json(os.path.join(out_dir, "summary.json"), payload)
    print(f"NE5 done. mean|λ*_emp-λ*_th|={payload['mean_abs_err_lam_star']:.3f} -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
