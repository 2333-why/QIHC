# -*- coding: utf-8 -*-
"""NE3: η_sc refresh-timing scan — threshold + power law (Theorem 2).

Usage:
  python experiments/nsfc_evidence/run_ne3_refresh.py --profile smoke
  python experiments/nsfc_evidence/run_ne3_refresh.py --profile full
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
from qihc.theory.refresh import fit_power_law  # noqa: E402
from qihc.theory.trust import inject_field_noise  # noqa: E402


def run_refresh_schedule(
    problems,
    refresh_every: int,
    total_sweeps: int,
    sigma: float,
    seed: int,
    checkpoints: list[int],
) -> dict:
    """
    Simulate η_sc = 1/refresh_every:
    - semantic field refreshed every `refresh_every` sweeps
    - between refreshes, field is stale (plus optional noise)
    """
    rng = np.random.default_rng(seed)
    # per-checkpoint feasible flags across problems
    feas_at: dict[int, list[bool]] = {c: [] for c in checkpoints}

    for pi, p in enumerate(problems):
        h_star = np.asarray(p.logits, dtype=float).ravel()
        # initial noisy field
        h = inject_field_noise(h_star, sigma, rng) if sigma > 0 else h_star.copy()
        # accumulate sweeps in chunks of refresh_every
        done = 0
        last_mask_feasible = False
        while done < total_sweeps:
            # refresh semantic field
            h = inject_field_noise(h_star, sigma, rng) if sigma > 0 else h_star.copy()
            chunk = min(refresh_every, total_sweeps - done)
            # run p-bit with current (possibly stale until next refresh) field
            res = solve_with_logits(p, h, steps=chunk, seed=seed + pi * 100 + done)
            last_mask_feasible = bool(res["feasible"])
            done += chunk
            for c in checkpoints:
                if done >= c and len(feas_at[c]) == pi:
                    feas_at[c].append(last_mask_feasible)
        # fill any remaining checkpoints
        for c in checkpoints:
            while len(feas_at[c]) <= pi:
                feas_at[c].append(last_mask_feasible)

    r_curve = []
    for c in checkpoints:
        flags = feas_at[c]
        r_curve.append(1.0 - float(np.mean(flags)))
    fit = fit_power_law(np.asarray(checkpoints, dtype=float), np.asarray(r_curve, dtype=float))
    return {
        "refresh_every": refresh_every,
        "eta_sc": 1.0 / refresh_every,
        "sigma": sigma,
        "checkpoints": checkpoints,
        "r": r_curve,
        "feasible_final": 1.0 - r_curve[-1],
        "gamma": fit["gamma"],
        "r2": fit["r2"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="NE3 η_sc refresh scan")
    parser.add_argument("--profile", choices=["smoke", "full"], default="full")
    parser.add_argument("--n-tasks", type=int, default=None)
    parser.add_argument("--total-sweeps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    n_tasks = args.n_tasks or (8 if args.profile == "smoke" else 40)
    total = args.total_sweeps or (120 if args.profile == "smoke" else 400)
    refresh_list = [1, 2, 4, 8] if args.profile == "smoke" else [1, 2, 4, 8, 16, 32]
    sigmas = [0.0, 0.8] if args.profile == "smoke" else [0.0, 0.4, 0.8]
    checkpoints = [max(20, total // 5), max(40, 2 * total // 5), max(60, 3 * total // 5), max(80, 4 * total // 5), total]
    checkpoints = sorted(set(checkpoints))

    out_dir = resolve_out_dir("ne3_refresh", args.output_dir)
    problems = load_problems("bundled", n_tasks=n_tasks, seed=args.seed)

    rows = []
    for sigma in sigmas:
        for re in refresh_list:
            row = run_refresh_schedule(
                problems,
                refresh_every=re,
                total_sweeps=total,
                sigma=sigma,
                seed=args.seed,
                checkpoints=checkpoints,
            )
            rows.append(row)
            print(
                f"  σ={sigma} refresh_every={re} η_sc={row['eta_sc']:.3f} "
                f"feas={row['feasible_final']:.3f} γ={row['gamma']:.3f}"
            )

    # Plot 1: r(t) log-log families for each sigma at selected refresh
    fig, axes = plt.subplots(1, len(sigmas), figsize=(5 * len(sigmas), 4.5), dpi=130, squeeze=False)
    for ax, sigma in zip(axes[0], sigmas):
        for row in rows:
            if row["sigma"] != sigma:
                continue
            t = np.asarray(row["checkpoints"], dtype=float)
            r = np.maximum(np.asarray(row["r"], dtype=float), 1e-4)
            ax.loglog(t, r, "o-", label=f"η={row['eta_sc']:.3f}")
        ax.set_xlabel("budget t (sweeps)")
        ax.set_ylabel("residual infeasibility r(t)")
        ax.set_title(f"NE3: r(t) power-law (σ={sigma})")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    save_fig(os.path.join(out_dir, "ne3_powerlaw_rt.png"))

    # Plot 2: γ vs η_sc and feasible vs η_sc
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=130)
    for sigma in sigmas:
        sub = [r for r in rows if r["sigma"] == sigma]
        etas = [r["eta_sc"] for r in sub]
        gammas = [r["gamma"] for r in sub]
        feas = [r["feasible_final"] for r in sub]
        axes[0].plot(etas, gammas, "o-", label=f"σ={sigma}")
        axes[1].plot(etas, feas, "s-", label=f"σ={sigma}")
    axes[0].set_xlabel(r"$\eta_{sc}$")
    axes[0].set_ylabel(r"fitted $\gamma$")
    axes[0].set_title("NE3: γ–η_sc (Theorem 2)")
    axes[0].set_xscale("log")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].set_xlabel(r"$\eta_{sc}$")
    axes[1].set_ylabel("final feasible rate")
    axes[1].set_title("NE3: adiabatic plateau")
    axes[1].set_xscale("log")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    save_fig(os.path.join(out_dir, "ne3_eta_threshold.png"))

    # Estimate η* as smallest η where feasible within 2pp of max for σ=0.8 (or last sigma)
    eta_star = {}
    for sigma in sigmas:
        sub = sorted([r for r in rows if r["sigma"] == sigma], key=lambda x: x["eta_sc"])
        if not sub:
            continue
        best = max(r["feasible_final"] for r in sub)
        eta_star[str(sigma)] = None
        for r in sub:
            if r["feasible_final"] >= best - 0.02:
                eta_star[str(sigma)] = r["eta_sc"]
                break

    payload = {
        "profile": args.profile,
        "n_tasks": n_tasks,
        "total_sweeps": total,
        "checkpoints": checkpoints,
        "eta_star_estimate": eta_star,
        "rows": rows,
    }
    save_json(os.path.join(out_dir, "summary.json"), payload)
    print(f"NE3 done. η*≈{eta_star} -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
