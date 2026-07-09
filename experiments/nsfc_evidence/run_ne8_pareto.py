# -*- coding: utf-8 -*-
"""NE8: Feasibility–compute–energy 3D Pareto under {η_sc, τ} (Theorem 4 / innovation 3).

Usage:
  python experiments/nsfc_evidence/run_ne8_pareto.py --profile smoke
  python experiments/nsfc_evidence/run_ne8_pareto.py --profile full
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
from qihc.theory.trust import (  # noqa: E402
    estimate_trust_proxies,
    gated_lambda,
    inject_field_noise,
    proxy_to_snr,
)

# Energy accounting assumptions (documented; relative units OK for Pareto shape)
E_LLM_CALL = 1.0  # relative energy per semantic refresh
E_PBIT_SWEEP = 0.002  # relative energy per p-bit sweep per variable


def run_point(problems, refresh_every: int, mode: str, total_sweeps: int, sigma: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    feas = []
    llm_calls = 0
    pbit_flips = 0
    for i, p in enumerate(problems):
        h_star = np.asarray(p.logits, dtype=float).ravel()
        n = h_star.size
        done = 0
        last = False
        n_refresh = 0
        while done < total_sweeps:
            n_refresh += 1
            h_noisy = inject_field_noise(h_star, sigma, rng) if sigma > 0 else h_star.copy()
            if mode == "never":
                h = 0.05 * np.ones_like(h_star)
                lam = 0.0
            elif mode == "always":
                lam = 1.0
                h = lam * h_noisy
            else:
                proxies = estimate_trust_proxies(h_noisy, rng=rng)
                lam = gated_lambda(proxy_to_snr(proxies["proxy"]), mode="gated")
                h = lam * h_noisy
            chunk = min(refresh_every, total_sweeps - done)
            res = solve_with_logits(p, h, steps=chunk, seed=seed + i * 17 + done)
            last = bool(res["feasible"])
            pbit_flips += chunk * n
            done += chunk
        feas.append(last)
        llm_calls += n_refresh

    n_prob = max(len(problems), 1)
    mean_llm = llm_calls / n_prob
    mean_flips = pbit_flips / n_prob
    energy = mean_llm * E_LLM_CALL + mean_flips * E_PBIT_SWEEP
    return {
        "refresh_every": refresh_every,
        "eta_sc": 1.0 / refresh_every,
        "mode": mode,
        "sigma": sigma,
        "feasible_rate": float(np.mean(feas)),
        "mean_llm_calls": float(mean_llm),
        "mean_pbit_flips": float(mean_flips),
        "energy_rel": float(energy),
        "total_sweeps": total_sweeps,
    }


def pareto_front_2d(points: list[dict], xkey: str, ykey: str, maximize_y: bool = True) -> list[dict]:
    """Simple 2D Pareto: minimize x, maximize y."""
    pts = sorted(points, key=lambda p: p[xkey])
    front = []
    best_y = -np.inf if maximize_y else np.inf
    for p in pts:
        y = p[ykey]
        if maximize_y and y > best_y + 1e-12:
            front.append(p)
            best_y = y
        elif (not maximize_y) and y < best_y - 1e-12:
            front.append(p)
            best_y = y
    return front


def main() -> int:
    parser = argparse.ArgumentParser(description="NE8 3D Pareto")
    parser.add_argument("--profile", choices=["smoke", "full"], default="full")
    parser.add_argument("--n-tasks", type=int, default=None)
    parser.add_argument("--total-sweeps", type=int, default=None)
    parser.add_argument("--sigma", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    n_tasks = args.n_tasks or (8 if args.profile == "smoke" else 30)
    total = args.total_sweeps or (100 if args.profile == "smoke" else 300)
    refresh_list = [1, 4, 16] if args.profile == "smoke" else [1, 2, 4, 8, 16, 32]
    modes = ["always", "never", "gated"]
    out_dir = resolve_out_dir("ne8_pareto", args.output_dir)

    problems = load_problems("bundled", n_tasks=n_tasks, seed=args.seed)
    rows = []
    for mode in modes:
        for re in refresh_list:
            row = run_point(problems, re, mode, total, args.sigma, args.seed)
            rows.append(row)
            print(
                f"  {mode:6s} η={row['eta_sc']:.3f} feas={row['feasible_rate']:.3f} "
                f"llm={row['mean_llm_calls']:.1f} E={row['energy_rel']:.3f}"
            )

    # 2D projections
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=130)
    colors = {"always": "#e45756", "never": "#9e9ac8", "gated": "#4c78a8"}
    for mode in modes:
        sub = [r for r in rows if r["mode"] == mode]
        axes[0].scatter(
            [r["mean_llm_calls"] for r in sub],
            [r["feasible_rate"] for r in sub],
            c=colors[mode],
            label=mode,
            s=60,
        )
        axes[1].scatter(
            [r["energy_rel"] for r in sub],
            [r["feasible_rate"] for r in sub],
            c=colors[mode],
            label=mode,
            s=60,
        )
    # Pareto of gated
    gated = [r for r in rows if r["mode"] == "gated"]
    front = pareto_front_2d(gated, "mean_llm_calls", "feasible_rate")
    if front:
        axes[0].plot(
            [p["mean_llm_calls"] for p in front],
            [p["feasible_rate"] for p in front],
            color="#4c78a8",
            lw=2,
            label="gated Pareto",
        )
    axes[0].set_xlabel("mean LLM calls")
    axes[0].set_ylabel("feasible rate")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("NE8: feasible vs compute")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel("relative energy")
    axes[1].set_ylabel("feasible rate")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("NE8: feasible vs energy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    save_fig(os.path.join(out_dir, "ne8_pareto_2d.png"))

    # 3D scatter
    fig = plt.figure(figsize=(7, 6), dpi=130)
    ax = fig.add_subplot(111, projection="3d")
    for mode in modes:
        sub = [r for r in rows if r["mode"] == mode]
        ax.scatter(
            [r["mean_llm_calls"] for r in sub],
            [r["energy_rel"] for r in sub],
            [r["feasible_rate"] for r in sub],
            c=colors[mode],
            label=mode,
            s=50,
        )
    ax.set_xlabel("LLM calls")
    ax.set_ylabel("energy")
    ax.set_zlabel("feasible")
    ax.set_title(f"NE8: 3D Pareto (σ={args.sigma})")
    ax.legend()
    save_fig(os.path.join(out_dir, "ne8_pareto_3d.png"))

    payload = {
        "profile": args.profile,
        "n_tasks": n_tasks,
        "total_sweeps": total,
        "sigma": args.sigma,
        "energy_model": {"E_LLM_CALL": E_LLM_CALL, "E_PBIT_SWEEP": E_PBIT_SWEEP},
        "gated_pareto_llm": front,
        "rows": rows,
    }
    save_json(os.path.join(out_dir, "summary.json"), payload)
    print(f"NE8 done. {len(rows)} points -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
