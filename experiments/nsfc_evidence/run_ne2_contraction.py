# -*- coding: utf-8 -*-
"""NE2: Mean-field contraction convergence vs β (Theorem 1c).

Usage:
  python experiments/nsfc_evidence/run_ne2_contraction.py --profile smoke
  python experiments/nsfc_evidence/run_ne2_contraction.py --profile full
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

from experiments.nsfc_evidence.three_law_common import resolve_out_dir, save_fig, save_json  # noqa: E402
from qihc.orchestrator.reasoning import subset_to_ising  # noqa: E402
from qihc.theory.mean_field import contraction_lipschitz, mean_field_fixed_point  # noqa: E402
from experiments.nsfc_evidence.three_law_common import load_problems  # noqa: E402


def fit_rate(residuals: list[float]) -> float:
    """Geometric rate from late residuals: r_{t+1} ≈ ρ r_t."""
    r = np.asarray(residuals, dtype=float)
    r = r[r > 1e-12]
    if r.size < 3:
        return 0.0
    # use middle segment to avoid init / floor
    seg = r[max(1, len(r) // 4) : max(2, 3 * len(r) // 4)]
    if seg.size < 2:
        seg = r
    ratios = seg[1:] / np.maximum(seg[:-1], 1e-12)
    return float(np.median(ratios))


def main() -> int:
    parser = argparse.ArgumentParser(description="NE2 contraction convergence")
    parser.add_argument("--profile", choices=["smoke", "full"], default="full")
    parser.add_argument("--n-tasks", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    n_tasks = args.n_tasks or (8 if args.profile == "smoke" else 40)
    betas = [0.2, 0.5, 1.0, 2.0, 5.0] if args.profile == "full" else [0.5, 1.0, 2.0]
    out_dir = resolve_out_dir("ne2_contraction", args.output_dir)

    problems = load_problems("bundled", n_tasks=n_tasks, seed=args.seed)
    rows = []
    fig, ax = plt.subplots(figsize=(8, 5), dpi=130)

    for beta in betas:
        all_residuals = []
        rates = []
        beta_L_list = []
        n_unique_fp = []
        for i, p in enumerate(problems):
            weight, field = subset_to_ising(p.logits, top_k=p.top_k, exclusion_pairs=p.exclusion_pairs)
            L = contraction_lipschitz(weight)
            beta_L_list.append(beta * L)
            # multiple inits to count fixed points
            fps = []
            for k in range(5):
                m0 = np.random.default_rng(args.seed + i * 10 + k).uniform(-0.5, 0.5, size=field.size)
                m_star, residuals, _ = mean_field_fixed_point(field, weight, beta=beta, m0=m0, max_iters=150)
                fps.append(np.round(m_star, 4))
                if k == 0:
                    all_residuals.append(residuals)
                    rates.append(fit_rate(residuals))
            uniq = {tuple(fp.tolist()) for fp in fps}
            n_unique_fp.append(len(uniq))

        # average residual curve (pad)
        max_t = max(len(r) for r in all_residuals)
        mat = np.full((len(all_residuals), max_t), np.nan)
        for i, r in enumerate(all_residuals):
            mat[i, : len(r)] = r
        mean_r = np.nanmean(mat, axis=0)
        t = np.arange(1, mean_r.size + 1)
        ax.semilogy(t, np.maximum(mean_r, 1e-12), label=f"β={beta} (βL≈{np.mean(beta_L_list):.2f})")

        rows.append(
            {
                "beta": beta,
                "mean_beta_L": float(np.mean(beta_L_list)),
                "mean_rate": float(np.mean(rates)),
                "mean_n_fixed_points": float(np.mean(n_unique_fp)),
                "fraction_unique_fp": float(np.mean([n == 1 for n in n_unique_fp])),
            }
        )

    ax.set_xlabel("Iteration t")
    ax.set_ylabel(r"$\|m^{(t)}-m^\ast\|_\infty$")
    ax.set_title("NE2: Mean-field contraction vs β (Theorem 1c)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_fig(os.path.join(out_dir, "ne2_contraction_curves.png"))

    # rate vs betaL scatter
    fig, ax = plt.subplots(figsize=(6, 4), dpi=130)
    xs = [r["mean_beta_L"] for r in rows]
    ys = [r["mean_rate"] for r in rows]
    ax.plot(xs, ys, "o-", color="#4c78a8")
    ax.axvline(1.0, color="#e45756", ls="--", label="βL=1 threshold")
    ax.set_xlabel("mean βL")
    ax.set_ylabel("empirical geometric rate")
    ax.set_title("NE2: rate vs βL")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_fig(os.path.join(out_dir, "ne2_rate_vs_betaL.png"))

    payload = {"profile": args.profile, "n_tasks": n_tasks, "rows": rows}
    save_json(os.path.join(out_dir, "summary.json"), payload)
    print(json_dumps(payload))
    return 0


def json_dumps(obj) -> str:
    import json

    return json.dumps(obj, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    raise SystemExit(main())
