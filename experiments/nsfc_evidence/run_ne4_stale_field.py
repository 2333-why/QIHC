# -*- coding: utf-8 -*-
"""NE4: Stale-field TV/KL linear bound (Lemma 2).

Usage:
  python experiments/nsfc_evidence/run_ne4_stale_field.py --profile smoke
  python experiments/nsfc_evidence/run_ne4_stale_field.py --profile full
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
)
from qihc import IsingModel  # noqa: E402
from qihc.orchestrator.reasoning import subset_to_ising  # noqa: E402
from qihc.theory.refresh import estimate_tv_from_samples, tv_bound_stale_field  # noqa: E402


def sample_spins(weight, field, steps: int, seed: int, n_samples: int = 64) -> np.ndarray:
    rng = np.random.default_rng(seed)
    samples = []
    j_dict = {}
    n = weight.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if weight[i, j] != 0:
                j_dict[(i, j)] = float(weight[i, j])
                j_dict[(j, i)] = float(weight[i, j])
    for s in range(n_samples):
        model = IsingModel(size=n, Weight=weight.copy(), Field=field.copy())
        np.random.seed(int(rng.integers(0, 1_000_000)))
        spins, _, _ = model.gibbs_sampling_Maxcut(
            J=j_dict, steps=steps, T_start=1.0, T_end=0.3, k=1.0, sequential=True
        )
        # Sampler returns dict {node: ±1}; convert to dense array
        if isinstance(spins, dict):
            arr = np.ones(n, dtype=int)
            for node, val in spins.items():
                arr[int(node)] = int(val)
        else:
            arr = np.asarray(spins, dtype=int).ravel()
        samples.append(arr)
    return np.stack(samples, axis=0)


def main() -> int:
    parser = argparse.ArgumentParser(description="NE4 stale-field bound")
    parser.add_argument("--profile", choices=["smoke", "full"], default="full")
    parser.add_argument("--n-tasks", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    n_tasks = args.n_tasks or (4 if args.profile == "smoke" else 12)
    steps = args.steps or (80 if args.profile == "smoke" else 200)
    n_samples = args.n_samples or (20 if args.profile == "smoke" else 48)
    deltas = [0.1, 0.2, 0.4, 0.8, 1.2] if args.profile == "full" else [0.2, 0.5, 1.0]
    out_dir = resolve_out_dir("ne4_stale_field", args.output_dir)

    problems = load_problems("bundled", n_tasks=n_tasks, seed=args.seed)
    points = []
    rng = np.random.default_rng(args.seed)

    for pi, p in enumerate(problems):
        weight, field = subset_to_ising(p.logits, top_k=p.top_k, exclusion_pairs=p.exclusion_pairs)
        base = sample_spins(weight, field, steps=steps, seed=args.seed + pi, n_samples=n_samples)
        for dscale in deltas:
            delta = rng.normal(0.0, dscale, size=field.size)
            field2 = field + delta
            samp2 = sample_spins(weight, field2, steps=steps, seed=args.seed + 1000 + pi, n_samples=n_samples)
            tv_emp = estimate_tv_from_samples(base, samp2)
            tv_bound = tv_bound_stale_field(delta, beta=1.0, n=field.size)
            points.append(
                {
                    "task": pi,
                    "delta_norm": float(np.linalg.norm(delta)),
                    "tv_emp": float(tv_emp),
                    "tv_bound": float(tv_bound),
                    "dscale": dscale,
                }
            )

    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=130)
    xs = [p["delta_norm"] for p in points]
    ys = [p["tv_emp"] for p in points]
    bs = [p["tv_bound"] for p in points]
    ax.scatter(xs, ys, c="#4c78a8", alpha=0.7, label="empirical TV")
    # sort bound for line
    order = np.argsort(xs)
    ax.plot(np.array(xs)[order], np.array(bs)[order], color="#e45756", ls="--", label="Lemma 2 upper bound")
    ax.set_xlabel(r"$\|\Delta h\|_2$")
    ax.set_ylabel("TV distance")
    ax.set_title("NE4: Stale-field TV vs bound (Lemma 2)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_fig(os.path.join(out_dir, "ne4_tv_bound.png"))

    # fraction below bound
    below = float(np.mean([p["tv_emp"] <= p["tv_bound"] + 1e-9 for p in points]))
    payload = {
        "profile": args.profile,
        "n_tasks": n_tasks,
        "steps": steps,
        "n_samples": n_samples,
        "fraction_below_bound": below,
        "points": points,
    }
    save_json(os.path.join(out_dir, "summary.json"), payload)
    print(f"NE4 done. fraction_below_bound={below:.3f} -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
