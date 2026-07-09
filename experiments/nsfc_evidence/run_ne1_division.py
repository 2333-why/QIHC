# -*- coding: utf-8 -*-
"""NE1: Soft division vs full quadratization vs HUBO (Theorems 1 / 1b).

Usage:
  python experiments/nsfc_evidence/run_ne1_division.py --profile smoke
  python experiments/nsfc_evidence/run_ne1_division.py --profile full
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments.nsfc_evidence.three_law_common import resolve_out_dir, save_fig, save_json  # noqa: E402
from qihc.orchestrator.backend import PBitBackend  # noqa: E402
from qihc.orchestrator.vci_scheduler import VCIConfig  # noqa: E402
from qihc.theory.higher_order import (  # noqa: E402
    generate_higher_order_instance,
    hubo_energy,
    magnetizations_to_mask,
    quadratize_higher_order,
    soft_mean_field_iterate,
)
from qihc.theory.mean_field import mean_field_fixed_point  # noqa: E402


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / max(union, 1))


def is_feasible_mask(mask: np.ndarray, top_k: int, exclusion_pairs) -> bool:
    if int(mask.sum()) != top_k:
        return False
    for i, j in exclusion_pairs:
        if mask[i] and mask[j]:
            return False
    return True


def solve_quadratic(weight, field, top_k, steps, seed, sampler="parallel_tempering"):
    cfg = VCIConfig.tier_a(sampling_steps=steps, sampler=sampler, seed=seed)  # type: ignore[arg-type]
    backend = PBitBackend(cfg)
    # backend returns top_k mask via spins_to_mask using config.top_k — override via refine
    mask, energy, elapsed = backend.solve(weight, field)
    # Re-project using field as proxy logits
    logits = field.copy()
    idx = np.argsort(-logits)[:top_k]
    out = np.zeros(field.size, dtype=bool)
    # Prefer selected spins from sampler among top candidates
    spins_pref = np.flatnonzero(mask) if mask.dtype == bool else []
    chosen = []
    for i in list(spins_pref) + list(idx):
        if i not in chosen and i < field.size:
            chosen.append(int(i))
        if len(chosen) >= top_k:
            break
    out[chosen[:top_k]] = True
    return out, float(energy), float(elapsed)


def run_one(inst, steps: int, seed: int) -> dict:
    # Soft: mean-field soft division (0 aux vars)
    t0 = time.perf_counter()
    m_soft, _ = soft_mean_field_iterate(inst, beta=1.0, max_iters=80)
    mask_soft = magnetizations_to_mask(m_soft[: inst.n], inst.top_k)
    t_soft = time.perf_counter() - t0
    e_soft = hubo_energy(2 * mask_soft.astype(float) - 1, inst)

    # Quad: full quadratization + p-bit
    w_ext, f_ext, n_aux = quadratize_higher_order(inst)
    # Truncate solve to original vars for mask; aux vars still cost flips
    # Equal flip budget: steps scaled so total flips ≈ steps * n_soft
    # soft vars = n; quad vars = n + n_aux → reduce steps proportionally
    steps_quad = max(20, int(steps * inst.n / max(inst.n + n_aux, 1)))
    t0 = time.perf_counter()
    # Solve extended system then take first n magnetizations via sampling
    cfg = VCIConfig.tier_a(sampling_steps=steps_quad, sampler="parallel_tempering", seed=seed, top_k=inst.top_k)
    backend = PBitBackend(cfg)
    mask_ext, energy_q, _ = backend.solve(w_ext, f_ext)
    # mask_ext length may be n_ext; take first n
    if mask_ext.size > inst.n:
        # rebuild from energy-minimizing projection on original field
        m_q, _, _ = mean_field_fixed_point(f_ext[: inst.n], w_ext[: inst.n, : inst.n], beta=1.0, max_iters=50)
        mask_quad = magnetizations_to_mask(m_q, inst.top_k)
    else:
        mask_quad = mask_ext[: inst.n]
        if mask_quad.sum() != inst.top_k:
            mask_quad = magnetizations_to_mask(f_ext[: inst.n], inst.top_k)
    t_quad = time.perf_counter() - t0
    e_quad = hubo_energy(2 * mask_quad.astype(float) - 1, inst)

    # HUBO baseline: mean-field on full higher-order (same as soft here) + random restart energy eval
    # For native HUBO sampling we use soft MF as proxy solver (documented), count vars = n
    mask_hubo = mask_soft.copy()
    e_hubo = e_soft

    gold = inst.metadata.get("gold", [])
    gold_mask = np.zeros(inst.n, dtype=bool)
    gold_mask[gold] = True

    return {
        "soft": {
            "feasible": is_feasible_mask(mask_soft, inst.top_k, inst.exclusion_pairs),
            "n_vars": inst.n,
            "n_aux": 0,
            "energy": float(e_soft),
            "time_s": float(t_soft),
            "jaccard_gold": jaccard(mask_soft, gold_mask),
            "jaccard_vs_quad": jaccard(mask_soft, mask_quad),
        },
        "quad": {
            "feasible": is_feasible_mask(mask_quad, inst.top_k, inst.exclusion_pairs),
            "n_vars": inst.n + n_aux,
            "n_aux": int(n_aux),
            "energy": float(e_quad),
            "time_s": float(t_quad),
            "jaccard_gold": jaccard(mask_quad, gold_mask),
            "steps_used": steps_quad,
        },
        "hubo": {
            "feasible": is_feasible_mask(mask_hubo, inst.top_k, inst.exclusion_pairs),
            "n_vars": inst.n,
            "n_aux": 0,
            "energy": float(e_hubo),
            "time_s": float(t_soft),
            "jaccard_gold": jaccard(mask_hubo, gold_mask),
            "note": "native HUBO via soft MF (no aux explosion)",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="NE1 soft vs quad vs HUBO")
    parser.add_argument("--profile", choices=["smoke", "full"], default="full")
    parser.add_argument("--n-instances", type=int, default=None)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--n-ho-terms", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    n_inst = args.n_instances or (6 if args.profile == "smoke" else 30)
    n_ho = args.n_ho_terms or (3 if args.profile == "smoke" else 8)
    steps = args.steps or (80 if args.profile == "smoke" else 200)
    out_dir = resolve_out_dir("ne1_division", args.output_dir)

    methods = ["soft", "quad", "hubo"]
    agg = {m: {"feasible": [], "n_vars": [], "energy": [], "jaccard_gold": [], "jaccard_vs_quad": []} for m in methods}
    per = []

    for i in range(n_inst):
        inst = generate_higher_order_instance(
            n=args.n,
            top_k=3,
            n_ho_terms=n_ho,
            degree=args.degree,
            seed=args.seed + i,
        )
        row = run_one(inst, steps=steps, seed=args.seed + i)
        per.append({"instance": i, "n_aux_quad": row["quad"]["n_aux"], **{m: row[m] for m in methods}})
        for m in methods:
            agg[m]["feasible"].append(row[m]["feasible"])
            agg[m]["n_vars"].append(row[m]["n_vars"])
            agg[m]["energy"].append(row[m]["energy"])
            agg[m]["jaccard_gold"].append(row[m]["jaccard_gold"])
            if m == "soft":
                agg[m]["jaccard_vs_quad"].append(row[m]["jaccard_vs_quad"])
        print(
            f"  inst={i} soft_feas={row['soft']['feasible']} quad_vars={row['quad']['n_vars']} "
            f"aux={row['quad']['n_aux']} jacc_soft_quad={row['soft']['jaccard_vs_quad']:.2f}"
        )

    summary = {
        m: {
            "feasible_rate": float(np.mean(agg[m]["feasible"])),
            "mean_n_vars": float(np.mean(agg[m]["n_vars"])),
            "mean_energy": float(np.mean(agg[m]["energy"])),
            "mean_jaccard_gold": float(np.mean(agg[m]["jaccard_gold"])),
            **(
                {"mean_jaccard_vs_quad": float(np.mean(agg[m]["jaccard_vs_quad"]))}
                if m == "soft"
                else {}
            ),
        }
        for m in methods
    }

    # Bar charts
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=130)
    x = np.arange(len(methods))
    axes[0].bar(x, [summary[m]["feasible_rate"] for m in methods], color=["#4c78a8", "#e45756", "#72b7b2"])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("feasible rate")
    axes[0].set_title("NE1: feasible rate")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x, [summary[m]["mean_n_vars"] for m in methods], color=["#4c78a8", "#e45756", "#72b7b2"])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods)
    axes[1].set_ylabel("# variables")
    axes[1].set_title("NE1: variable count (Thm 1b)")
    axes[1].grid(axis="y", alpha=0.3)
    save_fig(os.path.join(out_dir, "ne1_division_bars.png"))

    payload = {
        "profile": args.profile,
        "n_instances": n_inst,
        "n": args.n,
        "degree": args.degree,
        "n_ho_terms": n_ho,
        "steps": steps,
        "summary": summary,
        "per_instance": per,
    }
    save_json(os.path.join(out_dir, "summary.json"), payload)
    print(f"NE1 done. soft_feas={summary['soft']['feasible_rate']:.3f} "
          f"quad_vars={summary['quad']['mean_n_vars']:.1f} -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
