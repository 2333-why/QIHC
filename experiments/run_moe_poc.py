# -*- coding: utf-8 -*-
"""
QIHC MoE routing closed-loop PoC (Tier B default).

Usage (from repository root):
    # Tier B: DistilGPT-2 frontend + p-bit backend (needs: pip install -e ".[llm]")
    python experiments/run_moe_poc.py

    # Tier A: mock frontend, no GPU / transformers
    python experiments/run_moe_poc.py --tier a

    # Tier C preset (gpt2, more experts) — upgrade path
    python experiments/run_moe_poc.py --tier c --steps 600
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qihc.orchestrator import QIHCConfig, QIHCOrchestrator


DEFAULT_PROMPTS = [
    "The mixture-of-experts layer routes tokens to specialized feed-forward networks.",
    "Quantum-inspired probabilistic computing can accelerate discrete routing decisions.",
    "Load balancing across experts prevents collapse onto a single module.",
    "Simulated annealing explores rugged energy landscapes in combinatorial optimization.",
    "A small language model encodes semantic context for expert selection.",
    "Parallel tempering exchanges replicas at different temperatures during sampling.",
    "Heterogeneous systems combine deterministic AI frontends with stochastic backends.",
    "Top-k routing selects a sparse subset of experts per input token.",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QIHC MoE routing PoC")
    p.add_argument("--tier", choices=["a", "b", "c"], default="b")
    p.add_argument("--steps", type=int, default=None, help="override sampling steps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sampler", default=None, help="gibbs|parallel_tempering|sqa|sa_sync|sa_async")
    p.add_argument("--device", default=None, help="cpu|cuda|auto")
    p.add_argument(
        "--output-dir",
        default=os.path.join("experiments", "outputs", "moe_poc"),
    )
    return p.parse_args()


def build_config(args: argparse.Namespace) -> QIHCConfig:
    factories = {"a": QIHCConfig.tier_a, "b": QIHCConfig.tier_b, "c": QIHCConfig.tier_c}
    cfg = factories[args.tier](seed=args.seed)
    if args.steps is not None:
        cfg.sampling_steps = args.steps
    if args.sampler is not None:
        cfg.sampler = args.sampler  # type: ignore[assignment]
    if args.device is not None:
        cfg.device = None if args.device == "auto" else args.device
    return cfg


def plot_comparison(result, out_path: str) -> None:
    greedy_scores = [float(np.dot(d.logits, d.expert_mask)) for d in result.greedy]
    pbit_scores = [float(np.dot(d.logits, d.expert_mask)) for d in result.pbit]
    x = np.arange(len(greedy_scores))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=120)

    axes[0].bar(x - 0.15, greedy_scores, width=0.3, label="Greedy top-k")
    axes[0].bar(x + 0.15, pbit_scores, width=0.3, label="QIHC p-bit")
    axes[0].set_xlabel("Request index")
    axes[0].set_ylabel("Routing score (sum logits)")
    axes[0].set_title("Per-request routing quality")
    axes[0].legend()

    labels = ["Greedy", "QIHC p-bit"]
    lb = [result.metrics["load_balance_greedy"], result.metrics["load_balance_pbit"]]
    axes[1].bar(labels, lb, color=["#3b6fd4", "#e07b26"])
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Load-balance score")
    axes[1].set_title("Batch-level expert load balance")

    fig.suptitle("QIHC MoE Routing PoC")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def main() -> int:
    args = parse_args()
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    cfg = build_config(args)
    print("=== QIHC MoE PoC ===")
    print(json.dumps(cfg.to_summary(), indent=2, ensure_ascii=False))

    orch = QIHCOrchestrator(cfg)
    result = orch.run_batch(DEFAULT_PROMPTS)

    print("\n--- Metrics ---")
    for k, v in result.metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print("  (routing_score_* includes expert capacity penalty when batch_joint=True)")

    out_dir = args.output_dir
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(repo_root, out_dir)
    os.makedirs(out_dir, exist_ok=True)

    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {"metrics": result.metrics, "config": result.config_summary},
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Saved metrics: {metrics_path}")

    plot_comparison(result, os.path.join(out_dir, "moe_poc_comparison.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
