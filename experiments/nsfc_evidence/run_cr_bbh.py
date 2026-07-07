# -*- coding: utf-8 -*-
"""
CR-protocol aligned BBH experiment (arXiv:2407.00071).

Paper-aligned modes (NOT logits-greedy):
  zeroshot   — LLM T=0 direct answer
  linear     — N sampled completions → majority vote
  quadratic  — sample → QUBO reason select → enhanced prompt → LLM T=0
  vci-1/2    — CR-encoded constrained cooperative inference (QIHC contribution)

Usage:
    # CPU smoke (mock LLM from logits)
    python experiments/nsfc_evidence/run_cr_bbh.py --source bundled

    # Server: real LLM + HF BBH (paper-comparable)
    python experiments/nsfc_evidence/run_cr_bbh.py \\
        --source hf --use-llm --model-name Qwen/Qwen2.5-7B-Instruct \\
        --n-samples 50 --limit-per-task 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from qihc.orchestrator.bbh import load_bbh_tasks  # noqa: E402
from qihc.orchestrator.bbh_parser import DEFAULT_BBH_HF_TASKS  # noqa: E402
from qihc.orchestrator.cr_pipeline import CRPaperMode, run_cr_paper_benchmark  # noqa: E402

DEFAULT_MODES: list[CRPaperMode] = [
    "zeroshot",
    "linear",
    "quadratic",
    "vci-1",
    "vci-2",
]


def run_cr_benchmark(
    tasks,
    modes: list[CRPaperMode],
    budget_steps: int,
    n_samples: int,
    seed: int,
    use_llm: bool,
    model_name: str,
) -> dict[str, Any]:
    """Backward-compatible wrapper around paper-aligned benchmark."""
    return run_cr_paper_benchmark(
        tasks=tasks,
        modes=modes,
        budget_steps=budget_steps,
        n_samples=n_samples,
        seed=seed,
        use_llm=use_llm,
        model_name=model_name,
    )


def plot_cr_summary(summary: dict, out_path: str, title: str) -> None:
    modes = list(summary.keys())
    acc = [summary[m]["accuracy"] for m in modes]
    feas = [summary[m]["feasible_rate"] for m in modes]
    gain = [summary[m].get("gain_over_zeroshot", 0.0) for m in modes]
    x = np.arange(len(modes))
    w = 0.28
    fig, ax1 = plt.subplots(figsize=(10, 4.8), dpi=120)
    ax1.bar(x - w, acc, w, label="Accuracy", color="#4c78a8")
    ax1.bar(x, feas, w, label="Feasible rate", color="#72b7b2")
    ax1.bar(x + w, gain, w, label="Gain vs zeroshot", color="#f58518")
    ax1.set_xticks(x)
    ax1.set_xticklabels(modes, rotation=15)
    ax1.set_ylim(-0.15, 1.05)
    ax1.set_ylabel("Rate / gain")
    ax1.set_title(title)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(axis="y", alpha=0.3)
    ax1.axhline(0, color="gray", linewidth=0.8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="CR paper-aligned BBH benchmark")
    parser.add_argument("--source", choices=["bundled", "hf"], default="bundled")
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--n-samples", type=int, default=50, help="CR reason samples per question")
    parser.add_argument("--budget-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--limit-per-task", type=int, default=25)
    parser.add_argument("--hf-tasks", nargs="*", default=None)
    parser.add_argument("--modes", nargs="*", default=list(DEFAULT_MODES))
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    out_dir = args.output_dir or os.path.join(REPO_ROOT, "experiments", "outputs", "nsfc_evidence", "cr_protocol")
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(REPO_ROOT, out_dir)

    tasks = load_bbh_tasks(
        source=args.source,
        hf_tasks=args.hf_tasks or (DEFAULT_BBH_HF_TASKS[:5] if args.source == "hf" else None),
        limit_per_task=args.limit_per_task if args.source == "hf" else None,
    )
    if args.limit:
        tasks = tasks[: args.limit]

    print(f"CR paper benchmark: {len(tasks)} tasks, n_samples={args.n_samples}, use_llm={args.use_llm}")
    if not args.use_llm:
        print("  [smoke] mock LLM frontend — use --use-llm for paper-comparable accuracy")

    data = run_cr_benchmark(
        tasks,
        modes=args.modes,  # type: ignore[arg-type]
        budget_steps=args.budget_steps,
        n_samples=args.n_samples,
        seed=args.seed,
        use_llm=args.use_llm,
        model_name=args.model_name,
    )

    payload = {
        "experiment": "cr_paper_bbh",
        "source": args.source,
        "use_llm": args.use_llm,
        "model_name": args.model_name if args.use_llm else None,
        "modes": args.modes,
        **data,
        "reference": "Combinatorial Reasoning arXiv:2407.00071",
    }

    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "cr_protocol.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    plot_cr_summary(
        data["summary"],
        os.path.join(out_dir, "cr_protocol.png"),
        f"CR paper protocol ({args.source}, n={len(tasks)})",
    )

    zs = data["summary"].get("zeroshot", {}).get("accuracy", 0.0)
    print(f"\n=== CR paper summary (n={len(tasks)}, zeroshot={zs:.2%}) ===")
    for mode, s in data["summary"].items():
        gain = s.get("gain_over_zeroshot", 0.0)
        print(
            f"  {mode:12s} acc={s['accuracy']:.2%} exact={s.get('exact_match_rate', s['accuracy']):.2%} "
            f"feas={s['feasible_rate']:.2%} "
            f"gain={gain:+.2%} llm_calls≈{s.get('mean_llm_calls', 0):.0f}"
        )
    if data.get("llm_stats"):
        print(f"  LLM stats: {data['llm_stats']}")
    print(f"Saved: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
