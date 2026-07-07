# -*- coding: utf-8 -*-
"""
Case H (P2): 7B vs 14B — CR paper baselines + VCI-2 on synthetic constrained set.
"""
from __future__ import annotations

import argparse
import os
import sys

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.nsfc_evidence.common import (  # noqa: E402
    load_synthetic_problems,
    resolve_out_dir,
    run_cr_paper_modes,
    save_json,
)
from qihc.orchestrator.bbh_llm import enrich_problems_with_llm_logits  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Case H: 7B vs 14B CR+VCI compare")
    parser.add_argument("--n-tasks", type=int, default=200)
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--model-7b", default=None)
    parser.add_argument("--model-14b", default=None)
    parser.add_argument("--budget-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--modes",
        nargs="*",
        default=["zeroshot", "quadratic", "vci-2"],
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    model_7b = args.model_7b or os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
    model_14b = args.model_14b or os.environ.get("MODEL_NAME_14B", "Qwen/Qwen2.5-14B-Instruct")

    out_dir = resolve_out_dir("case_h_model_compare", args.output_dir)
    base = load_synthetic_problems(n_tasks=args.n_tasks, seed=args.seed)

    results: dict[str, dict] = {}
    for label, model in [("7B", model_7b), ("14B", model_14b)]:
        print(f"\n--- CR+VCI with {label}: {model} ---")
        problems = enrich_problems_with_llm_logits(base, model_name=model)
        summary = run_cr_paper_modes(
            problems,
            list(args.modes),
            args.budget_steps,
            args.n_samples,
            args.seed,
            use_llm=False,
            model_name=model,
        )
        results[label] = {"model_name": model, "summary": summary}

    payload = {
        "experiment": "case_h_model_compare",
        "case": "P2-H",
        "description": "7B vs 14B CR paper + VCI on synthetic constrained (LLM logits)",
        "n_tasks": len(base),
        "modes": list(args.modes),
        "baseline": "zeroshot (CR paper mock on LLM logits)",
        "models": results,
    }
    save_json(f"{out_dir}/model_compare.json", payload)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = list(results.keys())
    x = np.arange(len(args.modes))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    for i, label in enumerate(labels):
        offset = (i - 0.5) * w
        vals = [results[label]["summary"][m]["feasible_rate"] for m in args.modes]
        ax.bar(x + offset, vals, w, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(args.modes)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Feasible rate")
    ax.set_title(f"Case H: model compare (n={len(base)})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/model_compare.png", bbox_inches="tight")
    plt.close()

    print(f"\n=== Case H model compare (n={len(base)}) ===")
    for label in labels:
        print(f"  Model {label} ({results[label]['model_name']}):")
        for mode, s in results[label]["summary"].items():
            print(
                f"    {mode:12s} acc={s['accuracy']:.2%} feas={s['feasible_rate']:.2%} "
                f"gain={s['gain_over_zeroshot']:+.2%}"
            )
    print(f"Saved: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
