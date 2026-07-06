# -*- coding: utf-8 -*-
"""
Case E (P1): Complete CR baselines on bundled 40 (zero-shot | random | linear | quadratic | vci-1 | vci-2).

Ensures random/linear are present on the canonical bundled constraint set.
"""
from __future__ import annotations

import argparse
import sys

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.nsfc_evidence.common import plot_dual_bars, resolve_out_dir, save_json  # noqa: E402
from experiments.nsfc_evidence.run_cr_bbh import run_cr_benchmark  # noqa: E402
from qihc.orchestrator.bbh import load_bbh_tasks  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Case E: full CR baselines on bundled")
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--budget-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    out_dir = resolve_out_dir("case_e_cr_bundled_full", args.output_dir)
    tasks = load_bbh_tasks(source="bundled")
    modes = ["zeroshot", "random", "linear", "quadratic", "vci-1", "vci-2"]
    data = run_cr_benchmark(
        tasks,
        modes=modes,
        budget_steps=args.budget_steps,
        n_samples=args.n_samples,
        seed=args.seed,
        use_llm=False,
        model_name="",
    )

    payload = {
        "experiment": "case_e_cr_bundled_full",
        "case": "P1-E",
        "description": "Full CR baselines including random/linear on bundled n=40",
        "n_tasks": len(tasks),
        "modes": modes,
        **data,
    }
    save_json(f"{out_dir}/cr_protocol.json", payload)
    plot_dual_bars(
        {m: {"feasible_rate": s["feasible_rate"], "exact_match_rate": s["accuracy"]} for m, s in data["summary"].items()},
        f"{out_dir}/cr_protocol.png",
        f"Case E: CR bundled full (n={len(tasks)})",
    )

    print(f"\n=== Case E CR bundled full (n={len(tasks)}) ===")
    for mode, s in data["summary"].items():
        print(f"  {mode:12s} acc={s['accuracy']:.2%} feas={s['feasible_rate']:.2%}")
    print(f"Saved: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
