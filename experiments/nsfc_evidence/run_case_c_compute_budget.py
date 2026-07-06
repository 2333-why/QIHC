# -*- coding: utf-8 -*-
"""
Case C' (P0): Same-compute budget table — aggregate from run_dir artifacts or inline run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.nsfc_evidence.common import resolve_out_dir, run_vci_modes, save_json, strip_per_task  # noqa: E402


def _load_json(path: Path) -> dict | None:
    if path.is_file():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def _rows_from_dual(data: dict, arm: str) -> list[dict]:
    rows = []
    for mode, s in data.get("summary", {}).items():
        rows.append(
            {
                "arm": arm,
                "mode": mode,
                "llm_calls": s.get("llm_calls", 1),
                "mean_pbit_steps": s.get("mean_pbit_steps", 0),
                "feasible_rate": s.get("feasible_rate"),
                "exact_match_rate": s.get("exact_match_rate"),
                "accuracy": s.get("accuracy"),
            }
        )
    return rows


def _rows_from_cr(data: dict, arm: str) -> list[dict]:
    rows = []
    for mode, s in data.get("summary", {}).items():
        rows.append(
            {
                "arm": arm,
                "mode": mode,
                "llm_calls": data.get("n_samples", 1) if mode not in ("zeroshot", "random") else 1,
                "mean_pbit_steps": s.get("mean_pbit_steps", 0),
                "feasible_rate": s.get("feasible_rate"),
                "accuracy": s.get("accuracy"),
            }
        )
    return rows


def aggregate_from_run_dir(run_dir: Path) -> list[dict]:
    rows: list[dict] = []
    mapping = [
        ("dual_bundled/dual_axis.json", "bundled_40"),
        ("case_b_dual_synthetic/dual_axis.json", "synthetic_200"),
        ("cr_bundled/cr_protocol.json", "cr_bundled_40"),
        ("case_a_cr_subset/cr_protocol.json", "cr_synthetic_200"),
    ]
    for rel, arm in mapping:
        data = _load_json(run_dir / rel)
        if not data:
            continue
        if "dual_axis" in rel or "dual_axis.json" in str(data.get("experiment", "")):
            rows.extend(_rows_from_dual(data, arm))
        else:
            rows.extend(_rows_from_cr(data, arm))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Case C': compute budget table")
    parser.add_argument("--run-dir", default=None, help="Aggregate from existing suite run")
    parser.add_argument("--inline", action="store_true", help="Run quick bundled inline if no run-dir")
    parser.add_argument("--budget-steps", type=int, default=400)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    out_dir = resolve_out_dir("case_c_compute_budget", args.output_dir)
    rows: list[dict] = []

    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = REPO_ROOT / run_dir
        rows = aggregate_from_run_dir(run_dir)

    if not rows and args.inline:
        from qihc.orchestrator.bbh import load_bbh_problems

        problems = load_bbh_problems(source="bundled", seed=0)
        summary = strip_per_task(run_vci_modes(problems, ["greedy", "vci-1", "vci-2"], args.budget_steps, 0))
        rows = _rows_from_dual({"summary": summary}, "bundled_inline")

    payload = {
        "experiment": "case_c_compute_budget",
        "case": "P0-C",
        "description": "Same-compute budget: LLM calls × p-bit steps vs quality/feasibility",
        "rows": rows,
        "note": "vci-2 uses steps/2 per round × 2 rounds ≈ vci-1 total p-bit budget",
    }
    save_json(f"{out_dir}/compute_budget.json", payload)

    print(f"\n=== Case C' compute budget ({len(rows)} rows) ===")
    for r in rows:
        print(
            f"  [{r['arm']:18s}] {r['mode']:10s} "
            f"llm={r['llm_calls']} pbit≈{r.get('mean_pbit_steps', 0):.0f} "
            f"feas={r.get('feasible_rate', 0):.2%}"
        )
    print(f"Saved: {out_dir}/compute_budget.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
