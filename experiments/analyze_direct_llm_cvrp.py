#!/usr/bin/env python3
"""Failure-aware report for the standalone direct-LLM NL-CVRP experiment."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean


def load_json(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    args = parser.parse_args()
    instances = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected = {(row["name"], seed) for row in instances for seed in args.seeds}
    records = [row for row in load_json(args.output / "results.json", []) if row.get("method") == "llm_direct"]
    failures = [row for row in load_json(args.output / "failures.json", []) if row.get("method") == "llm_direct"]
    indexed = {(row["instance"], int(row["search_seed"])): row for row in records}
    feasible = [row for row in indexed.values() if row.get("reference_feasible", row.get("feasible", False))]
    gaps = [float(row["optimality_gap"]) for row in feasible if row.get("optimality_gap") is not None]
    violation_types = Counter(
        violation.get("type", "unknown")
        for row in records
        for violation in row.get("reference_violations", [])
    )
    report = {
        "protocol": {
            "method": "llm_direct_unrepaired",
            "definition": "The LLM receives numerical CVRP data and natural-language requirements, emits full routes, and is strictly verified without p-bit, solver, completion, or repair.",
            "bks_note": "CVRPLIB BKS is for the original unaugmented CVRP and is only a lower-bound reference for the augmented NL-CVRP.",
        },
        "coverage": {
            "instances": len(instances),
            "seeds": args.seeds,
            "expected_jobs": len(expected),
            "completed_jobs": len(indexed),
            "failed_jobs": len(failures),
            "missing_jobs": len(expected - indexed.keys()),
        },
        "outcomes": {
            "strict_feasible_rate": len(feasible) / max(len(expected), 1),
            "mean_objective_on_strict_feasible": mean(float(row["final_objective"]) for row in feasible) if feasible else None,
            "mean_bks_lower_bound_gap_on_strict_feasible": mean(gaps) if gaps else None,
            "mean_generation_elapsed_s_on_completed": mean(float(row["total_elapsed_s"]) for row in indexed.values()) if indexed else None,
            "strict_violation_type_counts": dict(sorted(violation_types.items())),
        },
    }
    target = args.output.parent / "direct_llm_report.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
