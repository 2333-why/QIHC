#!/usr/bin/env python3
"""Paired, failure-aware report for the hard NL-CVRP experiment arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def rows(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = parser.parse_args()
    data_path = args.run_root / "data" / "hard_nlcvrp.jsonl"
    instances = [json.loads(line)["name"] for line in data_path.read_text(encoding="utf-8").splitlines() if line]
    expected = {(name, seed) for name in instances for seed in args.seeds}
    arms = {}
    for arm in ("full", "no_feedback", "pbit_only", "oracle"):
        records = { (row["instance"], int(row["search_seed"])): row
                    for row in rows(args.run_root / arm / "results.json") }
        failures = { (row["instance"], int(row["search_seed"])): row
                     for row in rows(args.run_root / arm / "failures.json") }
        valid = [records[key] for key in expected if key in records and records[key].get("reference_feasible")]
        gaps = [float(row["optimality_gap"]) for row in valid if row.get("optimality_gap") is not None]
        arms[arm] = {
            "expected": len(expected), "completed": len(expected & records.keys()),
            "failed": len(expected & failures.keys()),
            "reference_feasible_rate": len(valid) / max(len(expected), 1),
            "mean_gap_to_original_cvrp_bks_on_reference_feasible": mean(gaps) if gaps else None,
            "mean_elapsed_s_on_reference_feasible": mean(float(row["total_elapsed_s"]) for row in valid) if valid else None,
            "records": records,
        }
    full = arms["full"]["records"]
    paired = {}
    for arm in ("no_feedback", "pbit_only", "oracle"):
        other = arms[arm]["records"]
        comparable = [key for key in expected if key in full and key in other
                      and full[key].get("reference_feasible") and other[key].get("reference_feasible")
                      and full[key].get("optimality_gap") is not None and other[key].get("optimality_gap") is not None]
        differences = [float(other[key]["optimality_gap"]) - float(full[key]["optimality_gap"])
                       for key in comparable]
        paired["full_vs_" + arm] = {
            "paired_feasible_n": len(comparable),
            "mean_paired_cost_gap_advantage_full": mean(differences) if differences else None,
            "full_win_fraction": sum(delta > 0 for delta in differences) / len(differences) if differences else None,
        }
    cpp = rows(args.run_root / "cpp" / "records.jsonl")
    report = {
        "arms": {name: {key: value for key, value in item.items() if key != "records"}
                 for name, item in arms.items()},
        "paired": paired,
        "cpp": {"n": len(cpp), "exact_match_rate": mean(bool(row.get("exact_match")) for row in cpp) if cpp else None,
                "a4_pass_rate": mean(bool(row.get("validation", {}).get("passed")) for row in cpp) if cpp else None},
        "note": "Missing and failed jobs count against success rate. The public BKS is for the original CVRP, so its gap is a lower-bound comparison, NOT an optimality gap for the augmented NL-CVRP.",
    }
    target = args.run_root / "hard_comparison.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
