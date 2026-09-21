#!/usr/bin/env python3
"""Failure-aware comparison of hard CVRP/NL-CVRP methods."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, list) else []


def manifest_config(path: Path) -> dict:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value.get("config", {}) if isinstance(value, dict) else {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    args = parser.parse_args()
    data = [json.loads(line) for line in (args.run_root / "data" / "hard_nlcvrp.jsonl").read_text(encoding="utf-8").splitlines() if line]
    expected = {(row["name"], seed) for row in data for seed in args.seeds}
    sources = {
        "qihc_full": ("full", "llm"),
        "qihc_no_feedback": ("no_feedback", "llm"),
        "pbit_only": ("pbit_only", "knn"),
        "random_neighborhood_pbit": ("random_pbit", "random"),
        "oracle_constraints_qihc": ("oracle", "llm"),
        "greedy_oracle": ("baselines", "greedy"),
        "ortools_gls_oracle": ("baselines", "ortools"),
        "hgs_constraint_unaware": ("baselines", "hgs"),
        "llm_direct_unrepaired": ("baselines", "llm_direct"),
    }
    all_records: dict[str, dict[tuple[str, int], dict]] = {}
    report: dict[str, dict] = {}
    for label, (directory, method) in sources.items():
        rows = [row for row in load(args.run_root / directory / "results.json") if row.get("method") == method]
        records = {(row["instance"], int(row["search_seed"])): row for row in rows}
        failures = [row for row in load(args.run_root / directory / "failures.json") if row.get("method") == method]
        feasible = [records[key] for key in expected if key in records and records[key].get("reference_feasible", records[key].get("feasible", False))]
        gaps = [float(row["optimality_gap"]) for row in feasible if row.get("optimality_gap") is not None]
        all_records[label] = records
        report[label] = {
            "expected": len(expected),
            "completed": len(expected & records.keys()),
            "failed_jobs": len(failures),
            "strict_feasible_rate": len(feasible) / max(len(expected), 1),
            "mean_objective_on_all_recorded": (
                mean(float(row["final_objective"]) for row in records.values()) if records else None
            ),
            "mean_cost_on_feasible": mean(float(row["final_objective"]) for row in feasible) if feasible else None,
            "mean_gap_to_unaugmented_cvrp_bks_on_feasible": mean(gaps) if gaps else None,
            "mean_elapsed_s_on_completed": mean(float(row["total_elapsed_s"]) for row in records.values()) if records else None,
            "mean_strict_violation_count_on_recorded": (
                mean(len(row.get("reference_violations", [])) for row in records.values()) if records else None
            ),
        }
    full = all_records["qihc_full"]
    paired = {}
    for label, records in all_records.items():
        if label == "qihc_full":
            continue
        keys = [key for key in expected if key in full and key in records
                and full[key].get("reference_feasible", full[key].get("feasible", False))
                and records[key].get("reference_feasible", records[key].get("feasible", False))]
        deltas = [float(records[key]["final_objective"]) - float(full[key]["final_objective"]) for key in keys]
        paired[f"qihc_full_vs_{label}"] = {
            "paired_feasible_n": len(keys),
            "mean_cost_advantage_qihc": mean(deltas) if deltas else None,
            "qihc_win_rate": sum(delta > 0 for delta in deltas) / len(deltas) if deltas else None,
            "tie_rate": sum(abs(delta) <= 1e-9 for delta in deltas) / len(deltas) if deltas else None,
        }
    output = {
        "benchmark": {
            "instances": len(data), "seeds": args.seeds,
            "customer_range": [min(len(x["customers"]) for x in data), max(len(x["customers"]) for x in data)],
            "constraint_counts": sorted({len(x.get("constraints", [])) for x in data}),
        },
        "recorded_configuration": {
            "qihc_full": manifest_config(args.run_root / "full" / "manifest_rank0.json"),
            "baselines": manifest_config(args.run_root / "baselines" / "manifest_rank0.json"),
        },
        "methods": report,
        "paired": paired,
        "interpretation": {
            "llm_direct_unrepaired": "The LLM sees the numerical instance plus natural-language requirements and emits complete routes. No p-bit, solver, completion, or repair is used.",
            "hgs_constraint_unaware": "HGS optimizes the original CVRP and does not receive the added language constraints; the common strict verifier still checks them.",
            "bks_gap": "The BKS belongs to the original unaugmented CVRP and is only a lower-bound reference, not an optimality gap for NL-CVRP.",
        },
    }
    target = args.run_root / "method_comparison.json"
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
