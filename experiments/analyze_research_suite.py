#!/usr/bin/env python3
"""Failure-aware, instance-clustered paired reporting for the research suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import numpy as np


def read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def primal_integral(row: dict, horizon: float) -> float:
    """Integral of objective / shared initial objective; lower is better."""
    initial = float(row["initial_objective"])
    previous_time, objective, integral = 0.0, initial, 0.0
    for record in row.get("records", []):
        timestamp = min(horizon, max(previous_time, float(record.get("elapsed_s", 0.0))))
        integral += (timestamp - previous_time) * objective / max(abs(initial), 1e-12)
        objective = float(record["incumbent_after"])
        previous_time = timestamp
    integral += (horizon - previous_time) * objective / max(abs(initial), 1e-12)
    return integral / max(horizon, 1e-12)


def analyze(root: Path) -> dict:
    manifest = json.loads((root / "suite_manifest.json").read_text(encoding="utf-8"))
    expected = {(name, int(seed)) for name in manifest["test_instances"] for seed in manifest["search_seeds"]}
    methods = {}
    raw = {}
    for label in manifest["evaluation_arms"]:
        directory = root / "evaluation" / label
        rows = read(directory / "results.json")
        indexed = {(row["instance"], int(row["search_seed"])): row for row in rows}
        valid = {key: row for key, row in indexed.items() if key in expected and row.get("reference_feasible", row.get("feasible", False))}
        raw[label] = valid
        diagnostics = [record.get("feedback_diagnostics", {}) for row in valid.values() for record in row.get("records", [])]
        methods[label] = {
            "expected_jobs": len(expected), "completed": len(expected & indexed.keys()),
            "strict_feasible": len(valid), "strict_feasible_rate": len(valid) / max(len(expected), 1),
            "failed_jobs": len(read(directory / "failures.json")),
            "mean_cost_feasible": mean(row["final_objective"] for row in valid.values()) if valid else None,
            "mean_total_elapsed_s": mean(row["total_elapsed_s"] for row in indexed.values()) if indexed else None,
            "mean_primal_integral": mean(primal_integral(row, float(row.get("config", {}).get("search_time_limit_s") or (row.get("records") or [{}])[-1].get("elapsed_s", 1.0) or 1.0))
                                         for row in valid.values() if row.get("initial_objective") is not None) if any(row.get("initial_objective") is not None for row in valid.values()) else None,
            "mean_unique_assignments": mean(d.get("unique_assignments", 0) for d in diagnostics) if diagnostics else None,
            "mean_effective_sample_size": mean(d.get("effective_sample_size", 0) for d in diagnostics) if diagnostics else None,
            "budget_overshoot_s": max((max(0.0, float(row.get("search_elapsed_s", 0.0)) - float(row.get("config", {}).get("search_time_limit_s", manifest["search_time_limit_s"])))
                                       for row in valid.values() if row.get("config", {}).get("search_time_limit_s", manifest["search_time_limit_s"])), default=0.0),
            "stop_reasons": {reason: sum(row.get("stop_reason") == reason for row in valid.values())
                             for reason in {row.get("stop_reason", "unknown") for row in valid.values()}},
            "by_size": {str(size): {
                "feasible_jobs": sum(row.get("n_customers") == size for row in valid.values()),
                "mean_cost": mean(row["final_objective"] for row in valid.values() if row.get("n_customers") == size),
            } for size in {row.get("n_customers", 0) for row in valid.values()} if any(row.get("n_customers") == size for row in valid.values())},
        }
    paired = {}
    rng = np.random.default_rng(2026)
    for label, values in raw.items():
        if label == "population":
            continue
        reference_label = "population"
        if label in {"cp_sat_pool", "metropolis_pool", "uniform_pool"}:
            reference_label = "pbit_qubo"
        elif label.startswith("budget_") and label.endswith("_cp_sat"):
            reference_label = label.removesuffix("_cp_sat") + "_pbit_qubo"
        elif label.startswith("budget_") and label.endswith("_no_feedback"):
            reference_label = label.removesuffix("_no_feedback") + "_population"
        reference = raw.get(reference_label, {})
        keys = sorted(reference.keys() & values.keys())
        grouped = {}
        init_mismatches = []
        for key in keys:
            left, right = reference[key], values[key]
            gain = (float(right["final_objective"]) - float(left["final_objective"])) / max(abs(float(right["final_objective"])), 1e-12)
            grouped.setdefault(key[0], []).append(gain)
            if (left.get("initial_objective") is None or right.get("initial_objective") is None
                    or abs(float(left["initial_objective"]) - float(right["initial_objective"])) > 1e-7):
                init_mismatches.append(key)
        gains = np.asarray([mean(v) for v in grouped.values()])
        interval = None; pvalue = None
        if len(gains) >= 2:
            draws = rng.choice(gains, size=(4000, len(gains)), replace=True).mean(axis=1)
            interval = np.quantile(draws, [0.025, 0.975]).tolist()
            try:
                from scipy.stats import wilcoxon
                pvalue = float(wilcoxon(gains).pvalue) if np.any(gains) else 1.0
            except ImportError:
                pass
        paired[label] = {
            "paired_jobs": len(keys), "independent_instances": len(grouped),
            "mean_relative_advantage_population": float(gains.mean()) if len(gains) else None,
            "instance_cluster_bootstrap_95ci": interval, "wilcoxon_instance_p": pvalue,
            "instance_win_rate_population": float(np.mean(gains > 0)) if len(gains) else None,
            "shared_initial_mismatches": init_mismatches,
            "complete_coverage": set(keys) == expected,
            "reference_arm": reference_label,
            "comparison_kind": "paired_shared_initial" if not init_mismatches else "end_to_end_different_initialization",
            "mean_relative_advantage_reference": float(gains.mean()) if len(gains) else None,
            "by_size": {str(size): mean(
                (float(values[key]["final_objective"]) - float(reference[key]["final_objective"])) / max(abs(float(values[key]["final_objective"])), 1e-12)
                for key in keys if reference[key].get("n_customers") == size)
                for size in {reference[key].get("n_customers", 0) for key in keys}
                if any(reference[key].get("n_customers") == size for key in keys)},
        }
    tests = sorted((value["wilcoxon_instance_p"], label) for label, value in paired.items() if value["wilcoxon_instance_p"] is not None)
    previous = 0.0
    for rank, (pvalue, label) in enumerate(tests):
        previous = max(previous, min(1.0, (len(tests) - rank) * pvalue))
        paired[label]["holm_adjusted_p"] = previous
    report = {"profile": manifest["profile"], "methods": methods, "paired": paired,
              "semantic_robustness": {path.parent.name: json.loads(path.read_text(encoding="utf-8"))
                                      for path in (root / "semantic_robustness").glob("*/summary.json")},
              "statistical_unit": "numerical instance; search seeds averaged within instance",
              "hardware_claim": "software simulations only; no physical p-bit energy measurements",
              "smoke_uses_llm": manifest.get("uses_llm", False),
              "missing_or_infeasible_jobs_remain_in_feasibility_denominator": True}
    (root / "research_results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"arms": len(methods), "paired_tests": len(paired)}))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    analyze(parser.parse_args().root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
