#!/usr/bin/env python3
"""Bootstrap confidence intervals and paired comparisons for QIHC-LNS results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, samples: int = 10000) -> list[float]:
    if values.size == 0:
        return [float("nan"), float("nan")]
    indices = rng.integers(0, values.size, size=(samples, values.size))
    means = values[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--reference", default="knn")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rows = json.loads((args.result_dir / "results.json").read_text(encoding="utf-8"))
    rows = [row for row in rows if row.get("status") == "ok"]
    rng = np.random.default_rng(args.seed)
    by_method: dict[str, list[dict]] = {}
    for row in rows:
        by_method.setdefault(row["method"], []).append(row)
    report = {"methods": {}, "paired_vs_reference": {}}
    for method, records in by_method.items():
        values = np.asarray([row["improvement_fraction"] for row in records], dtype=float)
        report["methods"][method] = {
            "n": len(records),
            "mean_improvement_fraction": float(values.mean()),
            "improvement_ci95": bootstrap_ci(values, rng, args.bootstrap_samples),
            "feasible_rate": float(np.mean([row["feasible"] for row in records])),
            "mean_time_s": float(np.mean([row["total_elapsed_s"] for row in records])),
        }
    reference = {
        (row["instance"], row["search_seed"]): row for row in by_method.get(args.reference, [])
    }
    for method, records in by_method.items():
        if method == args.reference:
            continue
        differences = []
        for row in records:
            key = (row["instance"], row["search_seed"])
            if key in reference:
                differences.append(row["improvement_fraction"] - reference[key]["improvement_fraction"])
        values = np.asarray(differences, dtype=float)
        report["paired_vs_reference"][method] = {
            "reference": args.reference,
            "n_pairs": int(values.size),
            "mean_difference": float(values.mean()) if values.size else float("nan"),
            "difference_ci95": bootstrap_ci(values, rng, args.bootstrap_samples),
            "win_rate": float(np.mean(values > 0)) if values.size else float("nan"),
        }
    target = args.result_dir / "statistical_report.json"
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
