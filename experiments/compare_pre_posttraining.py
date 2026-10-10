#!/usr/bin/env python3
"""Pair the same instance/seed jobs before and after LLM post-training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def load_ok(path: Path) -> dict[tuple[str, int], dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {
        (str(row["instance"]), int(row["search_seed"])): row
        for row in rows
        if row.get("status", "ok") == "ok" and row.get("reference_feasible", row.get("feasible", False))
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--instances", type=Path)
    args = parser.parse_args()
    before = load_ok(args.before / "full" / "results.json")
    after = load_ok(args.after / "full" / "results.json")
    allowed = None
    if args.instances:
        allowed = set(json.loads(args.instances.read_text(encoding="utf-8")))
    keys = sorted(
        key for key in before.keys() & after.keys()
        if allowed is None or key[0] in allowed
    )
    deltas = [float(before[key]["final_objective"]) - float(after[key]["final_objective"]) for key in keys]
    cold_deltas = [float(before[key]["initial_objective"]) - float(after[key]["initial_objective"]) for key in keys]
    summary = {
        "paired_feasible_n": len(keys),
        "instance_filter": sorted(allowed) if allowed is not None else None,
        "mean_final_cost_before": mean(float(before[key]["final_objective"]) for key in keys) if keys else None,
        "mean_final_cost_after": mean(float(after[key]["final_objective"]) for key in keys) if keys else None,
        "mean_final_cost_improvement": mean(deltas) if deltas else None,
        "posttrained_win_rate": sum(delta > 0 for delta in deltas) / len(deltas) if deltas else None,
        "mean_cold_start_cost_before": mean(float(before[key]["initial_objective"]) for key in keys) if keys else None,
        "mean_cold_start_cost_after": mean(float(after[key]["initial_objective"]) for key in keys) if keys else None,
        "mean_cold_start_cost_improvement": mean(cold_deltas) if cold_deltas else None,
        "cold_start_posttrained_win_rate": sum(delta > 0 for delta in cold_deltas) / len(cold_deltas) if cold_deltas else None,
        "pairs": [
            {"instance": key[0], "seed": key[1], "final_improvement": delta,
             "cold_start_improvement": cold_delta}
            for key, delta, cold_delta in zip(keys, deltas, cold_deltas)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
