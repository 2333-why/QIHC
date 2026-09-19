#!/usr/bin/env python3
"""Check completed experiment jobs against the exact requested job grid."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_cvrp_lns import job_key, read_valid_jsonl
from qihc.problems.cvrp.instance import load_jsonl


def check_collection(data: Path, output: Path, seeds: list[int], methods: list[str]) -> dict:
    instances = load_jsonl(data)
    names = [instance.name for instance in instances]
    if len(names) != len(set(names)):
        raise ValueError("Dataset contains duplicate instance names")
    expected = {(name, method, seed) for name in names for method in methods for seed in seeds}
    rows = [
        row
        for path in sorted(output.glob("results_rank*.jsonl"))
        for row in read_valid_jsonl(path)
    ]
    counts = Counter(key for row in rows if (key := job_key(row)) is not None)
    success = {key for row in rows if row.get("status") == "ok" if (key := job_key(row)) is not None}
    errors = {key for row in rows if row.get("status") == "error" if (key := job_key(row)) is not None}
    missing = expected - success
    return {
        "expected": len(expected),
        "raw_rows": len(rows),
        "unique_success": len(expected & success),
        "unique_failed_without_success": len(expected & (errors - success)),
        "duplicate_extra_rows": sum(max(0, count - 1) for count in counts.values()),
        "unexpected_jobs": len(set(counts) - expected),
        "missing_count": len(missing),
        "missing": [list(key) for key in sorted(missing)],
        "complete": not missing,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--search-seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--methods", nargs="+", default=["llm"])
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    report = check_collection(args.data, args.output, args.search_seeds, args.methods)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 1 if args.require_complete and not report["complete"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
