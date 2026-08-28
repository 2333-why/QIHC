#!/usr/bin/env python3
"""Merge disjoint formal-result directories without modifying raw shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = []
    seen = set()
    for source in args.inputs:
        for name, destination in [("results.json", rows), ("failures.json", failures)]:
            path = source / name
            if not path.exists():
                continue
            for row in json.loads(path.read_text(encoding="utf-8")):
                key = (row.get("instance"), row.get("search_seed"), row.get("method"))
                if name == "results.json" and key in seen:
                    raise ValueError(f"Duplicate result key {key}")
                if name == "results.json":
                    seen.add(key)
                destination.append(row)
    (args.output / "results.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output / "failures.json").write_text(
        json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output / "sources.json").write_text(
        json.dumps([str(path) for path in args.inputs], indent=2), encoding="utf-8"
    )
    print(f"merged {len(rows)} results and {len(failures)} failures into {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
