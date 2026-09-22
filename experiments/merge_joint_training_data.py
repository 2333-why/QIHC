#!/usr/bin/env python3
"""Merge and de-duplicate joint SFT/DPO/GRPO datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


FILES = ("sft.jsonl", "dpo.jsonl", "grpo.jsonl")


def stable_key(row: dict) -> str:
    payload = json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {"sources": [str(path) for path in args.inputs]}
    for filename in FILES:
        selected: dict[str, dict] = {}
        raw = 0
        for directory in args.inputs:
            source = directory / filename
            if not source.is_file():
                raise FileNotFoundError(source)
            for line in source.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Expected object in {source}")
                raw += 1
                selected.setdefault(stable_key(value), value)
        target = args.output / filename
        with target.open("w", encoding="utf-8") as handle:
            for value in selected.values():
                handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        summary[filename.removesuffix(".jsonl")] = {
            "raw": raw, "unique": len(selected), "duplicates_removed": raw - len(selected)
        }
    if summary["sft"]["unique"] == 0 or summary["grpo"]["unique"] == 0:
        raise SystemExit("SFT and GRPO datasets must be non-empty")
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
