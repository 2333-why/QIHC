#!/usr/bin/env python3
"""Create deterministic, disjoint train/validation/test NL-CVRP splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qihc.problems.cvrp.instance import CVRPInstance, load_jsonl, save_jsonl


def deterministic_split(
    instances: list[CVRPInstance],
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> dict[str, list[CVRPInstance]]:
    if not 0 < train_fraction < 1 or not 0 <= validation_fraction < 1:
        raise ValueError("split fractions are outside [0, 1]")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be below 1")
    names = [instance.name for instance in instances]
    if len(names) != len(set(names)):
        raise ValueError("instance names must be unique before splitting")
    ordered = sorted(
        instances,
        key=lambda instance: hashlib.sha256(
            f"{seed}:{instance.name}".encode("utf-8")
        ).hexdigest(),
    )
    count = len(ordered)
    train_count = round(count * train_fraction)
    validation_count = round(count * validation_fraction)
    return {
        "train": ordered[:train_count],
        "validation": ordered[train_count : train_count + validation_count],
        "test": ordered[train_count + validation_count :],
    }


def split_stats(instances: list[CVRPInstance]) -> dict:
    sizes = [len(instance.customers) for instance in instances]
    return {
        "n": len(instances),
        "min_customers": min(sizes) if sizes else None,
        "max_customers": max(sizes) if sizes else None,
        "mean_customers": sum(sizes) / len(sizes) if sizes else None,
        "instances": [instance.name for instance in instances],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    splits = deterministic_split(
        load_jsonl(args.data),
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    for name, instances in splits.items():
        save_jsonl(instances, args.output / f"{name}.jsonl")
    manifest = {
        "source": str(args.data),
        "seed": args.seed,
        "train_fraction": args.train_fraction,
        "validation_fraction": args.validation_fraction,
        "splits": {name: split_stats(rows) for name, rows in splits.items()},
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({name: len(rows) for name, rows in splits.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
