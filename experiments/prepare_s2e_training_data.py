#!/usr/bin/env python3
"""Create gold SFT, verifier-derived DPO and GRPO prompt data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from qihc.problems.cvrp.instance import load_jsonl
from qihc.s2e.cpp import ConstraintProgramPackage


def dump(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--data", type=Path, required=True); p.add_argument("--records", type=Path, required=True); p.add_argument("--output", type=Path, required=True); args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    instances = {x.name: x for x in load_jsonl(args.data)}
    predicted = {x["instance"]: x for x in [json.loads(line) for line in args.records.read_text(encoding="utf-8").splitlines() if line]}
    sft, dpo, grpo = [], [], []
    for name, instance in instances.items():
        gold = ConstraintProgramPackage.from_specs(name, instance.description, instance.customer_ids, instance.constraints, {"source": "gold"}).canonical_json()
        prompt = instance.description
        sft.append({"prompt": prompt, "completion": gold})
        grpo.append({"prompt": prompt, "gold_completion": gold})
        row = predicted.get(name)
        if row and row.get("status") == "ok":
            rejected = json.dumps(row["cpp"], sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            if rejected != gold: dpo.append({"prompt": prompt, "chosen": gold, "rejected": rejected})
    dump(args.output / "sft.jsonl", sft); dump(args.output / "dpo.jsonl", dpo); dump(args.output / "grpo.jsonl", grpo)
    (args.output / "summary.json").write_text(json.dumps({"sft": len(sft), "dpo": len(dpo), "grpo": len(grpo)}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__": raise SystemExit(main())
