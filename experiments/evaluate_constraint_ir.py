#!/usr/bin/env python3
"""Multi-GPU evaluation of natural-language constraint parsing into Constraint IR."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qihc.problems.cvrp.instance import ConstraintSpec, load_jsonl  # noqa: E402
from qihc.problems.cvrp.llm_selector import LocalLLMNeighborhoodSelector  # noqa: E402


def canonical(spec: ConstraintSpec) -> str:
    kind = "same_resource" if spec.type == "same_vehicle" else spec.type
    return json.dumps(
        {"type": kind, "hard": spec.hard, "params": spec.params},
        sort_keys=True,
        ensure_ascii=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    dist = None
    if world > 1:
        import torch.distributed as torch_dist

        torch_dist.init_process_group("gloo")
        dist = torch_dist
    args.output.mkdir(parents=True, exist_ok=True)
    instances = load_jsonl(args.data)[: args.limit or None]
    shard = [instance for idx, instance in enumerate(instances) if idx % world == rank]
    parser_model = LocalLLMNeighborhoodSelector(
        args.model_path,
        destroy_size=1,
        routes_per_customer=1,
        device=f"cuda:{local_rank}",
        temperature=args.temperature,
    )
    target = args.output / f"constraint_ir_rank{rank}.jsonl"
    with target.open("w", encoding="utf-8") as handle:
        for instance in shard:
            try:
                predicted, audit = parser_model.parse_constraint_ir(
                    instance.description, instance.customer_ids
                )
                gold_set = {canonical(spec) for spec in instance.constraints}
                predicted_set = {canonical(spec) for spec in predicted}
                gold_types = {spec.type for spec in instance.constraints}
                predicted_types = {spec.type for spec in predicted}
                record = {
                    "instance": instance.name,
                    "status": "ok",
                    "schema_valid": True,
                    "exact_match": gold_set == predicted_set,
                    "gold_count": len(gold_set),
                    "predicted_count": len(predicted_set),
                    "type_tp": len(gold_types & predicted_types),
                    "type_fp": len(predicted_types - gold_types),
                    "type_fn": len(gold_types - predicted_types),
                    "constraint_tp": len(gold_set & predicted_set),
                    "constraint_fp": len(predicted_set - gold_set),
                    "constraint_fn": len(gold_set - predicted_set),
                    "gold": [canonical(spec) for spec in instance.constraints],
                    "predicted": [canonical(spec) for spec in predicted],
                    "audit": audit,
                }
            except Exception as exc:
                record = {
                    "instance": instance.name,
                    "status": "error",
                    "schema_valid": False,
                    "exact_match": False,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(json.dumps({"instance": instance.name, "status": record["status"]}), flush=True)
    if dist:
        dist.barrier()
    if rank == 0:
        records = []
        for shard_rank in range(world):
            shard_path = args.output / f"constraint_ir_rank{shard_rank}.jsonl"
            records.extend(json.loads(line) for line in shard_path.read_text(encoding="utf-8").splitlines() if line)
        ok = [record for record in records if record["status"] == "ok"]
        type_tp = sum(record["type_tp"] for record in ok)
        type_fp = sum(record["type_fp"] for record in ok)
        type_fn = sum(record["type_fn"] for record in ok)
        constraint_tp = sum(record["constraint_tp"] for record in ok)
        constraint_fp = sum(record["constraint_fp"] for record in ok)
        constraint_fn = sum(record["constraint_fn"] for record in ok)

        def scores(tp: int, fp: int, fn: int) -> dict:
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            return {
                "precision": precision,
                "recall": recall,
                "f1": 2 * precision * recall / max(precision + recall, 1e-12),
            }

        summary = {
            "n": len(records),
            "schema_valid_rate": len(ok) / max(len(records), 1),
            "exact_match_rate": sum(record["exact_match"] for record in ok) / max(len(records), 1),
            "type_micro": scores(type_tp, type_fp, type_fn),
            "constraint_micro": scores(constraint_tp, constraint_fp, constraint_fn),
            "errors": len(records) - len(ok),
        }
        (args.output / "constraint_ir_results.json").write_text(
            json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (args.output / "constraint_ir_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    if dist:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
