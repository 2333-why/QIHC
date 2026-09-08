#!/usr/bin/env python3
"""Formal single-node/multi-GPU experiment runner for QIHC-LNS.

Launch with ``torchrun --standalone --nproc_per_node=4`` on the offline H100 node.
Each rank owns one GPU and a disjoint subset of (instance, seed, method) jobs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import socket
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qihc.problems.cvrp import (  # noqa: E402
    KNNNeighborhoodSelector,
    LNSConfig,
    QIHCLNSSolver,
    RandomNeighborhoodSelector,
    generate_synthetic_instance,
    greedy_initial_solution,
    load_cvrplib,
    verify_solution,
)
from qihc.problems.cvrp.instance import CVRPInstance, RouteSolution, load_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["synthetic", "cvrplib", "jsonl"], default="synthetic")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--bks-json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sizes", type=int, nargs="+", default=[50, 100])
    parser.add_argument("--instance-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--search-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--vehicle-count", type=int, default=10)
    parser.add_argument("--vehicle-capacity", type=int, default=60)
    parser.add_argument("--methods", nargs="+", default=["greedy", "random", "knn"])
    parser.add_argument("--sampler", choices=["numpy", "torch"], default="torch")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--destroy-size", type=int, default=12)
    parser.add_argument("--routes-per-customer", type=int, default=4)
    parser.add_argument("--sampling-steps", type=int, default=1000)
    parser.add_argument("--num-chains", type=int, default=2048)
    parser.add_argument("--top-samples", type=int, default=32)
    parser.add_argument("--update-fraction", type=float, default=0.25)
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--llm-refresh-interval", type=int, default=5)
    parser.add_argument("--llm-temperature", type=float, default=0.2)
    parser.add_argument("--proposal-bias", type=float, default=1.0)
    parser.add_argument("--logit-feedback-rate", type=float, default=0.8)
    parser.add_argument("--disable-logit-feedback", action="store_true")
    parser.add_argument("--feedback-elite-fraction", type=float, default=0.25)
    parser.add_argument("--feedback-negative-weight", type=float, default=0.5)
    parser.add_argument("--disable-safe-expansion", action="store_true")
    parser.add_argument("--safe-heuristic-routes", type=int, default=1)
    parser.add_argument("--safe-random-routes", type=int, default=1)
    parser.add_argument("--baseline-time-limit", type=int, default=30)
    parser.add_argument("--hgs-binary", type=Path)
    return parser.parse_args()


def distributed_context() -> tuple[int, int, int, object | None]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    dist = None
    if world_size > 1:
        import torch.distributed as torch_dist

        if not torch_dist.is_initialized():
            torch_dist.init_process_group("gloo")
        dist = torch_dist
    return rank, world_size, local_rank, dist


def load_instances(args: argparse.Namespace) -> list[CVRPInstance]:
    if args.dataset == "synthetic":
        instances = []
        for size in args.sizes:
            for seed in args.instance_seeds:
                vehicles = max(args.vehicle_count, int(np.ceil(size / 8)))
                instances.append(
                    generate_synthetic_instance(
                        n_customers=size,
                        vehicle_count=vehicles,
                        vehicle_capacity=args.vehicle_capacity,
                        seed=seed,
                        semantic_constraints=True,
                    )
                )
        return instances
    if args.data is None:
        raise ValueError("--data is required for cvrplib/jsonl")
    if args.dataset == "jsonl":
        instances = load_jsonl(args.data)
    else:
        bks = {}
        if args.bks_json:
            bks = json.loads(args.bks_json.read_text(encoding="utf-8"))
        paths = sorted(args.data.rglob("*.vrp")) if args.data.is_dir() else [args.data]
        instances = [load_cvrplib(path, bks.get(path.stem)) for path in paths]
    return instances[: args.limit or None]


def make_selector(args: argparse.Namespace, method: str, local_rank: int, output: Path):
    if method == "random":
        return RandomNeighborhoodSelector(args.destroy_size, args.routes_per_customer)
    if method == "knn":
        return KNNNeighborhoodSelector(args.destroy_size, args.routes_per_customer)
    if method == "llm":
        if not args.model_path:
            raise ValueError("--model-path is required for method=llm")
        from qihc.problems.cvrp.llm_selector import LocalLLMNeighborhoodSelector

        return LocalLLMNeighborhoodSelector(
            model_path=args.model_path,
            destroy_size=args.destroy_size,
            routes_per_customer=args.routes_per_customer,
            device=f"cuda:{local_rank}" if args.sampler == "torch" else "cpu",
            temperature=args.llm_temperature,
            refresh_interval=args.llm_refresh_interval,
            audit_path=output / f"llm_audit_rank{local_rank}.jsonl",
        )
    raise ValueError(f"Unknown method: {method}")


def environment_manifest(args: argparse.Namespace, rank: int, world_size: int, local_rank: int) -> dict:
    manifest = {
        "argv": sys.argv,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": vars(args),
    }
    manifest["config"] = {
        key: str(value) if isinstance(value, Path) else value for key, value in manifest["config"].items()
    }
    try:
        import torch

        manifest["torch"] = torch.__version__
        manifest["cuda_available"] = torch.cuda.is_available()
        manifest["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available() and local_rank < torch.cuda.device_count():
            manifest["gpu"] = torch.cuda.get_device_name(local_rank)
            manifest["gpu_memory_bytes"] = torch.cuda.get_device_properties(local_rank).total_memory
    except ImportError:
        manifest["torch"] = None
    return manifest


def run_job(
    instance: CVRPInstance,
    method: str,
    search_seed: int,
    args: argparse.Namespace,
    selector,
    local_rank: int,
) -> dict:
    if instance.metadata.get("known_feasible_routes"):
        initial = RouteSolution(
            [list(route) for route in instance.metadata["known_feasible_routes"]],
            source="dataset_known_feasible",
        )
    else:
        initial = greedy_initial_solution(instance)
    initial_result = verify_solution(instance, initial)
    if not initial_result.feasible:
        raise ValueError(f"No feasible incumbent: {initial_result.violations}")
    if method == "greedy":
        summary = {
            "instance": instance.name,
            "method": method,
            "search_seed": search_seed,
            "n_customers": len(instance.customers),
            "initial_objective": initial_result.objective,
            "final_objective": initial_result.objective,
            "improvement_fraction": 0.0,
            "feasible": initial_result.feasible,
            "accepted_moves": 0,
            "iterations": 0,
            "total_elapsed_s": 0.0,
            "pbit_elapsed_s": 0.0,
            "mean_qubo_variables": 0.0,
            "mean_candidate_route_recall": 0.0,
            "llm_fallback_rate": 0.0,
            "best_known_cost": instance.best_known_cost,
            "optimality_gap": (
                (initial_result.cost - instance.best_known_cost) / instance.best_known_cost
                if instance.best_known_cost
                else None
            ),
            "records": [],
        }
        return summary
    if method in {"ortools", "hgs"}:
        from qihc.problems.cvrp.baselines import solve_hgs, solve_ortools

        if method == "ortools":
            baseline = solve_ortools(instance, args.baseline_time_limit, search_seed)
        else:
            if args.hgs_binary is None:
                raise ValueError("--hgs-binary is required for method=hgs")
            baseline = solve_hgs(instance, args.hgs_binary, args.baseline_time_limit, search_seed)
        final = baseline.verification
        return {
            "instance": instance.name,
            "method": method,
            "search_seed": search_seed,
            "n_customers": len(instance.customers),
            "initial_objective": initial_result.objective,
            "final_objective": final.objective,
            "improvement_fraction": (
                initial_result.objective - final.objective
            ) / max(abs(initial_result.objective), 1e-12),
            "feasible": final.feasible,
            "accepted_moves": 0,
            "iterations": 0,
            "total_elapsed_s": baseline.elapsed_s,
            "pbit_elapsed_s": 0.0,
            "mean_qubo_variables": 0.0,
            "mean_candidate_route_recall": 0.0,
            "llm_fallback_rate": 0.0,
            "best_known_cost": instance.best_known_cost,
            "optimality_gap": (
                (final.cost - instance.best_known_cost) / instance.best_known_cost
                if instance.best_known_cost
                else None
            ),
            "records": [],
            "solution": baseline.solution.to_dict(),
            "baseline_metadata": baseline.metadata,
        }
    config = LNSConfig(
        iterations=args.iterations,
        destroy_size=args.destroy_size,
        routes_per_customer=args.routes_per_customer,
        sampler=args.sampler,
        sampling_steps=args.sampling_steps,
        num_chains=args.num_chains,
        top_samples=args.top_samples,
        update_fraction=args.update_fraction,
        device=f"cuda:{local_rank}" if args.sampler == "torch" else "cpu",
        seed=search_seed,
        patience=args.patience,
        proposal_bias=args.proposal_bias,
        logit_feedback_rate=args.logit_feedback_rate,
        enable_logit_feedback=not args.disable_logit_feedback,
        feedback_elite_fraction=args.feedback_elite_fraction,
        feedback_negative_weight=args.feedback_negative_weight,
        safe_candidate_expansion=not args.disable_safe_expansion,
        safe_heuristic_routes=args.safe_heuristic_routes,
        safe_random_routes=args.safe_random_routes,
    )
    result = QIHCLNSSolver(config, selector=selector).solve(instance, initial=initial)
    summary = result.to_summary()
    summary.update(
        {
            "method": method,
            "search_seed": search_seed,
            "n_customers": len(instance.customers),
            "best_known_cost": instance.best_known_cost,
            "optimality_gap": (
                (result.verification.cost - instance.best_known_cost) / instance.best_known_cost
                if instance.best_known_cost
                else None
            ),
            "records": [asdict(record) for record in result.records],
            "solution": result.solution.to_dict(),
            "problem_prompt": instance.description,
            "constraints": [
                {"type": c.type, "hard": c.hard, "weight": c.weight, "params": c.params}
                for c in instance.constraints
            ],
        }
    )
    return summary


def aggregate(output: Path, world_size: int) -> None:
    rows = []
    failures = []
    for rank in range(world_size):
        path = output / f"results_rank{rank}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            (failures if record.get("status") == "error" else rows).append(record)
    compact = [{k: v for k, v in row.items() if k not in {"records", "solution", "traceback"}} for row in rows]
    (output / "results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "failures.json").write_text(json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8")
    if compact:
        keys = sorted({key for row in compact for key in row})
        with (output / "results.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(compact)
    groups: dict[str, list[dict]] = {}
    for row in rows:
        key = f"{row['method']}|n={row['n_customers']}"
        groups.setdefault(key, []).append(row)
    summary = {}
    for key, values in groups.items():
        improvements = np.asarray([value["improvement_fraction"] for value in values], dtype=float)
        times = np.asarray([value["total_elapsed_s"] for value in values], dtype=float)
        summary[key] = {
            "n": len(values),
            "feasible_rate": float(np.mean([value["feasible"] for value in values])),
            "mean_improvement_fraction": float(improvements.mean()),
            "std_improvement_fraction": float(improvements.std(ddof=1)) if len(values) > 1 else 0.0,
            "mean_total_elapsed_s": float(times.mean()),
            "mean_pbit_elapsed_s": float(np.mean([value["pbit_elapsed_s"] for value in values])),
            "mean_candidate_route_recall": float(
                np.mean([value.get("mean_candidate_route_recall", 0.0) for value in values])
            ),
            "mean_candidate_compression": float(
                np.mean([value.get("mean_candidate_compression", 0.0) for value in values])
            ),
            "mean_llm_fallback_rate": float(
                np.mean([value.get("llm_fallback_rate", 0.0) for value in values])
            ),
        }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rank, world_size, local_rank, dist = distributed_context()
    manifest = environment_manifest(args, rank, world_size, local_rank)
    (args.output / f"manifest_rank{rank}.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    instances = load_instances(args)
    selectors = {
        method: make_selector(args, method, local_rank, args.output)
        for method in args.methods
        if method not in {"greedy", "ortools", "hgs"}
    }
    jobs = [
        (instance, method, seed)
        for instance in instances
        for seed in args.search_seeds
        for method in args.methods
    ]
    shard = [job for index, job in enumerate(jobs) if index % world_size == rank]
    result_path = args.output / f"results_rank{rank}.jsonl"
    with result_path.open("w", encoding="utf-8") as handle:
        for instance, method, seed in shard:
            identity = {"instance": instance.name, "method": method, "search_seed": seed}
            try:
                record = run_job(
                    instance,
                    method,
                    seed,
                    args,
                    selectors.get(method),
                    local_rank,
                )
                record["status"] = "ok"
            except Exception as exc:
                record = {
                    **identity,
                    "status": "error",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(json.dumps({**identity, "status": record["status"]}, ensure_ascii=False), flush=True)
    if dist is not None:
        dist.barrier()
    if rank == 0:
        aggregate(args.output, world_size)
    if dist is not None:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
