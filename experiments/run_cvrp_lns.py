#!/usr/bin/env python3
"""Formal single-node/multi-GPU experiment runner for QIHC-LNS.

Launch with ``torchrun --standalone --nproc_per_node=8`` on the eight-A100 node.
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
import copy
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
    parser.add_argument("--constraint-source", choices=["dataset", "cpp", "llm"], default="dataset",
                        help="Compile natural-language constraints before search; dataset is the legacy oracle mode.")
    parser.add_argument("--cpp-records", type=Path, help="records.jsonl produced by run_s2e_pipeline.py")
    parser.add_argument("--initialization", choices=["incumbent", "pbit-cold"], default="incumbent")
    parser.add_argument("--cold-batch-size", type=int, default=6)
    parser.add_argument("--cold-routes-per-customer", type=int, default=4)
    parser.add_argument("--token-feedback-strength", type=float, default=1.0)
    parser.add_argument("--adapter-path", type=str)
    parser.add_argument("--llm-refresh-interval", type=int, default=5)
    parser.add_argument("--llm-temperature", type=float, default=0.2)
    parser.add_argument("--llm-max-new-tokens", type=int, default=768)
    parser.add_argument("--llm-direct-max-new-tokens", type=int, default=8192)
    parser.add_argument("--llm-max-input-tokens", type=int, default=32768)
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep valid existing rank JSONL rows and skip completed jobs.",
    )
    return parser.parse_args()


def read_valid_jsonl(path: Path) -> list[dict]:
    """Read complete JSON objects, ignoring a partially written final line."""

    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def job_key(record: dict) -> tuple[str, str, int] | None:
    try:
        return (
            str(record["instance"]),
            str(record["method"]),
            int(record["search_seed"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def deduplicate_job_records(records: list[dict]) -> list[dict]:
    """Keep one record per experiment job, preferring a successful record."""

    selected: dict[tuple[str, str, int], dict] = {}
    for record in records:
        key = job_key(record)
        if key is None:
            continue
        previous = selected.get(key)
        if previous is None or (
            previous.get("status") != "ok" and record.get("status") == "ok"
        ):
            selected[key] = record
    return list(selected.values())


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
    if method == "llm_direct":
        if not args.model_path:
            raise ValueError("--model-path is required for method=llm_direct")
        from qihc.problems.cvrp.direct_llm import LocalDirectLLMGenerator

        return LocalDirectLLMGenerator(
            model_path=args.model_path,
            device=f"cuda:{local_rank}" if args.sampler == "torch" else "cpu",
            temperature=args.llm_temperature,
            max_new_tokens=args.llm_direct_max_new_tokens,
            max_input_tokens=args.llm_max_input_tokens,
        )
    if method == "llm":
        if not args.model_path:
            raise ValueError("--model-path is required for method=llm")
        from qihc.problems.cvrp.llm_selector import LocalLLMNeighborhoodSelector

        return LocalLLMNeighborhoodSelector(
            model_path=args.model_path,
            adapter_path=args.adapter_path,
            destroy_size=args.destroy_size,
            routes_per_customer=args.routes_per_customer,
            device=f"cuda:{local_rank}" if args.sampler == "torch" else "cpu",
            temperature=args.llm_temperature,
            max_new_tokens=args.llm_max_new_tokens,
            refresh_interval=args.llm_refresh_interval,
            audit_path=output / f"llm_audit_rank{local_rank}.jsonl",
            token_feedback_strength=args.token_feedback_strength,
            max_input_tokens=args.llm_max_input_tokens,
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


def prepare_compiled_instance(instance: CVRPInstance, args: argparse.Namespace, selector, cpp_records: dict) -> CVRPInstance:
    """Keep the reference constraints separate from the generated solver input."""
    if args.constraint_source == "dataset":
        return instance
    from qihc.s2e import CPPValidator, ConstraintSynthesizer, compile_cpp
    from qihc.s2e.cpp import ConstraintProgramPackage

    if args.constraint_source == "cpp":
        record = cpp_records.get(instance.name)
        if not record or record.get("status") != "ok":
            raise ValueError(f"No successful CPP record for {instance.name}")
        cpp = ConstraintProgramPackage.from_dict(record["cpp"])
    else:
        if selector is None or not hasattr(selector, "parse_constraint_ir"):
            raise ValueError("--constraint-source llm requires method=llm and --model-path")
        class SelectorBackend:
            def generate(self, description, customer_ids):
                return selector.parse_constraint_ir(description, customer_ids)
        cpp, _ = ConstraintSynthesizer(SelectorBackend()).synthesize_verified(instance, CPPValidator())
    if cpp.instance_name != instance.name or set(cpp.customer_ids) != set(instance.customer_ids):
        raise ValueError(f"CPP identity/customer mismatch for {instance.name}")
    validation = CPPValidator().validate(cpp, instance)
    plan = compile_cpp(cpp, instance, validation)
    compiled = copy.deepcopy(instance)
    compiled.constraints = cpp.specs()
    compiled.metadata["constraint_compilation"] = plan.to_dict()
    compiled.metadata["constraint_source"] = args.constraint_source
    compiled.metadata["cpp_checksum"] = cpp.checksum
    return compiled


def run_job(
    instance: CVRPInstance,
    method: str,
    search_seed: int,
    args: argparse.Namespace,
    selector,
    local_rank: int,
    reference_instance: CVRPInstance | None = None,
) -> dict:
    reference_instance = reference_instance or instance
    # Make stochastic LLM decoding and p-bit sampling reproducible per job.
    np.random.seed(search_seed)
    try:
        import torch

        torch.manual_seed(search_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(search_seed)
    except ImportError:
        pass
    if method == "llm_direct":
        from qihc.problems.cvrp.baselines import solve_direct_llm

        baseline = solve_direct_llm(instance, selector._generate)
        final = baseline.verification
        return {
            "instance": instance.name, "method": method, "search_seed": search_seed,
            "n_customers": len(instance.customers), "initial_objective": None,
            "final_objective": final.objective, "improvement_fraction": 0.0,
            "feasible": final.feasible, "reference_feasible": final.feasible,
            "reference_violations": final.violations, "accepted_moves": 0, "iterations": 0,
            "total_elapsed_s": baseline.elapsed_s, "pbit_elapsed_s": 0.0,
            "mean_qubo_variables": 0.0, "mean_candidate_route_recall": 0.0,
            "llm_fallback_rate": 0.0, "best_known_cost": instance.best_known_cost,
            "optimality_gap": ((final.cost - instance.best_known_cost) / instance.best_known_cost
                               if instance.best_known_cost and final.feasible else None),
            "records": [], "solution": baseline.solution.to_dict(),
            "baseline_metadata": baseline.metadata,
        }

    cold_start = args.initialization == "pbit-cold" and method not in {"greedy", "ortools", "hgs"}
    initial_started = time.perf_counter()
    if method in {"ortools", "hgs"}:
        initial = None
    elif cold_start:
        initial = None
    elif instance.metadata.get("known_feasible_routes"):
        initial = RouteSolution(
            [list(route) for route in instance.metadata["known_feasible_routes"]],
            source="dataset_known_feasible",
        )
    else:
        initial = greedy_initial_solution(instance)
    initial_result = verify_solution(instance, initial) if initial is not None else None
    initial_elapsed_s = time.perf_counter() - initial_started
    if initial_result is not None and not initial_result.feasible:
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
            "reference_feasible": initial_result.feasible,
            "reference_violations": initial_result.violations,
            "accepted_moves": 0,
            "iterations": 0,
            "total_elapsed_s": initial_elapsed_s,
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
            "solution": initial.to_dict(),
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
            "initial_objective": initial_result.objective if initial_result is not None else None,
            "final_objective": final.objective,
            "improvement_fraction": (
                initial_result.objective - final.objective
            ) / max(abs(initial_result.objective), 1e-12) if initial_result is not None else 0.0,
            "feasible": final.feasible,
            "reference_feasible": final.feasible,
            "reference_violations": final.violations,
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
        initialization=args.initialization,
        cold_batch_size=args.cold_batch_size,
        cold_routes_per_customer=args.cold_routes_per_customer,
    )
    result = QIHCLNSSolver(config, selector=selector).solve(instance, initial=initial)
    summary = result.to_summary()
    compiled_feasible = summary["feasible"]
    reference_check = verify_solution(reference_instance, result.solution)
    summary["compiled_feasible"] = compiled_feasible
    summary["feasible"] = reference_check.feasible
    summary.update(
        {
            "method": method,
            "search_seed": search_seed,
            "n_customers": len(instance.customers),
            "best_known_cost": instance.best_known_cost,
            "optimality_gap": (
                (result.verification.cost - instance.best_known_cost) / instance.best_known_cost
                if instance.best_known_cost and reference_check.feasible
                else None
            ),
            "records": [asdict(record) for record in result.records],
            "solution": result.solution.to_dict(),
            "problem_prompt": instance.description,
            "constraints": [
                {"type": c.type, "hard": c.hard, "weight": c.weight, "params": c.params}
                for c in instance.constraints
            ],
            "constraint_source": instance.metadata.get("constraint_source", "dataset"),
            "constraint_compilation": instance.metadata.get("constraint_compilation"),
            "reference_feasible": reference_check.feasible,
            "reference_violations": reference_check.violations,
            "constraint_exact_match": {
                (c.type, c.hard, json.dumps(c.params, sort_keys=True)) for c in instance.constraints
            } == {
                (c.type, c.hard, json.dumps(c.params, sort_keys=True)) for c in reference_instance.constraints
            } if reference_instance.constraints else None,
        }
    )
    return summary


def aggregate(output: Path, world_size: int) -> None:
    records = []
    for path in sorted(output.glob("results_rank*.jsonl")):
        records.extend(read_valid_jsonl(path))
    records = deduplicate_job_records(records)
    rows = [record for record in records if record.get("status") != "error"]
    failures = [record for record in records if record.get("status") == "error"]
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
        gaps = [float(value["optimality_gap"]) for value in values if value.get("optimality_gap") is not None]
        summary[key] = {
            "n": len(values),
            "feasible_rate": float(np.mean([value["feasible"] for value in values])),
            "mean_improvement_fraction": float(improvements.mean()),
            "std_improvement_fraction": float(improvements.std(ddof=1)) if len(values) > 1 else 0.0,
            "mean_total_elapsed_s": float(times.mean()),
            "mean_optimality_gap": float(np.mean(gaps)) if gaps else None,
            "reference_feasible_rate": float(np.mean([value.get("reference_feasible", value["feasible"]) for value in values])),
            "mean_construction_pbit_s": float(np.mean([value.get("construction_pbit_s", 0.0) for value in values])),
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
    if args.constraint_source == "cpp" and not args.cpp_records:
        raise ValueError("--cpp-records is required for --constraint-source cpp")
    cpp_records = {}
    if args.cpp_records:
        cpp_records = {row["instance"]: row for row in read_valid_jsonl(args.cpp_records)}
    selectors = {
        method: make_selector(args, method, local_rank, args.output)
        for method in args.methods
        if method not in {"greedy", "ortools", "hgs"}
    }
    prepared_instances: dict[tuple[str, str], CVRPInstance] = {}
    jobs = [
        (instance, method, seed)
        for instance in instances
        for seed in args.search_seeds
        for method in args.methods
    ]
    completed: set[tuple[str, str, int]] = set()
    if args.resume:
        for existing_path in args.output.glob("results_rank*.jsonl"):
            for record in read_valid_jsonl(existing_path):
                key = job_key(record)
                if key is not None and record.get("status") == "ok":
                    completed.add(key)
    shard = [job for index, job in enumerate(jobs) if index % world_size == rank]
    if completed:
        shard = [
            job for job in shard
            if (job[0].name, job[1], int(job[2])) not in completed
        ]
    result_path = args.output / f"results_rank{rank}.jsonl"
    if args.resume:
        # Failed rows are intentionally removed so their jobs can be retried
        # without leaving stale failures in the final aggregate.
        existing_rows = deduplicate_job_records(
            [
                record for record in read_valid_jsonl(result_path)
                if record.get("status") == "ok"
            ]
        )
        with result_path.open("w", encoding="utf-8") as cleanup:
            for record in existing_rows:
                cleanup.write(json.dumps(record, ensure_ascii=False) + "\n")
    mode = "a" if args.resume else "w"
    print(
        json.dumps(
            {
                "rank": rank,
                "resume": args.resume,
                "completed_jobs_found": len(completed),
                "remaining_jobs_on_rank": len(shard),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    with result_path.open(mode, encoding="utf-8") as handle:
        for instance, method, seed in shard:
            identity = {"instance": instance.name, "method": method, "search_seed": seed}
            try:
                preparation_key = (instance.name, method)
                if preparation_key not in prepared_instances:
                    prepared_instances[preparation_key] = prepare_compiled_instance(
                        instance, args, selectors.get(method), cpp_records
                    )
                record = run_job(
                    prepared_instances[preparation_key],
                    method,
                    seed,
                    args,
                    selectors.get(method),
                    local_rank,
                    reference_instance=instance,
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
