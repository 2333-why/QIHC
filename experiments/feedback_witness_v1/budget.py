"""Read-only E004-1.4 capacity audit. This does not execute or enforce a run."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def assess(c, selection):
    cases = selection["cases"]
    n = c["case_count"]
    if len(cases) != n or len({x["id"] for x in cases}) != n:
        raise ValueError("Case count and unique frozen selection must match")
    if n != 18 or len({x["dataset"] for x in cases}) != 18:
        raise ValueError("This iteration freezes one case per 18 strata")
    if set(c["arms"]) != {"S", "R", "P", "X", "A", "M"} or len(c["arms"]) != 6:
        raise ValueError("All six arms are required")
    if c["seeds"] != [0] or c["cross_seeds"] != [0] or c["rounds"] != 2:
        raise ValueError("Changing seeds or rounds needs a new budget/iteration")
    if c["max_generation_attempts"] != 1 or c["max_gpu_restarts"] != 0:
        raise ValueError("Retries/restarts are not covered by this bound")
    if c["gpu_batch_size"] != 1 or c["include_36_case_multiseed_validation"]:
        raise ValueError("Batching or additional validation changes this schedule")
    if not c["retain_invalid_initial_models"] or c["allow_outcome_adaptive_case_or_arm_drop"]:
        raise ValueError("Failed cases and fixed arms must remain in the experiment")
    workers, gpus = c["cpu_workers"], c["gpu_replicas"]
    if workers <= 0 or gpus <= 0:
        raise ValueError("Worker counts must be positive")
    trajectories = n * len(c["arms"]) * len(c["seeds"])
    patches = trajectories * c["rounds"]
    contract_waves = math.ceil(n / gpus)
    patch_waves = math.ceil(trajectories / gpus) * c["rounds"]
    gpu_seconds = (contract_waves * (c["contract_timeout_seconds"] + c["gpu_request_cleanup_seconds"])
                   + patch_waves * (c["patch_timeout_seconds"] + c["gpu_request_cleanup_seconds"]))
    cpu_waves = math.ceil(trajectories / workers)
    feedback_cpu = (cpu_waves * c["rounds"]
                    * (sum(c[k] for k in ("base_seconds", "probe_seconds", "final_seconds", "guard_seconds"))
                       + 4 * c["cpu_stage_cleanup_seconds"])
                    + math.ceil(n / workers) * (c["guard_seconds"] + c["cpu_stage_cleanup_seconds"]))
    cross_jobs = trajectories * len(c["cross_backends"]) * len(c["cross_seeds"])
    cross_seconds = math.ceil(cross_jobs / workers) * (c["cross_solver_seconds"] + c["cross_job_overhead_seconds"])
    # Initial model + every round's candidate, plus one canonical reference per case.
    # Accepted/rejected versions may alias these hashes; no cache savings assumed.
    semantic_jobs = n + patches + n
    semantic_seconds = math.ceil(semantic_jobs / workers) * (c["semantic_check_seconds"] + c["semantic_job_overhead_seconds"])
    scoring_seconds = math.ceil(cross_jobs / workers) * c["score_job_seconds"]
    offline = cross_seconds + semantic_seconds + scoring_seconds
    reserves = c["stage_reservations_seconds"]
    total = gpu_seconds + sum(reserves.values())
    slack = c["wall_limit_seconds"] - total
    checks = {
        "feedback_cpu_fits_reservation": feedback_cpu <= reserves["feedback_cpu"],
        "offline_fits_reservation": offline <= reserves["offline_evaluation"],
        "one_hour_spare": slack >= c["minimum_reserve_seconds"],
        "work_and_pack_fit": c["work_deadline_seconds"] + reserves["report_and_pack"] <= c["wall_limit_seconds"],
        "planned_work_fits_work_deadline": total - reserves["report_and_pack"] <= c["work_deadline_seconds"],
    }
    return {
        "experiment_id": c["experiment_id"],
        "status": "capacity_check_passed" if all(checks.values()) else "capacity_check_failed",
        "runner_implemented": (HERE/'runtime.py').exists() and (HERE/'supervise.py').exists(),
        "assumptions": ["four dedicated replicas and 24 single-thread CPU workers",
                        "bounded subprocess/request cleanup and phase barriers",
                        "no retries or additional validation",
                        "historical E003 baselines are read-only; no baseline rerun",
                        "an unfinished or timed-out task is not a successful result",
                        "this estimate is not an enforced deadline or a hardware guarantee"],
        "case_ids": [x["id"] for x in cases],
        "counts": {"trajectories": trajectories, "contract_calls": n, "patch_calls": patches,
                   "llm_calls": n + patches, "cross_solver_jobs_max": cross_jobs,
                   "semantic_jobs_max": semantic_jobs},
        "output_token_cap": n * c["contract_output_tokens"] + patches * c["patch_output_tokens"],
        "gpu_waves": {"contract": contract_waves, "patch": patch_waves},
        "seconds": {"gpu_request_envelope": gpu_seconds, "feedback_cpu_calculated": feedback_cpu,
                    "cross_solver_calculated": cross_seconds, "semantic_calculated": semantic_seconds,
                    "scoring_calculated": scoring_seconds, "offline_calculated": offline,
                    "reserved_stages": reserves, "planned_total": total, "spare": slack,
                    "wall_limit": c["wall_limit_seconds"]},
        "checks": checks,
    }


def historical(cache):
    d = load(cache)
    responses = [v for k, v in d.items() if k.startswith("responses/")]
    groups = {}
    for budget in sorted({r["token_budget"] for r in responses}):
        rows = [r for r in responses if r["token_budget"] == budget]
        ts = sorted(r["attributed_seconds"] for r in rows)
        groups[str(budget)] = {"calls": len(rows), "mean_seconds": statistics.mean(ts),
                               "p95_seconds": ts[int(.95 * (len(ts) - 1))], "max_seconds": max(ts)}
    return {"experiment_id": d["manifest.json"]["experiment_id"],
            "cache_sha256": hashlib.sha256(Path(cache).read_bytes()).hexdigest(),
            "wall_seconds": d["status.json"]["seconds_this_launch"],
            "calls": len(responses), "groups_by_output_cap": groups,
            "note": "Old prompts only; no measured E004-1.4 throughput."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=HERE / "config_8h.json")
    p.add_argument("--history-cache", type=Path)
    args = p.parse_args()
    c = load(args.config)
    selection = ROOT / c["selection_file"]
    report = assess(c, load(selection))
    report["config_sha256"] = hashlib.sha256(args.config.read_bytes()).hexdigest()
    report["selection_sha256"] = hashlib.sha256(selection.read_bytes()).hexdigest()
    if args.history_cache:
        report["historical_timing"] = historical(args.history_cache)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "capacity_check_passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
