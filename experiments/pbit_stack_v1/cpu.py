"""Score frozen anchors first, then new endpoints on the same references."""

from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
import subprocess
import sys

from experiments.dcp_external_v2.cpu import score_one
from experiments.feedback_active_v3.runtime import seed_for
from experiments.pbit_dual_v1 import common as prior
from experiments.pbit_joint_v1 import search
from qihc.experiment_runtime.storage import load, save, value_digest, worker_lock
from qihc.generic.feedback.verification import equivalent, score
from . import penalty
from .common import HISTORY, RUN, SEARCH, TRADITIONAL, arms, path, seed_list, state


def native(spec_file, destination, backend, random_seed, traced=False):
    if not destination.exists():
        args = [
            sys.executable,
            "-m",
            "experiments.pbit_stack_v1.cpu",
            str(spec_file),
            str(destination),
            backend,
            str(random_seed),
            str(int(traced)),
        ]
        try:
            subprocess.run(args, check=True, timeout=60)
        except subprocess.TimeoutExpired:
            save(destination, dict(x=None, status="process_time_limit", seconds=60))
    return load(destination)


def score_case(task):
    case, stage = task
    archived = load(path("archived", case, "scores"))
    matched = load(path("archived", case, "matched"))
    original = load(prior.original_artifact(case))
    dcp = case["dataset"] == "dcp"
    ref = load(Path(original["reference"])) if dcp else original["reference"]
    cid = case["id"]
    memo = {}
    semantics = {}

    for model, method in arms(case, stage):
        prediction = path("predictions", case, model)
        record = load(prediction)
        spec = record["model"]
        fingerprint = value_digest(spec)
        arm = f"{model}__{method}"
        for seed in seed_list(case, method):
            out = path("scores", case, arm, seed)
            if out.exists():
                memo[fingerprint, method, seed] = load(out)
                continue
            candidates = [f"{arm}__{seed}"]
            # Use a prior endpoint only with the same prediction and terminal.
            if spec == load(path("predictions", case, "incumbent"))["model"]:
                candidates.append(f"incumbent__{method}__{seed}")
            if not dcp and spec == load(path("predictions", case, "sft"))["model"]:
                candidates.append(f"sft__{method}__{seed}")
            reused = next(
                (archived[key] for key in candidates if key in archived),
                matched.get(f"{fingerprint}__{method}__{seed}"),
            )
            if stage == "improved":
                for old in (
                    "incumbent",
                    "incumbent_B",
                    "incumbent_C",
                    "incumbent_pbit",
                ):
                    old_score = path("scores", case, f"{old}__{method}", seed)
                    if (
                        old_score.exists()
                        and load(path("predictions", case, old))["model"] == spec
                    ):
                        reused = load(old_score)
                        break
            epi = None
            archived_episode = False
            if method == "learned_penalty":
                epi = load(path("episodes", case, arm, seed))
                if epi.get("reuse_schedule"):
                    reused = load(path("scores", case, f"{model}__schedule", seed))
            key = (fingerprint, method, seed)
            if reused is not None:
                row = dict(reused, reused="previous_score")
            elif key in memo:
                row = dict(memo[key], reused="identical_prediction")
            else:
                if spec is None:
                    epi = dict(x=None, status="input_invalid")
                elif method == "scip":
                    solver_seed = seed if dcp else seed_for(cid, "terminal")
                    cache_key = value_digest([cid, spec, "scip", 20, 32, solver_seed])
                    cache = HISTORY / "terminal_cache" / (cache_key + ".json")
                    if not dcp and cache.exists():
                        epi = load(cache)
                        archived_episode = True
                    else:
                        epi = native(
                            prediction,
                            path("episodes", case, arm, seed),
                            "scip",
                            solver_seed,
                            traced=not dcp,
                        )
                elif method == "schedule":
                    epi = search.episode(
                        spec, cid, dict(SEARCH, seed=seed), lambda *_: "cool"
                    )
                elif method == "rule_penalty":
                    epi = penalty.episode(spec, cid, seed, penalty.rule)
                row = evaluate(dcp, original, ref, epi)
                if archived_episode:
                    row["reused"] = "E005-1.7_terminal"
            if not dcp and "semantic" not in row:
                if fingerprint not in semantics:
                    semantics[fingerprint] = equivalent(spec, original["gold"])[
                        "status"
                    ]
                row["semantic"] = semantics[fingerprint]
            row.update(
                valid=spec is not None,
                feedback=record.get("action"),
                feedback_calls=record.get("feedback_calls", 0),
                probe_attempts=record.get("probe", {}).get("attempts", 0)
                + sum(
                    e.get("probe", {}).get("attempts", 0)
                    for e in record.get("evidence", [])
                ),
                probe_seconds=record.get("probe", {}).get("seconds", 0)
                + sum(
                    e.get("probe", {}).get("seconds", 0)
                    for e in record.get("evidence", [])
                ),
            )
            memo[key] = row
            save(out, row)

    if not dcp and stage == "current":
        gold_file = path("solver_inputs", case, "gold20")
        if not gold_file.exists():
            save(gold_file, dict(model=original["gold"]))
        for backend in TRADITIONAL:
            arm = "gold20__" + backend
            target = path("scores", case, arm, 17)
            if target.exists():
                continue
            if arm + "__17" in archived:
                row = dict(archived[arm + "__17"], reused="E003-1.1")
            else:
                epi = native(gold_file, path("episodes", case, arm, 0), backend, 0)
                row = dict(
                    evaluate(False, original, ref, epi), valid=None, original_seed=0
                )
            save(target, row)


def evaluate(dcp, original, ref, episode):
    if dcp:
        row = score_one(ref, episode)
    else:
        s = score(original["gold"], episode, ref)
        row = dict(
            hit=s["optimal_hit"],
            feasible=s["original_feasible"],
            gap=s["gap"],
            reference_proven=s["reference_proven"],
            status=episode.get("status", "evaluated"),
            actions=episode.get("actions", []),
            seconds=episode.get(
                "episode_seconds",
                episode.get("wall_seconds", episode.get("seconds", 0)),
            ),
            decision_seconds=episode.get("decision_seconds", 0),
            attempts=episode.get("heat_bath_attempts", episode.get("attempts", 0)),
        )
    row["penalty_writes"] = episode.get("penalty_writes", 0)
    return row


def finish(stage):
    from .report import report

    state(verify=True)
    cohort = load(RUN / "cases.json")
    if any(not (RUN / "gpu" / f"{stage}_{i}.json").exists() for i in range(4)):
        raise ValueError(f"Complete gpu-{stage} first")
    with (
        worker_lock(RUN / "cpu.lock"),
        ProcessPoolExecutor(
            max_workers=int(os.getenv("QIHC_CPU_WORKERS", "6"))
        ) as pool,
    ):
        for i, _ in enumerate(pool.map(score_case, [(c, stage) for c in cohort]), 1):
            if i % 50 == 0 or i == len(cohort):
                print(stage, "scored", i, "/", len(cohort), flush=True)
        report(stage, require_complete=True)


if __name__ == "__main__":
    from qihc.generic.solver_baselines.registry import solve

    source, destination, backend, random_seed, traced = sys.argv[1:]
    save(
        Path(destination),
        solve(
            load(Path(source))["model"],
            backend,
            20,
            int(random_seed),
            chains=32 if traced == "1" else 16,
            collect_trace=traced == "1",
        ),
    )
