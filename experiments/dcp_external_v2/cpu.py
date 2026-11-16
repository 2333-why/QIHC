"""CPU orchestration: isolated solver processes and scoring on original CP constraints."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
import subprocess
import sys
from qihc.experiment_runtime.storage import load, save, worker_lock, value_digest
from .common import (
    RUN,
    GOLD,
    PRED,
    ARMS,
    SEARCH,
    SEED,
    SECONDS,
    cases,
    path,
    rowpath,
    repeats,
    stage_name,
)


def solve_generated(cid):
    from experiments.pbit_joint_v1 import search
    from qihc.generic.solver_baselines.registry import solve

    memo = {}
    for model, method in ARMS:
        if model == "gold" or method in ("rank", "balanced"):
            continue
        for random_seed in repeats(model, method):
            dest = path(stage_name("episodes", random_seed), model + "__" + method, cid)
            spec = load(path("predictions", model, cid))["model"]
            key = (method, random_seed, value_digest(spec))
            if dest.exists():
                memo[key] = load(dest)
                continue
            if key not in memo:
                if spec is None:
                    result = dict(x=None, status="input_invalid")
                elif method == "scip":
                    result = solve(spec, method, SECONDS, SEED)
                else:
                    choose = (
                        (lambda s, st, _: search.rule(s, st))
                        if method == "rule"
                        else (lambda *_: "cool")
                    )
                    result = search.episode(
                        spec, cid, dict(SEARCH, seed=random_seed), choose
                    )
                memo[key] = result
            save(dest, memo[key])


def invoke(args, timeout):
    cmd = (
        [sys.executable]
        + (["-S"] if sys.flags.no_site else [])
        + ["-m", __spec__.name, *args]
    )
    p = subprocess.run(
        cmd, capture_output=True, text=True, errors="replace", timeout=timeout
    )
    if p.returncode:
        raise RuntimeError(
            f"{args}: exit {p.returncode}\n{p.stderr[-4000:]}\n{p.stdout[-600:]}"
        )


def gold_child(cid, method, random_seed):
    random_seed = int(random_seed)
    if sys.platform.startswith("linux"):
        import resource

        limit = 8 * 1024**3
        current, maximum = resource.getrlimit(resource.RLIMIT_AS)
        limits = [n for n in (current, maximum, limit) if n != resource.RLIM_INFINITY]
        resource.setrlimit(resource.RLIMIT_AS, (min(limits), maximum))
    from .traditional import run_gold
    from importlib.util import find_spec

    dep = {"cp_sat": "ortools", "scip": "pyscipopt", "highs": "highspy"}.get(method)
    if dep and find_spec(dep) is None:
        raise ImportError(f"Missing {dep}")
    ref = load(RUN / "reference" / (cid + ".json"))
    try:
        result = run_gold(ref, method, seed=random_seed)
    except MemoryError:
        result = dict(x=None, status="solver_memory_limit")
    except Exception as exc:
        from cpmpy.exceptions import (
            NotSupportedError,
            TransformationNotImplementedError,
        )

        if not isinstance(
            exc,
            (NotSupportedError, TransformationNotImplementedError, NotImplementedError),
        ):
            raise
        result = dict(x=None, status="unsupported", detail=str(exc))
    save(path(stage_name("episodes", random_seed), "gold__" + method, cid), result)


def score_one(ref, ep):
    from .reference import check_vector

    s = check_vector(ref, ep.get("x"))
    return dict(
        s,
        status=ep.get("status", "evaluated"),
        seconds=ep.get("episode_seconds", ep.get("wall_seconds", ep.get("seconds", 0))),
        actions=ep.get("actions", []),
        attempts=ep.get("heat_bath_attempts", ep.get("attempts")),
        intervention_writes=ep.get("intervention_writes"),
        decision_seconds=ep.get("decision_seconds"),
    )


def gold_task(task):
    cid, method, random_seed = task
    target = rowpath("gold", method, cid, random_seed)
    if target.exists():
        return
    ep_path = path(stage_name("episodes", random_seed), "gold__" + method, cid)
    if not ep_path.exists():
        try:
            invoke(["gold-child", cid, method, str(random_seed)], SECONDS + 5)
        except subprocess.TimeoutExpired:
            save(
                ep_path, dict(x=None, status="process_time_limit", seconds=SECONDS + 5)
            )
    ep = load(ep_path)
    ref = load(RUN / "reference" / (cid + ".json"))
    s = score_one(ref, ep)
    save(target, dict(s, valid=None))


def finish_case(case):
    cid = case["id"]
    if all(
        rowpath(m, a, cid, s).exists()
        for m, a in ARMS
        if m != "gold"
        for s in repeats(m, a)
    ):
        return
    invoke(["generated-child", cid], 1800)
    ref = load(RUN / "reference" / (cid + ".json"))
    checks = {}
    for model, method in ARMS:
        for random_seed in repeats(model, method):
            target = rowpath(model, method, cid, random_seed)
            if model == "gold" or target.exists():
                continue
            pred = load(path("predictions", model, cid))
            ep = load(
                path(stage_name("episodes", random_seed), model + "__" + method, cid)
            )
            key = tuple(ep["x"]) if ep.get("x") is not None else None
            if key not in checks:
                checks[key] = score_one(ref, ep)
            s = dict(checks[key])
            # Keep per-method costs and actions even if the assignment is identical.
            for k in (
                "status",
                "seconds",
                "actions",
                "attempts",
                "intervention_writes",
                "decision_seconds",
            ):
                s.pop(k, None)
            calls = pred.get("calls", [])
            probe = sum(
                e.get("probe", {}).get("seconds", 0) for e in pred.get("evidence", [])
            )
            probe += pred.get("probe", {}).get("seconds", 0)
            save(
                target,
                dict(
                    s,
                    valid=pred["model"] is not None,
                    status=ep.get("status", "evaluated"),
                    seconds=ep.get("episode_seconds", ep.get("wall_seconds", 0)),
                    actions=ep.get("actions", []),
                    attempts=ep.get("heat_bath_attempts"),
                    intervention_writes=ep.get("intervention_writes"),
                    decision_seconds=ep.get("decision_seconds"),
                    feedback_action=pred.get("action"),
                    calls=len(calls),
                    generation_seconds=sum(c["seconds"] for c in calls),
                    probe_seconds=probe,
                    prompt_tokens=sum(
                        c["usage"].get("prompt_tokens", 0) for c in calls
                    ),
                    completion_tokens=sum(
                        c["usage"].get("completion_tokens", 0) for c in calls
                    ),
                ),
            )


def parallel(tasks, func, stage):
    tasks = list(tasks)
    errors = []

    def run(task):
        try:
            func(task)
        except Exception as exc:
            return dict(
                task=task["id"] if isinstance(task, dict) else task, error=str(exc)
            )

    with ThreadPoolExecutor(
        max_workers=int(os.getenv("QIHC_CPU_WORKERS", "6"))
    ) as pool:
        for i, err in enumerate(pool.map(run, tasks), 1):
            if err:
                errors.append(err)
                print(err, flush=True)
            if i % 50 == 0:
                print(stage, i, "/", len(tasks), flush=True)
    save(RUN / (stage + "_errors.json"), errors)
    if errors:
        raise RuntimeError(
            f"{len(errors)} unfinished tasks. Inspect {stage}_errors.json and rerun {stage}."
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=("prepare", "finish", "gold-child", "generated-child")
    )
    parser.add_argument("parts", nargs="*")
    a = parser.parse_args()
    if a.stage == "gold-child":
        gold_child(*a.parts)
    elif a.stage == "generated-child":
        solve_generated(*a.parts)
    else:
        from .identity import bind, verify

        with worker_lock(RUN / (a.stage + ".lock")):
            if a.stage == "prepare":
                from .reference import prepare

                prepare()
                bind()
                tasks = (
                    (r["id"], b, s)
                    for r in cases()
                    for b in GOLD
                    for s in repeats("gold", b)
                )
                parallel(tasks, gold_task, "prepare")
            else:
                verify()
                required = [
                    path("predictions", m, r["id"]) for r in cases() for m in PRED
                ]
                required += [
                    path(stage_name("episodes", s), m + "__" + b, r["id"])
                    for r in cases()
                    for m in ("parent", "weighted")
                    for b in ("rank", "balanced")
                    for s in repeats(m, b)
                ]
                missing = next((p for p in required if not p.exists()), None)
                if missing:
                    raise RuntimeError(
                        f"GPU stage incomplete. Resume gpu first: {missing}"
                    )
                parallel(cases(), finish_case, "finish")
            from .report import report

            summary = report()
            if a.stage == "finish" and any(
                v["all_seed_completed"] != v["all_seed_expected"]
                for v in summary.values()
            ):
                raise RuntimeError(
                    "Incomplete matrix. Check progress.json and resume prepare for missing gold rows."
                )


if __name__ == "__main__":
    main()
