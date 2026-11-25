"""Score new artifacts on the frozen original objectives and constraints."""

from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
import subprocess
import sys

from experiments.dcp_external_v2.cpu import score_one
from experiments.pbit_joint_v1 import search
from qihc.experiment_runtime.storage import load, save, value_digest, worker_lock
from qihc.generic.feedback.verification import equivalent, score
from qihc.generic.solver_baselines.registry import solve
from .common import (
    RUN,
    SEARCH,
    artifact,
    cases,
    get_prediction,
    methods,
    original_artifact,
    seeds,
    verify,
)


def scip_episode(case, model, seed):
    path = artifact("episodes", case, f"{model}__scip", seed)
    if not path.exists():
        pred = artifact("predictions", case, model)
        args = [
            sys.executable,
            "-m",
            "experiments.pbit_dual_v1.cpu",
            str(pred),
            str(path),
            str(seed),
        ]
        try:
            subprocess.run(args, check=True, timeout=50)
        except subprocess.TimeoutExpired:
            save(path, dict(x=None, status="process_time_limit", seconds=50))
    return load(path)


def score_case(case):

    cid = case["id"]
    dataset = case["dataset"]
    old = load(original_artifact(case))
    ref = load(Path(old["reference"])) if dataset == "dcp" else old["reference"]
    memo, semantics = {}, {}
    for model, terminal in methods(case):
        arm = f"{model}__{terminal}"
        pred = get_prediction(case, model)
        spec = pred["model"]
        fingerprint = value_digest(spec)
        for s in seeds(case, terminal):
            dest = artifact("scores", case, arm, s)
            key = (fingerprint, terminal, s)
            if dest.exists():
                memo[key] = load(dest)
                continue
            episode = (
                load(artifact("episodes", case, arm, s))
                if terminal == "control"
                else None
            )
            score_method = (
                "schedule" if episode and episode.get("reuse_schedule") else terminal
            )
            if score_method != "control" and spec == old["prediction"]["model"]:
                if dataset == "dcp":
                    row = dict(old["legacy"][f"weighted__{score_method}__{s}"])
                else:
                    row = nlco_score(old["legacy"]["weighted__schedule"])
                row["reused"] = "weighted"
            elif (fingerprint, score_method, s) in memo:
                row = dict(
                    memo[fingerprint, score_method, s], reused="new_identical_model"
                )
            else:
                if score_method == "control":
                    pass
                elif spec is None:
                    episode = dict(x=None, status="input_invalid")
                elif score_method == "scip":
                    episode = scip_episode(case, model, s)
                else:
                    episode = search.episode(
                        spec, cid, dict(SEARCH, seed=s), lambda *_: "cool"
                    )
                if dataset == "dcp":
                    row = score_one(ref, episode)
                else:
                    if fingerprint not in semantics:
                        semantics[fingerprint] = equivalent(spec, old["gold"])["status"]
                    row = nlco_score(
                        dict(
                            score=score(old["gold"], episode, ref),
                            invalid=spec is None,
                            semantic=semantics[fingerprint],
                            actions=episode.get("actions", []),
                            attempts=episode.get("heat_bath_attempts", 0),
                            writes=episode.get("intervention_writes", 0),
                            episode_seconds=episode.get("episode_seconds", 0),
                            decision_seconds=episode.get("decision_seconds", 0),
                        )
                    )
                    row["status"] = episode.get("status", "evaluated")
                memo[fingerprint, score_method, s] = row
            calls = pred.get("calls", [])
            row.update(
                valid=spec is not None,
                feedback=pred.get("action"),
                feedback_calls=pred.get("feedback_calls", 0),
                calls=len(calls),
                prompt_tokens=sum(
                    (c.get("usage") or {}).get("prompt_tokens", 0) or 0 for c in calls
                ),
                completion_tokens=sum(
                    (c.get("usage") or {}).get("completion_tokens", 0) or 0
                    for c in calls
                ),
                generation_seconds=pred.get(
                    "generation_seconds", sum(c.get("seconds", 0) for c in calls)
                ),
                probe_seconds=sum(
                    e["probe"].get("seconds", 0) for e in pred.get("evidence", [])
                ),
                probe_attempts=sum(
                    e["probe"]["attempts"] for e in pred.get("evidence", [])
                ),
            )
            save(dest, row)


def nlco_score(cell):
    result = cell["score"]
    return dict(
        hit=result["optimal_hit"],
        feasible=result["original_feasible"],
        reference_proven=result["reference_proven"],
        gap=result["gap"],
        valid=not cell["invalid"],
        semantic=cell["semantic"],
        status="input_invalid" if cell["invalid"] else "evaluated",
        actions=cell["actions"],
        attempts=cell.get("attempts", 0),
        intervention_writes=cell.get("writes", 0),
        seconds=cell.get("episode_seconds", 0),
        decision_seconds=cell.get("decision_seconds", 0),
    )


def finish():
    verify()
    cohort = cases()
    with worker_lock(RUN / "finish.lock"):
        for case in cohort:
            for model, method in methods(case):
                if (
                    model != "weighted"
                    and not artifact("predictions", case, model).exists()
                ):
                    raise FileNotFoundError(artifact("predictions", case, model))
                if method == "control":
                    for s in seeds(case, method):
                        path = artifact("episodes", case, f"{model}__{method}", s)
                        if not path.exists():
                            raise FileNotFoundError(path)
        with ProcessPoolExecutor(
            max_workers=int(os.getenv("QIHC_CPU_WORKERS", "6"))
        ) as pool:
            for i, _ in enumerate(pool.map(score_case, cohort), 1):
                if i % 25 == 0 or i == len(cohort):
                    print("scoring", i, "/", len(cohort), flush=True)
    from .report import report

    report()


if __name__ == "__main__":
    prediction, output, seed = sys.argv[1:]
    model = load(Path(prediction))["model"]
    save(Path(output), solve(model, "scip", 20, int(seed)))
