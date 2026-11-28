"""Evaluate on the original problems, reuse identical endpoints, and report paired gains."""

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from experiments.feedback_active_v3.runtime import seed_for
from experiments.pbit_dual_v1.report import paired, summarize
from experiments.pbit_joint_v1 import search
from experiments.pbit_stack_v1.cpu import evaluate, native
from experiments.pbit_stack_v1 import common as stack
from qihc.experiment_runtime.storage import load, save, value_digest, worker_lock
from .common import (
    HIST,
    NEW,
    OTHER,
    RUN,
    TASKS,
    TERMINALS,
    TRADITIONAL,
    manifest,
    version,
    path,
    prior,
    solver_seeds,
)

ARCHIVE = (*HIST, *OTHER)


def score_case(c):
    dcp = c["dataset"] == "dcp"
    original = load(prior.original_artifact(c))
    ref = load(Path(original["reference"])) if dcp else original["reference"]
    hist = load(path("historical", c, "rows"))
    memo = {}
    for name, fp in load(path("historical", c, "fingerprints")).items():
        for t in TERMINALS:
            for s in stack.seed_list(c, t):
                memo[fp, t, s] = hist[f"{name}__{t}__{s}"]
    for arm in NEW:
        source = path("predictions", c, arm)
        pred = load(source)
        model = pred["model"]
        if arm in ("joint", "feedback_only"):
            calls = pred["feedback_calls"]
            probe = (pred.get("probe") or {}).get("attempts", 0) + pred[
                "sampling_attempts"
            ]
        else:
            calls, probe = 0, 0
        for t in TERMINALS:
            for s in stack.seed_list(c, t):
                out = path("scores", c, arm + "__" + t, s)
                key = (value_digest(model), t, s)
                if out.exists():
                    memo[key] = load(out)
                    continue
                reused = key in memo
                if reused:
                    row = dict(memo[key])
                else:
                    if model is None:
                        ep = dict(x=None, status="input_invalid")
                    elif t == "scip":
                        ep = native(
                            source,
                            path("episodes", c, arm + "__" + t, s),
                            "scip",
                            s if dcp else seed_for(c["id"], "terminal"),
                            traced=not dcp,
                        )
                    else:
                        ep = search.episode(
                            model,
                            c["id"],
                            dict(stack.SEARCH, seed=s),
                            lambda *_: "cool",
                        )
                    row = evaluate(dcp, original, ref, ep)
                # Solver metrics may be shared. Intervention accounting belongs to this arm.
                row.pop("semantic", None)
                parent = (
                    load(path("predictions", c, "mixed"))
                    if arm == "joint"
                    else (
                        load(path("inputs", c, "anchor"))
                        if arm == "feedback_only"
                        else pred
                    )
                )
                generation = parent.get("calls", [])
                call_log = (
                    generation
                    + pred.get("proposer_calls", [])
                    + pred.get("judge_calls", [])
                )
                row.update(
                    valid=model is not None,
                    reused=reused,
                    feedback=pred.get("action"),
                    feedback_calls=calls,
                    probe_attempts=probe,
                    probe_seconds=pred.get("sampling_seconds", 0)
                    + (pred.get("probe") or {}).get("seconds", 0),
                    calls=len(call_log),
                    generation_seconds=sum(x["seconds"] for x in call_log),
                    prompt_tokens=sum(
                        x["usage"].get("prompt_tokens", 0) for x in call_log
                    ),
                    completion_tokens=sum(
                        x["usage"].get("completion_tokens", 0) for x in call_log
                    ),
                    retry=pred.get("retry", False),
                    parse_failed=pred.get("parse_failed", False),
                )
                save(out, row)
                memo[key] = row


def report(require=False):
    m = manifest()
    cohort = load(RUN / "cases.json")
    cases = []
    for c in cohort:
        hist = load(path("historical", c, "rows"))
        rows = {}
        for name in (*ARCHIVE, *NEW):
            for terminal in TERMINALS:
                arm = name + "__" + terminal
                entries = {}
                for s in stack.seed_list(c, terminal):
                    p = path("scores", c, arm, s)
                    if name in ARCHIVE:
                        entries[s] = hist[f"{arm}__{s}"]
                    elif p.exists():
                        entries[s] = load(p)
                rows[arm] = entries
        for method in TRADITIONAL:
            arm = "traditional__" + method
            rows[arm] = {s: hist[f"{arm}__{s}"] for s in solver_seeds(c, method)}
        cases.append((c, rows))
    groups = dict(
        nlco780=[v for v in cases if v[0]["dataset"] == "nlco"],
        dcp210=[v for v in cases if v[0]["dataset"] == "dcp"],
    )
    groups["nlco_main"] = [v for v in groups["nlco780"] if v[0]["stage"] == "main"]
    groups["nlco_scale"] = [v for v in groups["nlco780"] if v[0]["stage"] == "scale"]
    groups["dcp159"] = [v for v in groups["dcp210"] if v[0]["main"]]
    result = dict(
        experiment="E005-1.20",
        source=m["code"],
        runtime_code=version(),
        training=m["counts"],
        groups={},
    )
    for group, subset in groups.items():
        stats, contrasts = {}, {}
        for arm in subset[0][1]:
            series = [rows[arm] for _, rows in subset]
            terminal = arm.rsplit("__", 1)[1]
            seeds = solver_seeds if arm.startswith("traditional__") else stack.seed_list
            stats[arm] = summarize(
                series, len(subset), len(seeds(subset[0][0], terminal))
            )
            stats[arm]["format_retries"] = sum(
                bool(r.get(17, {}).get("retry")) for r in series
            )
            stats[arm]["final_parse_errors"] = sum(
                bool(r.get(17, {}).get("parse_failed")) for r in series
            )
        for new, old in (
            ("mixed", "replay"),
            ("mixed", "anchor"),
            ("feedback_only", "anchor"),
            ("feedback_only", "old_feedback"),
            ("joint", "mixed"),
            ("joint", "champion"),
            ("joint", "double_review"),
            ("joint", "base"),
        ):
            for t in TERMINALS:
                a, b = new + "__" + t, old + "__" + t
                contrasts[f"{a} - {b}"] = paired(subset, a, b)
        required = [stats[f"{a}__{t}"] for a in NEW for t in TERMINALS]
        done = sum(r["repeats_completed"] for r in required)
        expected = sum(r["repeats_expected"] for r in required)
        result["groups"][group] = dict(
            cases=len(subset),
            completed=done,
            expected=expected,
            arms={k: v for k, v in stats.items() if not k.startswith("traditional__")},
            traditional={
                k.split("__")[1]: v
                for k, v in stats.items()
                if k.startswith("traditional__")
            },
            pairs=contrasts,
        )
        print(group, done, "/", expected, flush=True)
    result["completed"] = sum(
        result["groups"][g]["completed"] for g in ("nlco780", "dcp210")
    )
    from experiments.pbit_joint_v1.run import training_status

    result["adapters"] = {
        task: training_status(
            RUN / "trained" / task,
            2 * m["counts"]["dcp_patch" if task == "feedback" else "dcp"],
        )
        for task in TASKS
    }
    result["complete"] = result["completed"] == 9600 and all(
        row["passed"] for row in result["adapters"].values()
    )
    if require and not result["complete"]:
        raise ValueError("Incomplete transfer matrix")
    save(RUN / ("report.json" if result["complete"] else "partial_report.json"), result)


def finish():
    manifest(True)
    if not all((RUN / "gpu" / f"{i}.json").exists() for i in range(4)):
        raise ValueError("GPU inference shards incomplete")
    with (
        worker_lock(RUN / "finish.lock"),
        ProcessPoolExecutor(
            max_workers=int(os.getenv("QIHC_CPU_WORKERS", "6"))
        ) as pool,
    ):
        for i, _ in enumerate(pool.map(score_case, load(RUN / "cases.json")), 1):
            if i % 50 == 0:
                print("scored", i, flush=True)
        report(True)
