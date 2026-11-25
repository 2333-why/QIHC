"""Summarize new and frozen results, with matched-case comparisons."""

from collections import Counter, defaultdict
from statistics import mean

import numpy as np

from qihc.experiment_runtime.storage import load, save
from .common import RUN, artifact, cases, methods, seeds
from .cpu import nlco_score


CONTRASTS = (
    ("trained__schedule", "weighted__schedule"),
    ("weighted_feedback__schedule", "weighted__schedule"),
    ("weighted_feedback__schedule", "weighted_quadratic__schedule"),
    ("weighted__control", "weighted__schedule"),
    ("trained__control", "trained__schedule"),
    ("trained_feedback__schedule", "trained__schedule"),
    ("trained_feedback__control", "trained_feedback__schedule"),
    ("trained_feedback__control", "weighted__schedule"),
)


def case_results(case):
    frozen = load(RUN / "frozen" / case["dataset"] / (case["id"] + ".json"))
    rows = defaultdict(dict)
    if case["dataset"] == "dcp":
        for key, row in frozen["legacy"].items():
            arm, seed = key.rsplit("__", 1)
            rows[arm][int(seed)] = row
    else:
        for prefix, track in (("", "legacy"), ("prior14/", "legacy14")):
            for arm, row in frozen[track].items():
                rows[prefix + arm][17] = nlco_score(row)
    for model, method in methods(case):
        arm = f"{model}__{method}"
        counts = rows[arm]
        for s in seeds(case, method):
            path = artifact("scores", case, arm, s)
            if path.exists():
                counts[s] = load(path)
    return rows


def summarize(rows, expected, n_seeds):
    primary = [r[17] for r in rows if 17 in r]
    repeats = [row for rs in rows for row in rs.values()]
    gaps = [
        r["gap"]
        for r in primary
        if r["gap"] is not None and r.get("reference_proven", True)
    ]
    counters = {
        "actions": Counter(a for r in repeats for a in r.get("actions", [])),
        "status": Counter(r["status"] for r in primary),
        "feedback": Counter(r["feedback"] for r in primary if r.get("feedback")),
        "semantic": Counter(r["semantic"] for r in primary if "semantic" in r),
    }
    costs = {
        k: sum(r.get(k) or 0 for r in primary)
        for k in (
            "calls",
            "feedback_calls",
            "prompt_tokens",
            "completion_tokens",
            "generation_seconds",
            "probe_seconds",
            "probe_attempts",
            "seconds",
            "decision_seconds",
            "attempts",
            "intervention_writes",
        )
    }
    valid = [r["valid"] for r in primary if r.get("valid") is not None]
    return dict(
        expected=expected,
        complete=len(primary),
        valid=sum(valid) if valid else None,
        feasible=sum(r["feasible"] for r in primary),
        hits=sum(r["hit"] for r in primary),
        mean_gap=mean(gaps) if gaps else None,
        gap_cases=len(gaps),
        repeats_completed=len(repeats),
        repeats_expected=expected * n_seeds,
        # Return search-seed averages only after every seed for every case is present.
        mean_hit=(
            mean(mean(r["hit"] for r in rs.values()) for rs in rows)
            if len(repeats) == expected * n_seeds
            else None
        ),
        reused=sum(bool(r.get("reused")) for r in repeats),
        intervened=sum(any(a != "cool" for a in r.get("actions", ())) for r in repeats),
        costs=costs,
        **counters,
    )


def paired(subset, a, b):
    clusters = defaultdict(list)
    for case, rows in subset:
        if 17 not in rows.get(a, {}) or 17 not in rows.get(b, {}):
            return None
        change = int(rows[a][17]["hit"]) - int(rows[b][17]["hit"])
        clusters[case["template"] if case["dataset"] == "dcp" else case["id"]].append(
            change
        )
    totals = np.asarray([sum(values) for values in clusters.values()])
    sizes = np.asarray([len(values) for values in clusters.values()])
    draws = np.random.default_rng(17).integers(len(totals), size=(5000, len(totals)))
    deltas = totals[draws].sum(axis=1) / sizes[draws].sum(axis=1)
    changes = [x for values in clusters.values() for x in values]
    return dict(
        delta=mean(changes),
        gained=changes.count(1),
        lost=changes.count(-1),
        ci95=np.quantile(deltas, [0.025, 0.975]).tolist(),
    )


def report():
    cohort = [(case, case_results(case)) for case in cases()]
    groups = {
        "nlco780": [r for r in cohort if r[0]["dataset"] == "nlco"],
        "dcp235": [r for r in cohort if r[0]["dataset"] == "dcp"],
    }
    for stage in ("main", "scale"):
        groups[f"nlco_{stage}"] = [
            r for r in groups["nlco780"] if r[0]["stage"] == stage
        ]
    groups["dcp159"] = [r for r in groups["dcp235"] if r[0]["main"]]
    manifest = load(RUN / "manifest.json")
    result = dict(
        experiment="E005-1.18",
        code=manifest["code"],
        supervision=manifest["supervision"],
        gpu_shards=sum((RUN / "shards" / f"{i}.json").exists() for i in range(4)),
        scores={},
        paired={},
    )
    for name, subset in groups.items():
        if not subset:
            continue
        example = subset[0][0]
        added = {f"{m}__{a}": len(seeds(example, a)) for m, a in methods(example)}
        arms = sorted({a for _, rows in subset for a in rows})
        stats = {}
        for arm in arms:
            rows = [row.get(arm, {}) for _, row in subset]
            n_seeds = added[arm] if arm in added else max(map(len, rows))
            stats[arm] = dict(
                summarize(rows, len(subset), n_seeds), historical=arm not in added
            )
        complete = sum(stats[a]["repeats_completed"] for a in added)
        expected = sum(stats[a]["repeats_expected"] for a in added)
        result["scores"][name] = dict(
            new_complete=complete, new_expected=expected, arms=stats
        )
        result["paired"][name] = {
            f"{a} - {b}": paired(subset, a, b) for a, b in CONTRASTS
        }
        print(name, "new", complete, "/", expected, flush=True)
        if name in ("nlco780", "dcp159"):
            for arm in ("weighted__schedule", *sorted(added)):
                row = stats[arm]
                print(
                    arm,
                    "scored",
                    f"{row['complete']}/{row['expected']}",
                    "feasible",
                    row["feasible"],
                    "optimal",
                    row["hits"],
                    "seed_mean_hit",
                    row["mean_hit"],
                    flush=True,
                )
    save(RUN / "report.json", result)
