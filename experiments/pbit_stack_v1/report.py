"""Keep the frozen incumbent report and final paired results separate."""

from collections import defaultdict

from experiments.pbit_dual_v1.report import paired, summarize
from qihc.experiment_runtime.storage import load, save
from .common import RUN, TRADITIONAL, arms, path, seed_list

CONTRASTS = (
    ("incumbent_B__schedule", "incumbent__schedule"),
    ("incumbent_C__schedule", "incumbent__schedule"),
    ("incumbent_C__schedule", "incumbent_B__schedule"),
    ("incumbent_pbit__schedule", "incumbent__schedule"),
    ("new__schedule", "incumbent__schedule"),
    ("new__scip", "incumbent__scip"),
    ("incumbent_F__schedule", "incumbent__schedule"),
    ("incumbent_F__schedule", "incumbent_C__schedule"),
    ("incumbent_F__schedule", "incumbent_pbit__schedule"),
    ("new_F__schedule", "new__schedule"),
    ("incumbent__learned_penalty", "incumbent__schedule"),
    ("incumbent__learned_penalty", "incumbent__rule_penalty"),
    ("new_F__learned_penalty", "new_F__rule_penalty"),
    ("new_F__learned_penalty", "incumbent_C__schedule"),
    ("new_F__learned_penalty", "incumbent_pbit__schedule"),
)


def all_rows(case, stage):
    rows = defaultdict(dict)
    archived = load(path("archived", case, "scores"))
    for key, row in archived.items():
        if key.startswith("base20__"):
            arm, s = key.rsplit("__", 1)
            rows[arm][int(s)] = row
    planned = list(arms(case, "current"))
    if stage == "improved":
        planned += list(arms(case, "improved"))
    if case["dataset"] == "nlco":
        planned += [("gold20", t) for t in TRADITIONAL]
    for model, terminal in planned:
        arm = f"{model}__{terminal}"
        for s in seed_list(case, terminal):
            p = path("scores", case, arm, s)
            if p.exists():
                rows[arm][s] = load(p)
        rows.setdefault(arm, {})
    return rows


def report(stage, require_complete=False):
    data = [(c, all_rows(c, stage)) for c in load(RUN / "cases.json")]
    groups = dict(
        nlco780=[r for r in data if r[0]["dataset"] == "nlco"],
        nlco_main=[
            r for r in data if r[0]["dataset"] == "nlco" and r[0]["stage"] == "main"
        ],
        nlco_scale=[
            r for r in data if r[0]["dataset"] == "nlco" and r[0]["stage"] == "scale"
        ],
        dcp235=[r for r in data if r[0]["dataset"] == "dcp"],
        dcp159=[r for r in data if r[0]["dataset"] == "dcp" and r[0]["main"]],
    )
    corpus = RUN / "training.json"
    result = dict(
        experiment="E005-1.19",
        stage=stage,
        scores={},
        pairs={},
        source=load(RUN / "manifest.json")["code"],
        supervision=(
            load(corpus)["totals"] if stage == "improved" and corpus.exists() else None
        ),
        penalty_support=(
            load(corpus)["support"] if stage == "improved" and corpus.exists() else None
        ),
    )
    history = load(RUN / "historical18.json")
    for group, subset in groups.items():
        if not subset:
            continue
        names = sorted({n for _, rows in subset for n in rows})
        stats = {}
        for arm in names:
            states = [rows.get(arm, {}) for _, rows in subset]
            term = arm.rsplit("__", 1)[-1]
            count = len(seed_list(subset[0][0], term))
            score = summarize(states, len(subset), count)
            score["intervened"] = sum(
                any(a not in ("cool", "fixed") for a in r.get("actions", []))
                for row in states
                for r in row.values()
            )
            score["penalty_writes"] = sum(
                r.get("penalty_writes", 0) for row in states for r in row.values()
            )
            stats[arm] = score
        required = [r for a, r in stats.items() if not a.startswith("base20__")]
        complete = sum(r["repeats_completed"] for r in required)
        expected = sum(r["repeats_expected"] for r in required)
        historical = history["scores"][group]["arms"]
        leaders = {}
        for terminal in ("schedule", "scip"):
            eligible = {
                a: r["hits"]
                for a, r in stats.items()
                if a.endswith("__" + terminal) and r["complete"] == r["expected"]
            }
            eligible.update(
                {
                    "archive18/" + a: r["hits"]
                    for a, r in historical.items()
                    if a.endswith("__" + terminal)
                    and not a.startswith("gold__")
                    and r["complete"] == r["expected"]
                }
            )
            if eligible:
                best = max(eligible.values())
                leaders[terminal] = dict(
                    hits=best,
                    cases=len(subset),
                    arms=[a for a, hits in eligible.items() if hits == best],
                )
        result["scores"][group] = dict(
            complete=complete,
            expected=expected,
            arms=stats,
            archive18=historical,
            matched_leaders=leaders,
        )
        result["pairs"][group] = {
            f"{a} - {b}": paired(subset, a, b) for a, b in CONTRASTS
        }
        print(group, complete, "/", expected, flush=True)
        if group in ("nlco780", "dcp159"):
            for a, r in stats.items():
                if a.startswith(
                    ("incumbent", "new", "pvcd", "sft", "sdpo", "solver_rl")
                ):
                    print(
                        a,
                        "optimal",
                        f"{r['hits']}/{r['complete']}",
                        "feasible",
                        r["feasible"],
                        "seed_mean",
                        r["mean_hit"],
                        flush=True,
                    )
        if require_complete and complete != expected:
            raise ValueError(f"Incomplete {stage} matrix for {group}")
    save(RUN / f"{stage}_report.json", result)
    if require_complete and stage == "current":
        from experiments.pbit_dual_v1.common import freeze

        freeze(RUN / "current.json", result)
    return result
