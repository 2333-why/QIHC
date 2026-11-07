"""Same candidates, independent verifier and single-thread wall-clock allowance."""
import argparse
import copy
import time
from pathlib import Path

import numpy as np

from experiments.posttraining_v1.data import DATA, RUN, prepare
from experiments.posttraining_v1.witness import evidence, probe
from qihc.experiment_runtime.storage import load, save
from qihc.generic.feedback.active import _scip_violation
from qihc.generic.feedback.verification import equivalent
from qihc.generic.solver_baselines.registry import solve


def candidate(gold, fault):
    model = copy.deepcopy(gold)
    if fault == "drop_row":
        if not model["constraints"]:
            return None
        model["constraints"].pop(0)
    else:
        obj = model["objective"]
        obj["sense"] = "min" if obj["sense"] == "max" else "max"
    return model


def classical(model, gold, budget, seed):
    deadline = time.perf_counter() + budget
    states = []
    # Search for violations in either direction. Never consume the mutation label.
    def rows(spec):
        return {(tuple(c["a"]), c["op"], c["rhs"]) for c in spec["constraints"]}
    for source, other in ((model, gold), (gold, model)):
        for a, op, rhs in sorted(rows(other) - rows(source)):
            left = deadline - time.perf_counter()
            if left <= 0:
                break
            xs, _ = _scip_violation(source, dict(a=list(a), op=op, rhs=rhs), left, seed)
            states.extend(xs)
    if model["objective"] != gold["objective"]:
        for source in (model, gold):
            left = deadline - time.perf_counter()
            if left <= 0:
                break
            row = solve(source, "scip", left, seed)
            if row.get("x") is not None:
                states.append(row["x"])
    return evidence(model, gold, states)[:4] if states else []


def uniform(model, gold, budget, seed):
    deadline = time.perf_counter() + budget
    rng = np.random.default_rng(seed)
    lo = np.array([v["lower"] for v in model["variables"]])
    hi = np.array([v["upper"] for v in model["variables"]])
    while time.perf_counter() < deadline:
        cards = evidence(model, gold, rng.integers(lo, hi+1, size=(32,len(lo))))
        if cards:
            return cards[:4]
    return []


def run(budget, out):
    if budget <= 0: raise ValueError("Positive time budget required")
    if not (RUN/"split.json").exists(): prepare()
    validation = load(RUN/"validation.json")
    solve(validation[0]["gold"], "scip", 1, 17)  # exclude first native import
    result = []
    for row in validation:
        gold = row["gold"]
        for fault in ("drop_row", "reverse_objective"):
            model = candidate(gold, fault)
            if model is None:
                result.append(dict(id=row["case"]["id"], fault=fault, status="not_applicable"))
                continue
            status = equivalent(model, gold)["status"]
            item = dict(id=row["case"]["id"], fault=fault, status=status, probes={})
            for arm in ("uniform","scip","pbit"):
                started = time.perf_counter()
                if arm=="pbit":
                    cards, _ = probe(model,gold,17,budget)
                else:
                    cards = (uniform if arm=="uniform" else classical)(model,gold,budget,17)
                item["probes"][arm] = dict(hit=bool(cards), seconds=time.perf_counter()-started, cards=cards)
            result.append(item)
            save(out,dict(seed=17, budget_seconds=budget, cases=result))
    for fault in ("drop_row","reverse_objective"):
        items=[r for r in result if r["fault"]==fault and r["status"]=="disproved"]
        print(fault,len(items),{m:sum(r["probes"][m]["hit"] for r in items) for m in ("uniform","scip","pbit")})


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--seconds",type=float,default=1.)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    run(args.seconds,args.out)
