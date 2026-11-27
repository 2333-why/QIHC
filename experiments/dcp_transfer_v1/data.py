"""Train-only DCP templates, parameter variants, and an independent official oracle."""

import copy
import json
import math
import random


from experiments.dcp_external_v1.prepare import BUS, CLIQUE, KNAPSACK, formulation
from experiments.dcp_external_v2.reference import coordinates, source_model
from qihc.experiment_runtime.pbit_control import seed
from qihc.generic.bounded import objective, validate, violations

CAPITAL = "capital_budget"
COINS = "session1_money_change"
TEMPLATES = {BUS, CLIQUE, KNAPSACK, CAPITAL, COINS}


def split(cases):
    train = [c for c in cases if c["dataset"] == "dcp" and c["template"] in TEMPLATES]
    test = [c for c in cases if c not in train]
    dcp = [c for c in test if c["dataset"] == "dcp"]
    if (len(train), len(dcp), sum(c["main"] for c in dcp), len(test)) != (
        25,
        210,
        159,
        990,
    ):
        raise ValueError("Unexpected DCP or NLCO split")
    if any(c["main"] for c in train) or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Main test or duplicate instance in training split")
    return train, test


def vary(template, data, rng):
    d = copy.deepcopy(data)
    if template in (KNAPSACK, CAPITAL):
        profit, weight, budget = (
            ("values", "weights", "capacity")
            if template == KNAPSACK
            else ("npv", "cash_flow", "budget")
        )
        # Preserve the units of monetary inputs when the IR uses reduced coefficients.
        unit = math.gcd(*d[profit]) or 1
        i = rng.randrange(len(d[profit]))
        d[profit][i] += unit * rng.randint(1, 5)
        step = math.gcd(*d[weight]) or 1
        d[budget] = max(step, d[budget] + step * rng.choice([-2, -1, 1, 2]))
    elif template == CLIQUE:
        i, j = rng.sample(range(d["n"]), 2)
        d["adj"][i][j] = d["adj"][j][i] = 1 - d["adj"][i][j]
    elif template == BUS:
        # Existing shifts are retained, so any original feasible cover is available.
        tasks = sorted(rng.sample(range(d["num_work"]), min(3, d["num_work"])))
        if tasks not in d["shifts"]:
            d["shifts"].append(tasks)
            d["num_shifts"] += 1
    elif template == COINS:
        counts = [rng.randint(0, v) for v in d["available_coins"]]
        d["amount"] = sum(n * v for n, v in zip(counts, d["types_of_coins"]))
    else:
        raise KeyError(template)
    return d


def bounded_model(template, data):
    if template == COINS:
        n = len(data["available_coins"])
        model = formulation(
            KNAPSACK,
            dict(
                values=[1] * n, weights=data["types_of_coins"], capacity=data["amount"]
            ),
        )
        model["objective"]["sense"] = "min"
        model["constraints"][0]["op"] = "eq"
        for v, bound in zip(model["variables"], data["available_coins"]):
            v["upper"] = bound
    elif template == CAPITAL:
        model = formulation(
            KNAPSACK,
            dict(
                values=data["npv"], weights=data["cash_flow"], capacity=data["budget"]
            ),
        )
    else:
        model = formulation(template, data)
    # Positive objective rescaling preserves all rankings; keep the original units if possible.
    coefs = model["objective"]["linear"]
    scale = math.gcd(*coefs) if max(map(abs, coefs)) > 10000 else 1
    model["objective"]["linear"] = [v // scale for v in coefs]
    for row in model["constraints"]:
        if max(abs(row["rhs"]), *map(abs, row["a"])) > 10000:
            factor = math.gcd(row["rhs"], *row["a"]) or 1
            row["rhs"] //= factor
            row["a"] = [v // factor for v in row["a"]]
    return validate(model), scale


def certify(template, data, spec, scale):
    from ortools.sat.python import cp_model

    source, outputs = source_model(template, data)
    projected, aliases = coordinates(template, data, source, outputs)
    prefix = {BUS: "x", KNAPSACK: "x", CAPITAL: "x", CLIQUE: "c", COINS: "coin_counts"}[
        template
    ]
    if aliases != [f"{prefix}[{i}]" for i in range(len(spec["variables"]))]:
        raise ValueError(f"Unexpected public decision order: {aliases}")
    if not source.solve(solver="ortools", time_limit=30, num_search_workers=1):
        raise ValueError("Official instance has no proven optimum")
    if source.status().exitstatus.name != "OPTIMAL":
        raise ValueError("Official optimum unproven")
    witness = [int(v.value()) for v in projected]
    expected = int(source.objective_value())
    if violations(spec, witness).sum() or objective(spec, witness) * scale != expected:
        raise ValueError("Official witness disagrees with training IR")
    cp = cp_model.CpModel()
    xs = [cp.new_int_var(v["lower"], v["upper"], v["name"]) for v in spec["variables"]]

    def linear(a):
        return sum(v * k for v, k in zip(xs, a) if k)

    for c in spec["constraints"]:
        expr, rhs = linear(c["a"]), c["rhs"]
        cp.add({"le": expr <= rhs, "ge": expr >= rhs, "eq": expr == rhs}[c["op"]])
    direction = spec["objective"]["sense"]
    (cp.minimize if direction == "min" else cp.maximize)(
        linear(spec["objective"]["linear"])
    )
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30
    solver.parameters.num_search_workers = 1
    if (
        solver.solve(cp) != cp_model.OPTIMAL
        or round(solver.objective_value) * scale != expected
    ):
        raise ValueError("Translated IR optimum disagrees with official model")
    # Independently project the IR optimum back into the source model as well.
    for v, x in zip(projected, xs):
        source += v == int(solver.value(x))
    if (
        not source.solve(solver="ortools", time_limit=30, num_search_workers=1)
        or int(source.objective_value()) != expected
    ):
        raise ValueError("Translated witness rejected by official model")
    return aliases, witness


def variants(case, source, seen):
    data = source["data"]
    template = case["template"]
    rng = random.Random(seed(case["id"], "dcp-variants"))
    description = case["question"].split("\nInstance: ", 1)[0]
    accepted, audit = [], []
    for index in range(5):
        example = data if not index else vary(template, data, rng)
        key = (template, json.dumps(example, sort_keys=True))
        if key in seen and index:
            audit.append(dict(index=index, reason="duplicate"))
            continue
        try:
            gold, scale = bounded_model(template, example)
        except ValueError as exc:
            audit.append(dict(index=index, reason=str(exc)))
            continue
        aliases, witness = certify(template, example, gold, scale)
        if not index and aliases != source["aliases"]:
            raise ValueError("Parent projection changed")
        cid = case["id"] if not index else f"{case['id']}__variant{index}"
        q = description + "\nInstance: " + json.dumps(example, separators=(",", ":"))
        if scale != 1:
            q += f"\nExpress the objective in units of {scale} original units."
        q += (
            "\nDecision convention: x0, x1, ... represent the following source quantities in order. Return a bounded integer model in this order.\n"
            + json.dumps(aliases)
        )
        child = dict(
            id=cid,
            template=template,
            variable_names=[v["name"] for v in gold["variables"]],
            question=q,
        )
        accepted.append((child, gold, witness))
        audit.append(dict(index=index, id=cid, objective_scale=scale))
        seen.add(key)
    return accepted, audit
