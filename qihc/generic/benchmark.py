"""Synthetic NL benchmarks. Gold IR belongs to teachers/evaluators, not inference."""
from __future__ import annotations

import json
import numpy as np
from .energy import validate

FAMILIES = ("knapsack", "vertex_cover", "balanced_cut", "assignment")


def make_case(family, size, seed, split="train"):
    rng = np.random.default_rng(seed)
    n = size * size if family == "assignment" else size
    constraints, quadratic = [], []

    def con(a, op, rhs):
        constraints.append({"id": f"c{len(constraints)}", "a": list(map(int, a)), "op": op, "rhs": int(rhs)})

    if family == "knapsack":
        weights = rng.integers(1, 10, n).tolist()
        linear = rng.integers(1, 20, n).tolist()
        cap = sum(weights) // 3
        con(weights, "le", cap)
        sense = "max"
        text = f"Select items to maximize total value. Item values={linear}; item weights={weights}. Total selected weight must not exceed {cap}. x_i=1 means selecting item i."
    elif family in ("vertex_cover", "balanced_cut"):
        edges = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < min(.22, 5 / n)]
        if family == "vertex_cover":
            linear = rng.integers(1, 10, n).tolist()
            sense = "min"
            for i, j in edges:
                a = [0] * n
                a[i] = a[j] = 1
                con(a, "ge", 1)
            text = f"Choose a minimum-cost vertex cover. Vertex costs={linear}; undirected edges={edges}. Every edge must have at least one selected endpoint. x_i=1 means vertex i is selected."
        else:
            linear = [0] * n
            sense = "max"
            for i, j in edges:
                linear[i] += 1
                linear[j] += 1
                quadratic.append([i, j, -2])
            con([1] * n, "eq", n // 2)
            text = f"Maximize the number of edges crossing two groups. Undirected unweighted edges={edges}. Exactly {n // 2} vertices must belong to group 1; the others belong to group 0. x_i is vertex i's group."
    elif family == "assignment":
        linear = rng.integers(1, 30, n).tolist()
        sense = "min"
        for i in range(size):
            con([int(k // size == i) for k in range(n)], "eq", 1)
        for j in range(size):
            con([int(k % size == j) for k in range(n)], "eq", 1)
        text = f"Assign {size} workers to {size} jobs at minimum total cost. Each worker gets exactly one job and each job exactly one worker. Row-major costs={linear}. x_(i*{size}+j)=1 means worker i takes job j."
    else:
        raise ValueError(family)
    # Split-specific surface wording; all domains use exactly the same IR compiler.
    prefix = {"train": "Formulate this decision problem.", "validation": "Translate the following planning request precisely.",
              "test": "We need an optimal plan satisfying ALL of these rules."}[split]
    description = f"{prefix} All indices start at 0; there are {n} binary variables x_0,...,x_{n-1}. {text}"
    spec = validate({"n": n, "objective": {"sense": sense, "linear": linear, "quadratic": quadratic}, "constraints": constraints})
    return {"id": f"{family}-{size}-{seed}", "family": family, "size": size, "split": split,
            "description": description, "gold": spec}


def make_dataset(per_family=24, seed=20261017):
    if per_family < 8:
        raise ValueError("At least eight cases per family")
    rows = []
    for fi, family in enumerate(FAMILIES):
        for i in range(per_family):
            split = "train" if i < per_family * 2 // 3 else "validation" if i < per_family * 5 // 6 else "test"
            size = (3, 4, 5)[i % 3] if family == "assignment" else (12, 20, 32)[i % 3]
            rows.append(make_case(family, size, seed + fi * 100000 + i, split))
    return rows


def objective_prompt(description):
    return ("Translate the objective ONLY. Return JSON with exactly n (number of binary variables) and objective. "
            "objective has sense=min|max, linear=[integer coefficients for x_0..x_(n-1)], "
            "quadratic=[[i,j,integer coefficient],...] with unique i<j. Expand the true objective algebraically. "
            "Do not include constraint penalties or constant offsets. No prose/code.\n" + description)


def constraint_prompt(description, first):
    return ("Translate ALL hard constraints and select a search policy. Return JSON with exactly constraints and policy. "
            "constraints is a list of {id:unique string,a:[integer coefficients for all n variables],op:le|ge|eq,rhs:integer}. "
            "These are original linear constraints, NOT squared expansions. The compiler introduces bounded slack "
            "and squares each residual. policy must be one of fixed, adaptive, warm, cold. "
            "fixed: weight2,temp1; adaptive: weight0.1,temp1,violation feedback; warm: weight0.5,temp4,feedback; "
            "cold: weight2,temp0.25. All temperatures anneal to 2%; feedback caps weights at8 "
            "and raises each weight to at least2 in the final quarter. "
            "Never remove a constraint to reduce energy. No prose/code.\nProblem:\n" + description +
            "\nPreviously translated objective:\n" + json.dumps(first, separators=(",", ":")))


def parse_object(text):
    text = text.strip()
    if text.startswith("```json") and text.endswith("```"):
        text = text[7:-3].strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object")
    return value


def infer(generator, description):
    """No gold argument exists: failed translation is a recorded failure, not fallback."""
    first_text = generator.generate(objective_prompt(description))
    first = parse_object(first_text)
    if set(first) != {"n", "objective"}:
        raise ValueError("Objective response has unexpected fields")
    validate({**first, "constraints": []})
    second_text = generator.generate(constraint_prompt(description, first))
    second = parse_object(second_text)
    if set(second) != {"constraints", "policy"} or second["policy"] not in ("fixed", "adaptive", "warm", "cold"):
        raise ValueError("Invalid constraint/policy output")
    return validate({**first, "constraints": second["constraints"]}), second["policy"], [first_text, second_text]


def semantic_equal(predicted, gold):
    """Conservative structural match; may reject algebraically equivalent models."""
    def canonical(spec):
        obj = spec["objective"]
        cs = sorted((tuple(c["a"]), c["op"], c["rhs"]) for c in spec["constraints"])
        return (spec["n"], obj["sense"], tuple(obj["linear"]), sorted(map(tuple, obj["quadratic"])), cs)
    return canonical(predicted) == canonical(gold)


def reference_solve(spec, seconds=10):
    """Independent integer reference, with explicit time limit/status/bound."""
    from ortools.sat.python import cp_model
    import time
    started = time.perf_counter()
    model = cp_model.CpModel()
    x = [model.NewBoolVar(f"x{i}") for i in range(spec["n"])]
    expression = sum(c * x[i] for i, c in enumerate(spec["objective"]["linear"]))
    for i, j, c in spec["objective"]["quadratic"]:
        product = model.NewBoolVar(f"p{i}_{j}")
        model.AddMultiplicationEquality(product, [x[i], x[j]])
        expression += c * product
    for con in spec["constraints"]:
        lhs = sum(c * bit for c, bit in zip(con["a"], x))
        model.Add(lhs <= con["rhs"] if con["op"] == "le" else lhs >= con["rhs"] if con["op"] == "ge" else lhs == con["rhs"])
    (model.Minimize if spec["objective"]["sense"] == "min" else model.Maximize)(expression)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = seconds
    solver.parameters.num_search_workers = 1
    status = solver.Solve(model)
    feasible = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    return {"status": solver.StatusName(status), "x": [solver.Value(v) for v in x] if feasible else None,
            "objective": solver.ObjectiveValue() if feasible else None,
            "bound": solver.BestObjectiveBound(), "solver_seconds": solver.WallTime(),
            "seconds": time.perf_counter() - started}
