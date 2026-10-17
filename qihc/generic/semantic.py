"""Witness-checked semantic preferences, independent of sampler energy/reward.

Gold is available ONLY to data generation and evaluation. A counterexample proves
non-equivalence; a timeout or absence of a witness is never treated as a proof.
"""
from __future__ import annotations

from copy import deepcopy
import json
import numpy as np

from .energy import POLICIES, validate, objective, violations


def joint_prompt(description):
    return (
        "Translate this binary optimization problem into JSON with exactly n, objective, constraints, policy. "
        "n is the number of original binary variables. objective={sense:min|max,linear:[integer coefficients "
        "in variable order],quadratic:[[i,j,integer coefficient],...]}, with unique i<j. "
        "constraints=[{id:unique string,a:[integer coefficients in variable order],op:le|ge|eq,rhs:integer}]. "
        "Preserve ALL rules, coefficient signs, variable indices and inequality directions. Expand the original "
        "objective, not penalties. Constraints are original linear relations, not squared residuals. "
        "For exactly-one rules SUM all applicable variables; for each edge cover at least one endpoint. "
        "The compiler, not you, introduces slack and squares residuals. "
        "policy is fixed (weight2,temp1,no feedback), adaptive (weight0.1,temp1,feedback), "
        "warm (weight0.5,temp4,feedback), or cold (weight2,temp0.25,no feedback). "
        "All anneal to 2%; feedback caps weights at8 and raises weights to at least2 in the final quarter. "
        "Check dimensions, every rule and objective signs before responding. JSON only, no prose/code.\n"
        + description
    )


def completion(spec, policy):
    return json.dumps({**spec, "policy": policy}, separators=(",", ":"))


def parse_joint(text):
    from .benchmark import parse_object
    answer = parse_object(text)
    if set(answer) != {"n", "objective", "constraints", "policy"} or answer["policy"] not in POLICIES:
        raise ValueError("Expected full joint IR and valid policy")
    return validate({k: v for k, v in answer.items() if k != "policy"}), answer["policy"]


def infer_joint(generator, description):
    raw = generator.generate(joint_prompt(description))
    spec, policy = parse_joint(raw)
    return spec, policy, [raw]


def objective_witness(gold, candidate):
    """Exact polynomial identity check in the minimization convention on {0,1}^n.

    This is expression fidelity, not equality of argmin sets. Positive rescalings
    can preserve optima but change the requested physical objective's units.
    """
    if gold["n"] != candidate["n"]:
        return {"kind": "dimension", "gold_n": gold["n"], "candidate_n": candidate["n"]}
    n = gold["n"]
    sg = 1 if gold["objective"]["sense"] == "min" else -1
    sc = 1 if candidate["objective"]["sense"] == "min" else -1
    diff = sg * np.array(gold["objective"]["linear"]) - sc * np.array(candidate["objective"]["linear"])
    for i in np.flatnonzero(diff):
        x = [0] * n
        x[int(i)] = 1
        return {"kind": "objective", "x": x, "gold": float(sg * objective(gold, x)),
                "candidate": float(sc * objective(candidate, x))}
    gq = {(i, j): sg * c for i, j, c in gold["objective"]["quadratic"]}
    cq = {(i, j): sc * c for i, j, c in candidate["objective"]["quadratic"]}
    for i, j in sorted(gq.keys() | cq.keys()):
        if gq.get((i, j), 0) != cq.get((i, j), 0):
            x = [0] * n
            x[i] = x[j] = 1
            return {"kind": "objective", "x": x, "gold": float(sg * objective(gold, x)),
                    "candidate": float(sc * objective(candidate, x))}
    return None


def constraint_witness(gold, candidate, seconds=1.0):
    """Find an assignment feasible for exactly one of the two formulations."""
    from ortools.sat.python import cp_model
    if gold["n"] != candidate["n"]:
        return {"kind": "dimension", "gold_n": gold["n"], "candidate_n": candidate["n"]}
    for accept, reject in ((gold, candidate), (candidate, gold)):
        model = cp_model.CpModel()
        x = [model.NewBoolVar(f"x{i}") for i in range(gold["n"])]
        for c in accept["constraints"]:
            lhs = sum(a * b for a, b in zip(c["a"], x))
            model.Add(lhs <= c["rhs"] if c["op"] == "le" else lhs >= c["rhs"] if c["op"] == "ge" else lhs == c["rhs"])
        flags = []
        for c in reject["constraints"]:
            lhs = sum(a * b for a, b in zip(c["a"], x))
            flag = model.NewBoolVar(f"violate{len(flags)}")
            model.Add(lhs > c["rhs"] if c["op"] == "le" else lhs < c["rhs"] if c["op"] == "ge" else lhs != c["rhs"]).OnlyEnforceIf(flag)
            flags.append(flag)
        if not flags:
            continue
        model.AddBoolOr(flags)
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = seconds
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = 0
        if solver.Solve(model) in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            bits = [solver.Value(v) for v in x]
            gf = bool(np.all(violations(gold, bits) == 0))
            cf = bool(np.all(violations(candidate, bits) == 0))
            if gf == cf:
                raise AssertionError("Counterexample failed independent numeric verification")
            return {"kind": "constraints", "x": bits, "gold_feasible": gf, "candidate_feasible": cf}
    return None


def mutations(spec):
    """Plausible, schema-valid mistakes. Never label redundant edits as incorrect."""
    bad = deepcopy(spec)
    bad["objective"]["sense"] = "max" if spec["objective"]["sense"] == "min" else "min"
    yield "objective_direction", bad
    bad = deepcopy(spec)
    bad["objective"]["linear"][0] += 1
    yield "objective_coefficient", bad
    if spec["objective"]["quadratic"]:
        bad = deepcopy(spec)
        bad["objective"]["quadratic"][0][2] *= -1
        yield "quadratic_sign", bad
    if spec["constraints"]:
        bad = deepcopy(spec)
        bad["constraints"].pop(0)
        yield "missing_constraint", bad
        bad = deepcopy(spec)
        c = bad["constraints"][0]
        c["op"] = "ge" if c["op"] == "le" else "le"
        yield "constraint_relation", bad
        bad = deepcopy(spec)
        bad["constraints"][0]["rhs"] += 1
        yield "constraint_bound", bad
        bad = deepcopy(spec)
        a = bad["constraints"][0]["a"]
        nz = [i for i, v in enumerate(a) if v]
        if len(nz) > 1:
            for i in nz[1:]:
                a[i] = 0
            yield "sum_to_single_variable", bad


def verified_preference(case, candidate, policy, error_type):
    gold = case["gold"]
    validate(candidate)
    witness = objective_witness(gold, candidate) or constraint_witness(gold, candidate)
    if witness is None:
        return None
    return {"case_id": case["id"], "task": "semantic_modeling", "error_type": error_type,
            "prompt": joint_prompt(case["description"]), "chosen": completion(gold, policy),
            "rejected": completion(candidate, policy), "witness": witness}


def verify_witness(gold, candidate, witness):
    if witness["kind"] == "dimension":
        return gold["n"] != candidate["n"]
    x = witness.get("x", [])
    if len(x) != gold["n"] or any(type(v) is not int or v not in (0, 1) for v in x):
        return False
    if witness["kind"] == "objective":
        sg = 1 if gold["objective"]["sense"] == "min" else -1
        sc = 1 if candidate["objective"]["sense"] == "min" else -1
        return bool(sg * objective(gold, x) != sc * objective(candidate, x))
    if witness["kind"] == "constraints":
        return bool(np.all(violations(gold, x) == 0)) != bool(np.all(violations(candidate, x) == 0))
    return False


def semantic_metrics(predicted, gold):
    from .benchmark import semantic_equal
    obj_match = semantic_equal({**predicted, "constraints": []}, {**gold, "constraints": []})
    empty = {"sense": "min", "linear": [0] * gold["n"], "quadratic": []}
    con_match = predicted["n"] == gold["n"] and semantic_equal({**predicted, "objective": empty}, {**gold, "objective": empty})
    return {"objective_match": obj_match, "constraint_match": con_match,
            "semantic_match": obj_match and con_match}
