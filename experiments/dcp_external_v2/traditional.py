"""Six solvers on the pinned reference formulation; import one native solver per process."""

import math
import time
import sys
import numpy as np
from qihc.generic.energy import bounded_weights
from .common import SEED, SECONDS
from .reference import source_model, coordinates


def sample_source(model, variables, visible, method, seconds, started, seed):
    from cpmpy.transformations.normalize import toplevel_list
    from cpmpy.expressions.utils import argval
    from cpmpy.exceptions import IncompleteFunctionError

    constraints = toplevel_list(model.constraints)
    bounds = model.objective_.get_bounds()
    scale = max(1, abs(int(bounds[0])), abs(int(bounds[1])))
    sign = 1 if model.objective_is_min else -1
    sites = [
        (i, w) for i, v in enumerate(variables) for w in bounded_weights(v.ub - v.lb)
    ]
    order = {v.name: i for i, v in enumerate(variables)}
    take = [order[v.name] for v in visible]
    rng = np.random.default_rng(seed)
    n = 32
    bits = rng.integers(2, size=(n, len(sites)))
    states = np.tile([v.lb for v in variables], (n, 1)).astype(np.int64)
    for k, (i, w) in enumerate(sites):
        states[:, i] += w * bits[:, k]

    def evaluate(x):
        for v, value in zip(variables, x):
            v._value = int(value)
        try:
            bad = sum(not bool(argval(c)) for c in constraints)
            obj = float(argval(model.objective_)) * sign / scale
        except (IncompleteFunctionError, ZeroDivisionError, IndexError):
            return len(constraints) + 1, 1.0
        return bad, obj

    scores = [evaluate(x) for x in states]
    best = min(range(n), key=scores.__getitem__)
    key, witness = scores[best], states[best].copy()

    def energy(s):
        return 3 * s[0] + s[1]

    attempts = swaps = 0
    ladder = np.geomspace(0.02, 1.0, n)
    while sites and time.monotonic() - started < seconds:
        k = int(rng.integers(len(sites)))
        i, w = sites[k]
        progress = min(1.0, (time.monotonic() - started) / seconds)
        temp = (
            max(0.05, 0.25 / 2 ** int(5 * progress))
            if method == "pbit"
            else 0.02**progress
        )
        for j in range(n):
            change = (1 - 2 * bits[j, k]) * w
            states[j, i] += change
            trial = evaluate(states[j])
            if trial < key:
                key, witness = trial, states[j].copy()
            d = (energy(trial) - energy(scores[j])) / (
                ladder[j] if method == "pt" else temp
            )
            accept = (
                (rng.random() < 1 / (1 + math.exp(max(-700, min(700, d)))))
                if method == "pbit"
                else (math.log(max(1e-300, rng.random())) < min(0, -d))
            )
            if accept:
                bits[j, k] = 1 - bits[j, k]
                scores[j] = trial
            else:
                states[j, i] -= change
            attempts += 1
        if method == "pt":
            for j in range((attempts // n) % 2, n - 1, 2):
                ratio = (1 / ladder[j] - 1 / ladder[j + 1]) * (
                    energy(scores[j]) - energy(scores[j + 1])
                )
                if math.log(max(1e-300, rng.random())) < min(0, ratio):
                    states[[j, j + 1]] = states[[j + 1, j]]
                    bits[[j, j + 1]] = bits[[j + 1, j]]
                    scores[j], scores[j + 1] = scores[j + 1], scores[j]
                    swaps += 1
    return dict(
        x=witness[take].tolist(),
        attempts=attempts,
        swaps=swaps,
        status="sampled",
        full_assignment_feasible=not bool(key[0]),
    )


def run_gold(ref, method, seconds=SECONDS, seed=SEED):
    # CPMpy imports its solver registry eagerly. Hide foreign optional bindings
    # before importing it, especially the bundled OR-Tools HiGHS/SCIP libraries.
    for module in ("ortools", "pyscipopt", "highspy"):
        if module != {"cp_sat": "ortools", "scip": "pyscipopt", "highs": "highspy"}.get(
            method
        ):
            sys.modules[module] = None
    started = time.monotonic()
    if method == "cp_sat":
        from ortools.sat.python import cp_model

        cp = cp_model.CpModel()
        cp.proto.parse_text_format(ref["proto"])
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = SEED
        solver.parameters.keep_all_feasible_solutions_in_presolve = True
        solver.parameters.max_time_in_seconds = max(
            0.001, seconds - (time.monotonic() - started)
        )
        status = solver.solve(cp)
        x = (
            [
                int(solver.value(cp.get_int_var_from_proto_index(i)))
                for i in ref["indices"]
            ]
            if status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
            else None
        )
        result = dict(x=x, status=solver.status_name(status))
    else:
        import cpmpy as cp
        from cpmpy.transformations.get_variables import get_variables_model

        model, out = source_model(ref["template"], ref["data"])
        visible, aliases = coordinates(ref["template"], ref["data"], model, out)
        if aliases != ref["aliases"]:
            raise ValueError("Source variable order changed")
        if method in ("sa", "pt", "pbit"):
            result = sample_source(
                model,
                get_variables_model(model),
                visible,
                method,
                seconds,
                started,
                seed,
            )
        else:
            solver = cp.SolverLookup.get(method, model)
            options = (
                {"parallel/maxnthreads": 1, "randomization/randomseedshift": SEED}
                if method == "scip"
                else {
                    "threads": 1,
                    "random_seed": SEED,
                    "mip_rel_gap": 0.0,
                    "mip_abs_gap": 0.0,
                }
            )
            remaining = seconds - (time.monotonic() - started)
            ok = remaining > 0 and solver.solve(time_limit=remaining, **options)
            result = dict(
                x=[int(v.value()) for v in visible] if ok else None,
                status=str(solver.status()) if remaining > 0 else "build_time_limit",
            )
    return dict(result, seconds=time.monotonic() - started, budget_seconds=seconds)
