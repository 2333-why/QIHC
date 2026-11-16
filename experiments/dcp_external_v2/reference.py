"""Prepare private CP-SAT oracles and public decision conventions from a pinned release."""

import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
import os
from .common import ASSET, CENSUS, RUN, UP, frozen
from .inventory import inventory
from qihc.experiment_runtime.storage import load, save, value_digest


def upstream():
    if str(UP) not in sys.path:
        sys.path.insert(0, str(UP))


def source_model(template, data):
    upstream()
    from evaluation.reference import load_reference
    from cpmpy.expressions.variables import _BoolVarImpl, _IntVarImpl

    _BoolVarImpl.counter = _IntVarImpl.counter = 0
    source = (UP / "dataset" / template / (template + ".cpmpy.py")).read_text(
        encoding="utf8"
    )
    return load_reference(source, data)


def coordinates(template, data, model, outputs):
    import numpy as np
    from cpmpy.transformations.get_variables import get_variables_model
    from cpmpy.expressions.variables import _NumVarImpl, NegBoolView

    names = {}
    for field, array in outputs.items():
        for idx, var in np.ndenumerate(np.asarray(array, dtype=object)):
            if isinstance(var, _NumVarImpl) and not isinstance(var, NegBoolView):
                names[var.name] = field + "".join(f"[{j}]" for j in idx)
    variables = get_variables_model(model)
    if template in ("coin3_application", "flip_rows_cols"):
        variables = [v for v in variables if v.name in names]
    if template in ("capital_budget", "football"):
        # The monetary totals are determined by the purchase decisions; exposing
        # them as required IR variables exceeds the integer-domain cap.
        variables = [v for v in variables if v.name.startswith("x[")]
    if template == "session3_kidney_exchange":
        matrix = outputs["transplants"]
        variables = [
            matrix[i, j - 1]
            for i, compatible in enumerate(data["compatible"])
            for j in sorted(set(compatible))
        ]

    def key(v):
        return [
            int(s) if s.isdigit() else s
            for s in re.split(r"(\d+)", names.get(v.name, v.name))
        ]

    variables = sorted(variables, key=key)
    return variables, [names.get(v.name, v.name) for v in variables]


def make_reference(job):
    from ortools.sat.python import cp_model
    from cpmpy.solvers.ortools import CPM_ortools
    from evaluation.reference import SOLVER_OPTIONS

    cid, template, data = job
    cache = RUN / "reference" / (cid + ".json")
    if cache.exists():
        return load(cache)["aliases"]
    model, outputs = source_model(template, data)
    variables, aliases = coordinates(template, data, model, outputs)
    adapter = CPM_ortools(model)
    cp = adapter.ort_model
    indices = [int(adapter.solver_var(v).index) for v in variables]
    solver = cp_model.CpSolver()
    for k, v in SOLVER_OPTIONS.items():
        setattr(solver.parameters, k, v)
    solver.parameters.max_time_in_seconds = 120
    status = solver.solve(cp)
    if status != cp_model.OPTIMAL:
        raise RuntimeError(
            f"{cid}: official reference {solver.status_name(status)}, rerun prepare"
        )
    vector = [int(solver.value(cp.get_int_var_from_proto_index(j))) for j in indices]
    # Independent evaluation of the original CPMpy objective and constraints.
    from cpmpy.transformations.get_variables import get_variables_model
    from cpmpy.transformations.normalize import toplevel_list

    for v in get_variables_model(model):
        v._value = int(solver.value(adapter.solver_var(v)))
    optimum = int(model.objective_value())
    if optimum != round(solver.objective_value) or not all(
        c.value() for c in toplevel_list(model.constraints)
    ):
        raise ValueError(f"{cid}: reference assignment disagrees with source")
    save(
        cache,
        dict(
            indices=indices,
            aliases=aliases,
            optimum=optimum,
            x=vector,
            proto=str(cp.proto),
            template=template,
            data=data,
            seconds=solver.wall_time,
        ),
    )
    return aliases


def cohort():
    upstream()
    from evaluation.reference import available_instances

    audit = inventory(UP, ASSET)
    if audit != load(CENSUS):
        raise ValueError("Pinned census mismatch")
    release = {r["id"]: r for r in map(json.loads, ASSET.read_bytes().splitlines())}
    jobs, public = [], []
    for template in audit["templates"]:
        if template["objective"] == "satisfaction":
            continue
        t = template["template"]
        source = (UP / "dataset" / t / (t + ".cpmpy.py")).read_text(encoding="utf8")
        items = available_instances(source, release[t]["instances"])
        fields = items[0][1].keys()
        for key, raw in items:
            data = {k: raw[k] for k in fields}
            cid = t + "__" + key.replace(":", "_")
            jobs.append((cid, t, data))
            public.append(
                dict(
                    id=cid,
                    template=t,
                    main=key != "example"
                    and template["history"] == "no_known_template_match",
                    question=release[t]["description"]
                    + "\nInstance: "
                    + json.dumps(data, ensure_ascii=False, separators=(",", ":")),
                )
            )
    if len(public) != 235 or sum(r["main"] for r in public) != 159:
        raise ValueError("Unexpected cohort")
    return jobs, public


def init_worker():
    upstream()
    # Native solvers are loaded in their own processes.
    sys.modules["highspy"] = sys.modules["pyscipopt"] = None


def prepare():
    jobs, public = cohort()
    frozen(RUN / "dataset-id.json", dict(census=value_digest(load(CENSUS)), protocol=2))
    with ProcessPoolExecutor(
        max_workers=int(os.getenv("QIHC_CPU_WORKERS", "6")), initializer=init_worker
    ) as pool:
        for r, aliases in zip(public, pool.map(make_reference, jobs)):
            r["variable_names"] = [f"x{i}" for i in range(len(aliases))]
            r["question"] += (
                "\nDecision convention: x0, x1, ... represent the following source quantities in order. "
                "Return a bounded integer model in this order.\n" + json.dumps(aliases)
            )
            if not 0 < len(aliases) <= 256:
                raise ValueError(f'Unsupported projection: {r["id"]} ({len(aliases)})')
    frozen(RUN / "public.json", public)
    ledger = []
    for r in public:
        ref = load(RUN / "reference" / (r["id"] + ".json"))
        if (
            len(set(ref["aliases"])) != len(ref["aliases"])
            or not check_vector(ref, ref["x"])["hit"]
        ):
            raise ValueError("Official witness projection failed: " + r["id"])
        ledger.append(
            dict(
                id=r["id"],
                template=r["template"],
                main=r["main"],
                variables=len(ref["aliases"]),
                optimality_proven=True,
                optimal_witness_checked=True,
            )
        )
    frozen(RUN / "coverage.json", ledger)
    print(
        "Verified references:",
        len(public),
        "main:",
        sum(r["main"] for r in public),
        flush=True,
    )


def check_vector(reference, x):
    from ortools.sat.python import cp_model

    if x is None:
        return dict(feasible=False, hit=False, gap=None)
    model = cp_model.CpModel()
    model.proto.parse_text_format(reference["proto"])
    if len(x) != len(reference["indices"]):
        raise ValueError("Wrong solution dimension")
    for i, v in zip(reference["indices"], x):
        if int(v) != v:
            return dict(feasible=False, hit=False, gap=None)
        model.add(model.get_int_var_from_proto_index(i) == int(v))
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.keep_all_feasible_solutions_in_presolve = True
    solver.parameters.max_time_in_seconds = 120
    status = solver.solve(model)
    if status == cp_model.INFEASIBLE:
        return dict(feasible=False, hit=False, gap=None)
    if status != cp_model.OPTIMAL:
        raise RuntimeError(
            "Reference check inconclusive: " + solver.status_name(status)
        )
    val = int(round(solver.objective_value))
    optimum = reference["optimum"]
    return dict(
        feasible=True,
        objective=val,
        hit=val == optimum,
        gap=abs(val - optimum) / max(1, abs(optimum)),
    )


if __name__ == "__main__":
    prepare()
