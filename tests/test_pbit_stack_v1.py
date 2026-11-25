"""Regression checks for the staged and penalty-specific evaluation."""

from copy import deepcopy
from importlib.util import find_spec

import numpy as np
import pytest

from experiments.pbit_dual_v1 import common as prior
from experiments.pbit_joint_v1 import search
from experiments.pbit_stack_v1 import common, cpu, feedback, penalty, report
from qihc.experiment_runtime.storage import load, save
from qihc.generic.bounded import VERSION
from qihc.generic.feedback.verification import normalized
from qihc.modeling.active_protocol_v2 import parse_model


def model():
    return dict(
        schema=VERSION,
        variables=[dict(name=f"x{i}", lower=0, upper=1) for i in range(3)],
        objective=dict(sense="max", linear=[3, 2, 1], quadratic=[]),
        constraints=[dict(id="limit", a=[1, 1, 1], op="le", rhs=1)],
        logic=[],
    )


def test_penalty_keeps_protected_population_and_fixed_budget():
    spec = model()
    before = deepcopy(spec)
    cfg = dict(chains=8, window_steps=16, decisions=3)
    old = search.episode(spec, "one", dict(cfg, seed=17), lambda *_: "cool")
    new = penalty.episode(spec, "one", 17, lambda *_: "fixed", cfg)
    assert new["x"] == old["x"]
    assert new["attempts"] == old["heat_bath_attempts"] == 8 * 16 * 4
    st = search.initial(spec, 5, 8)
    st["x"] = [[1, 1, 1]] * 8
    base = penalty.advance(spec, st, "fixed", 1, 16)
    explored = penalty.advance(spec, st, "soften", 1, 16)
    assert explored["x"][:4] == base["x"][:4]
    assert explored["penalty_writes"] > 0
    assert spec == before
    rows = penalty.labels(spec, "one", cfg)
    assert {r["action"] for r in rows} == set(penalty.ACTIONS[1:])
    assert np.isfinite([r["target"] for r in rows]).all()


def test_objective_augmented_repair_is_verified_and_local():
    case = dict(
        id="example",
        question="Pick at most one to maximize reward.",
        variable_names=["x0", "x1", "x2"],
    )
    row = feedback.objective_example(case, model())
    assert row["cards"][0]["kind"] == "objective"
    patch = feedback.repair_example(row)
    repaired = parse_model(patch["target"], row["case"], row["candidate"])
    assert normalized(repaired) == normalized(row["gold"])
    assert "draft_residuals" in patch["prompt"]


def test_current_then_improved_scores_and_resume(tmp_path, monkeypatch):
    cp = pytest.importorskip("ortools.sat.python.cp_model")
    if not find_spec("pyscipopt"):
        pytest.skip("Requires PySCIPOpt")
    for module in (common, cpu, report):
        monkeypatch.setattr(module, "RUN", tmp_path / "stack")
    monkeypatch.setattr(prior, "RUN", tmp_path / "prior")
    case = dict(dataset="dcp", main=True, id="example", template="fixture")
    pred = model()
    cp_model = cp.CpModel()
    xs = [cp_model.new_bool_var(f"x{i}") for i in range(3)]
    cp_model.add(sum(xs) <= 1)
    cp_model.maximize(3 * xs[0] + 2 * xs[1] + xs[2])
    ref = tmp_path / "ref.json"
    save(ref, dict(proto=str(cp_model.proto), indices=[x.index for x in xs], optimum=3))
    save(prior.original_artifact(case), dict(reference=str(ref)))
    save(common.RUN / "cases.json", [case])
    save(common.RUN / "manifest.json", dict(code="fixture"))
    save(
        common.RUN / "historical18.json",
        {"scores": {g: {"arms": {}} for g in ("dcp235", "dcp159")}},
    )
    save(common.path("archived", case, "matched"), {})
    archive = dict(hit=True, feasible=True, status="evaluated", gap=0.0, valid=True)
    save(
        common.path("archived", case, "scores"),
        {f"incumbent__schedule__{s}": archive for s in (17, 29, 43)},
    )
    for stage in ("current", "improved"):
        for name, terminal in common.arms(case, stage):
            save(common.path("predictions", case, name), dict(model=pred))
            if terminal == "learned_penalty":
                for seed in common.seed_list(case, terminal):
                    save(
                        common.path("episodes", case, f"{name}__{terminal}", seed),
                        {"reuse_schedule": True},
                    )
        cpu.score_case((case, stage))
        cpu.score_case((case, stage))
        result = report.report(stage, require_complete=True)["scores"]["dcp159"]
        assert (
            result["complete"]
            == result["expected"]
            == (16 if stage == "current" else 52)
        )
        assert result["arms"]["incumbent__scip"]["hits"] == 1
        assert load(common.path("scores", case, "incumbent__schedule", 17))["reused"]
