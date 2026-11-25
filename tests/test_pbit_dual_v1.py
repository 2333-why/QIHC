"""Check energy observations, augmentation and the archived-baseline path."""

import numpy as np
import pytest

from experiments.pbit_dual_v1 import common, cpu, methods, report
from qihc.experiment_runtime.storage import load, save
from qihc.generic.bounded import VERSION, violations
from qihc.generic.feedback.search import Energy


def spec():
    return dict(
        schema=VERSION,
        variables=[dict(name=f"x{i}", lower=0, upper=1) for i in range(3)],
        objective=dict(sense="max", linear=[1, 1, 1], quadratic=[]),
        constraints=[dict(id="cap", a=[1, 1, 1], op="le", rhs=1)],
        logic=[],
    )


def test_energy_probe_and_verified_augmentation():
    gold = spec()
    draft = spec()
    draft["constraints"][0]["rhs"] = 2
    case = dict(
        id="test", variable_names=["x0", "x1", "x2"], question="Pick at most one."
    )
    energy = Energy(gold)
    xs = np.array([[1, 1, 1], [0, 1, 0]])
    observed = methods.landscape(energy, xs)
    np.testing.assert_allclose(observed[:, 0], energy.values(xs)[2])
    assert observed[0, 0] > observed[0, 1] > observed[0, 2]
    states, telemetry = methods.probe((draft, gold), 1, steps=8, chains=8)
    assert len(states) == 80 and telemetry["attempts"] == 640
    assert all(len(row) == 3 for row in telemetry["response"])
    variants = methods.clamp_examples(case, gold)
    assert len(variants) == 2
    for row in variants:
        assert violations(row["gold"], row["witness"]).sum() == 0
        assert violations(row["gold"], row["cards"][0]["x"]).sum() > 0
        assert violations(row["candidate"], row["cards"][0]["x"]).sum() == 0
    cfg = dict(chains=4, window_steps=8, decisions=2)
    control = methods.training_controls(gold, "test", cfg, (0.5,))
    assert len(control) >= 2
    assert all(-1 <= row["target"] <= 1 for row in control)


def test_unchanged_new_models_reuse_historical_results(tmp_path, monkeypatch):
    for module in (common, cpu, report):
        monkeypatch.setattr(module, "RUN", tmp_path)
    case = dict(dataset="nlco", id="test", stage="main")
    gold = spec()
    old = dict(
        score=dict(
            optimal_hit=True, original_feasible=True, reference_proven=True, gap=0.0
        ),
        invalid=False,
        semantic="verified_correct",
        actions=["cool"],
    )
    save(
        tmp_path / "frozen/nlco/test.json",
        dict(
            prediction={"model": gold},
            gold=gold,
            reference={},
            legacy={"weighted__schedule": old},
            legacy14={},
        ),
    )
    for model, terminal in common.methods(case):
        if model != "weighted":
            save(common.artifact("predictions", case, model), {"model": gold})
        if terminal == "control":
            save(
                common.artifact("episodes", case, f"{model}__control", 17),
                dict(reuse_schedule=True),
            )
    monkeypatch.setattr(
        cpu.search,
        "episode",
        lambda *a: (_ for _ in ()).throw(
            AssertionError("Previously scored terminal was run again")
        ),
    )
    cpu.score_case(case)
    cpu.score_case(case)
    for model, terminal in common.methods(case):
        row = load(common.artifact("scores", case, f"{model}__{terminal}", 17))
        assert row["hit"] and row["reused"] == "weighted"
    scores = report.case_results(case)
    result = report.paired([(case, scores)], "trained__control", "weighted__schedule")
    assert result["delta"] == 0
    assert result["ci95"] == [0.0, 0.0]


def test_dcp_new_arms_solve_and_score_with_native_isolation(tmp_path, monkeypatch):
    from importlib.util import find_spec

    cp = pytest.importorskip("ortools.sat.python.cp_model")
    if find_spec("pyscipopt") is None:
        pytest.skip("PySCIPOpt is not installed")
    for module in (common, cpu, report):
        monkeypatch.setattr(module, "RUN", tmp_path)
    case = dict(dataset="dcp", id="fixture", main=True, template="binary")
    target = spec()
    target["objective"]["linear"] = [3, 2, 1]
    old = spec()
    cp_ref = cp.CpModel()
    xs = [cp_ref.new_bool_var(f"x{i}") for i in range(3)]
    cp_ref.add(sum(xs) <= 1)
    cp_ref.maximize(3 * xs[0] + 2 * xs[1] + xs[2])
    ref = tmp_path / "reference.json"
    save(ref, dict(proto=str(cp_ref.proto), indices=[v.index for v in xs], optimum=3))
    historical = dict(
        hit=False, feasible=True, gap=2 / 3, status="evaluated", valid=True
    )
    legacy = {
        f"weighted__{arm}__{seed}": historical
        for arm in ("schedule", "scip")
        for seed in common.seeds(case, arm)
    }
    save(
        common.original_artifact(case),
        dict(prediction={"model": old}, reference=str(ref), legacy=legacy),
    )
    save(tmp_path / "cases.json", [case])
    save(tmp_path / "manifest.json", dict(code="fixture", supervision={}))
    for arm, terminal in common.methods(case):
        if arm != "weighted":
            save(
                common.artifact("predictions", case, arm),
                {"model": target if arm.startswith("trained") else old},
            )
        if terminal == "control":
            for seed in common.seeds(case, terminal):
                save(
                    common.artifact("episodes", case, f"{arm}__control", seed),
                    {"reuse_schedule": True},
                )
    cpu.score_case(case)
    cpu.score_case(case)
    report.report()
    result = load(tmp_path / "report.json")["scores"]["dcp159"]
    assert result["new_complete"] == result["new_expected"] == 25
    assert result["arms"]["trained__scip"]["hits"] == 1
    assert result["arms"]["weighted_feedback__scip"]["reused"] == 1
