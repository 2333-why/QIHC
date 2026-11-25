"""Check energy observations, augmentation and the archived-baseline path."""

import numpy as np

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
    from experiments.pbit_dual_v1 import gpu

    for module in (common, cpu, gpu, report):
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
