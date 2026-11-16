"""Regression checks for the DCP cohort, paired denominators and reference scoring."""

import pytest
from qihc.experiment_runtime.storage import load
from experiments.dcp_external_v2 import common, report


def test_frozen_population_and_planned_evaluations():
    census = load(common.CENSUS)["templates"]
    cohort = [
        (r, key)
        for r in census
        if r["objective"] != "satisfaction"
        for key in r["instances"]
    ]
    main = [
        (r, key)
        for r, key in cohort
        if r["history"] == "no_known_template_match" and key != "example"
    ]
    assert (len(cohort), len(main), len({r["template"] for r, _ in main})) == (
        235,
        159,
        40,
    )
    assert len(set(common.ARMS)) == len(common.ARMS) == 40
    assert sum(len(common.repeats(*arm)) for arm in common.ARMS) == 86


def test_scoring_distinguishes_projected_feasibility_and_optimality():
    cp = pytest.importorskip("ortools.sat.python.cp_model")
    from experiments.dcp_external_v2.reference import check_vector

    model = cp.CpModel()
    x = model.new_int_var(0, 2, "decision")
    aux = model.new_int_var(0, 2, "internal")
    model.add(aux + x == 2)
    model.maximize(aux)
    ref = dict(proto=str(model.proto), indices=[x.index], optimum=2)
    assert check_vector(ref, [0])["hit"]
    row = check_vector(ref, [1])
    assert row["feasible"] and not row["hit"] and row["gap"] == 0.5
    assert not check_vector(ref, [3])["feasible"]
    assert not check_vector(ref, None)["hit"]


def test_missing_paired_rows_and_template_weighting():
    subset = [dict(id=str(i), template="a" if i == 0 else "b") for i in range(3)]

    def row(hit):
        return dict(
            hit=hit,
            feasible=hit,
            valid=True,
            status="evaluated",
            gap=0 if hit else None,
        )

    raw = {str(i): row(i == 0) for i in range(3)}
    result = report.summarize(raw, subset)
    assert result["hit_rate"] == pytest.approx(1 / 3)
    assert result["template_macro_hit"] == 0.5
    assert result["gap_cases"] == 1
    assert report.summarize({}, subset)["hit_rate"] is None
    data = dict(old=raw, new=dict(raw))
    del data["new"]["0"]
    assert report.contrast(data, subset, [("old", 1), ("new", -1)]) is None
    data["new"]["0"] = row(False)
    delta = report.contrast(data, subset, [("old", 1), ("new", -1)])
    assert delta["positive"] == ["0"] and delta["delta"] == pytest.approx(1 / 3)
