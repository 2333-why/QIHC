from collections import Counter
import json
import pytest
from qihc.generic.compositional import dataset
from qihc.generic.direct_control import matched_data, evaluate_direct, parse_solution
from qihc.generic.energy import objective, violations


def test_matched_control_never_calls_pbit(monkeypatch):
    import qihc.generic.energy as energy
    def forbidden(*a, **k):
        raise AssertionError("No pbit allowed")
    monkeypatch.setattr(energy, "solve", forbidden)
    cases = dataset(20)
    data = matched_data(cases, seconds=1)
    assert data["budget"]["training_cases"] > 0
    for stage in ("sft", "dpo"):
        assert Counter(r["case_id"] for r in data["modeling_" + stage]) == Counter(r["case_id"] for r in data["direct_" + stage])
    train = {c["id"]: c["gold"] for c in cases if c["split"] == "train"}
    for row in data["direct_dpo"]:
        spec = train[row["case_id"]]
        good = parse_solution(row["chosen"], spec["n"])
        bad = parse_solution(row["rejected"], spec["n"])
        assert (violations(spec, good) == 0).all()
        sign = 1 if spec["objective"]["sense"] == "min" else -1
        assert not (violations(spec, bad) == 0).all() or sign * objective(spec, good) < sign * objective(spec, bad)


def test_direct_failures_count_and_coverage():
    cases = dataset(20)
    preds = [{"case_id": c["id"], "response": '{"x":[true]}'} for c in cases if c["split"] != "train"]
    rows = evaluate_direct(cases, preds)
    assert len(rows) == 40
    assert not any(r["valid"] or r["feasible"] for r in rows)
    with pytest.raises(ValueError):
        evaluate_direct(cases, preds[:-1])
    with pytest.raises(ValueError):
        parse_solution(json.dumps({"x": [True]}), 1)
