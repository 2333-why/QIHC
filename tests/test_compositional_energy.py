import copy
import itertools
import json
import pytest

from qihc.generic.benchmark import make_case
from qihc.generic.sparse import encode, decode, parse
from qihc.generic.compositional import dataset, training_rows, make_case as composed
from qihc.generic.energy import violations
from qihc.generic.semantic import verify_witness


@pytest.mark.parametrize("family,size", [("knapsack", 12), ("vertex_cover", 20), ("balanced_cut", 12), ("assignment", 4)])
def test_roundtrip(family, size):
    gold = make_case(family, size, 17)["gold"]
    assert decode(encode(gold)) == gold


def test_defaults_and_rejections():
    gold = make_case("knapsack", 12, 17)["gold"]
    wire = encode(gold)
    del wire["objective"]["quadratic"]
    assert decode(wire) == gold
    bad = copy.deepcopy(wire)
    bad["objective"]["linear"] += [bad["objective"]["linear"][0]]
    with pytest.raises(ValueError):
        decode(bad)
    wire["constraints"][0]["terms"][0][0] = True
    with pytest.raises(ValueError):
        decode(wire)


def test_composition_split_and_preferences():
    cases = dataset(20)
    assert len({c["id"] for c in cases}) == 60
    train = {tuple(c["rules"]) for c in cases if c["split"] == "train"}
    test = {tuple(c["rules"]) for c in cases if c["split"] == "test"}
    assert train.isdisjoint(test)
    sft, dpo = training_rows(cases)
    allowed = {c["id"]: c for c in cases if c["split"] == "train"}
    assert len(sft) == 60
    assert {r["error_type"] for r in dpo} >= {"quadratic_sign", "constraint_bound"}
    for row in sft + dpo:
        assert row["case_id"] in allowed
    for row in dpo:
        gold = allowed[row["case_id"]]["gold"]
        assert parse(row["chosen"]) == gold
        assert verify_witness(gold, parse(row["rejected"]), row["witness"])


def test_generated_rules_jointly_feasible():
    for seed in range(10):
        case = composed(6, seed, ("budget", "quota", "implication", "exclusion", "balance"), "test")
        assert any((violations(case["gold"], x) == 0).all() for x in itertools.product((0, 1), repeat=6))


def test_replay_checks_original_constraints_and_coverage():
    from experiments.compositional_energy import replay
    case = composed(6, 10, ("budget", "quota"), "test")
    rows = replay([case], [{"case_id": case["id"], "response": json.dumps(encode(case["gold"]))}], 8, 2, 1)
    assert len(rows) == 2
    assert {r["feedback"] for r in rows} == {True, False}
    assert all(r["semantic_match"] for r in rows)
    with pytest.raises(ValueError):
        replay([case], [])
