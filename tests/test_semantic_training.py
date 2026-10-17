from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from qihc.generic.benchmark import make_case, make_dataset
from qihc.generic.energy import violations
from qihc.generic.semantic import (completion, constraint_witness, infer_joint, joint_prompt,
                                  mutations, objective_witness, semantic_metrics, verified_preference, verify_witness)
from experiments import generic_energy as runner
from scripts.generic_energy_pipeline import training_command


@pytest.mark.parametrize("family,size", [("knapsack", 6), ("vertex_cover", 6), ("assignment", 3), ("balanced_cut", 6)])
def test_mutations_need_actual_witness(family, size):
    case = make_case(family, size, 42)
    count = 0
    for name, bad in mutations(case["gold"]):
        pair = verified_preference(case, bad, "adaptive", name)
        if pair:
            count += 1
            assert verify_witness(case["gold"], bad, pair["witness"])
            assert json.loads(pair["chosen"])["policy"] == json.loads(pair["rejected"])["policy"]
    assert count >= 3


def test_redundant_constraint_not_a_negative_and_equivalent_objective():
    spec = make_case("knapsack", 4, 10)["gold"]
    dup = deepcopy(spec)
    dup["constraints"].append({**deepcopy(spec["constraints"][0]), "id": "duplicate"})
    assert constraint_witness(spec, dup) is None
    assert objective_witness(spec, dup) is None
    opposite = deepcopy(spec)
    opposite["objective"]["sense"] = "min"
    opposite["objective"]["linear"] = [-c for c in spec["objective"]["linear"]]
    assert objective_witness(spec, opposite) is None


def test_witness_finds_missing_constraint_even_when_objective_improves():
    case = make_case("knapsack", 6, 1)
    bad = {**case["gold"], "constraints": []}
    row = verified_preference(case, bad, "fixed", "drop")
    assert row["witness"]["kind"] == "constraints"
    assert not row["witness"]["gold_feasible"] and row["witness"]["candidate_feasible"]
    assert np.sum(violations(case["gold"], row["witness"]["x"])) > 0


def test_joint_inference_no_gold_or_teacher_forcing():
    case = make_case("assignment", 3, 3)
    class Generator:
        def generate(self, prompt):
            assert prompt == joint_prompt(case["description"])
            assert "Previously translated" not in prompt
            return completion(case["gold"], "warm")
    predicted, policy, raws = infer_joint(Generator(), case["description"])
    assert predicted == case["gold"] and policy == "warm" and len(raws) == 1
    assert all(semantic_metrics(predicted, case["gold"]).values())


def fixture_teacher(tmp_path, monkeypatch):
    cases = [make_case("knapsack", 5, 2), make_case("assignment", 2, 3),
             make_case("balanced_cut", 6, 4), make_case("knapsack", 5, 9, "test")]
    monkeypatch.setattr(runner, "solve", lambda spec, policy, **kw: {
        "feasible": True, "objective": {"fixed": 0, "adaptive": 1, "warm": 2, "cold": 3}[policy], "violation": 0})
    args = SimpleNamespace(output=tmp_path, holdout_family="balanced_cut", steps=2, chains=2, semantic_training=True)
    runner.teacher(args, cases)
    return args, cases


def test_teacher_audit_leakage_and_tampering(tmp_path, monkeypatch):
    args, cases = fixture_teacher(tmp_path, monkeypatch)
    result = runner.audit_teacher(tmp_path, cases, "balanced_cut")
    assert result["training_cases"] == result["sft_rows"] == 2
    assert result["semantic_pairs"] >= 6
    rows = runner.read_rows(tmp_path / "dpo_semantic.jsonl")
    rows[0]["witness"]["x"] = [0] * 5
    runner.write_rows(tmp_path / "dpo_semantic.jsonl", rows)
    runner.write_rows(tmp_path / "dpo.jsonl", rows + runner.read_rows(tmp_path / "dpo_policy.jsonl"))
    with pytest.raises(RuntimeError, match="Unverified"):
        runner.audit_teacher(tmp_path, cases, "balanced_cut")


def test_mining_only_training_not_holdout(tmp_path, monkeypatch):
    args, cases = fixture_teacher(tmp_path, monkeypatch)
    args.mine_cases = 100
    descriptions = []
    class Generator:
        def __init__(self, args):
            pass
        def generate(self, prompt):
            case = next(c for c in cases if prompt == joint_prompt(c["description"]))
            descriptions.append(case["id"])
            bad = deepcopy(case["gold"])
            bad["objective"]["linear"][0] += 7
            return completion(bad, "warm")
    monkeypatch.setattr(runner, "Generator", Generator)
    runner.mine_semantic_negatives(args, cases)
    assert set(descriptions) == {cases[0]["id"], cases[1]["id"]}
    assert runner.audit_teacher(tmp_path, cases, "balanced_cut")["semantic_pairs"] > 6


def test_dual_gpu_command_forks_sft_with_matched_batch():
    for stage in ("sft", "dpo"):
        command = training_command("model", "data", "output", stage, 128, 2, "sft_adapter")
        assert "--nproc_per_node=2" in command and "--deepspeed-config" in command
        assert command[command.index("--gradient-accumulation-steps") + 1] == "2"
        if stage == "dpo":
            assert command[command.index("--adapter-path") + 1] == "sft_adapter"


def test_expanded_dataset_is_disjoint_and_has_size_holdout():
    cases = make_dataset(192, expanded=True)
    assert len(cases) == 768 and len({c["id"] for c in cases}) == 768
    assert all(not c["size_ood"] for c in cases if c["split"] == "train")
    assert any(c["size_ood"] for c in cases if c["split"] == "test")
    assert sum(c["split"] == "train" and c["family"] != "balanced_cut" for c in cases) == 384
