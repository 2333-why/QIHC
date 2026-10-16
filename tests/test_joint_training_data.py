import json
import sys
import pytest

from experiments.prepare_joint_training_data import load_llm_audits, main
from qihc.problems.cvrp.instance import generate_synthetic_instance, save_jsonl
from qihc.s2e.cpp import ConstraintProgramPackage


def test_canonical_proposal_matches_inference_order_and_discards_metadata():
    from experiments.prepare_joint_training_data import canonical_proposal
    payload = {"destroy_customers": [2, 2], "candidate_routes": {"2": [0, 1, 1]},
               "candidate_route_logits": {"2": {"0": 0.7777777, "1": -8.1}},
               "confidence": 0.987654, "retained_model_customers": 1, "recovered_truncated_logits": True}
    result = canonical_proposal(payload)
    parsed = json.loads(result)
    assert list(parsed) == ["destroy_customers", "candidate_routes", "candidate_route_logits", "confidence"]
    assert parsed["candidate_route_logits"] == {"2": {"0": 1, "1": -2}}
    assert parsed["destroy_customers"] == [2] and parsed["candidate_routes"] == {"2": [0, 1]}
    assert " " not in result
    assert canonical_proposal(parsed) == result


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_canonical_proposal_rejects_nonfinite_values(bad):
    from experiments.prepare_joint_training_data import canonical_proposal
    with pytest.raises(ValueError, match="Nonfinite"):
        canonical_proposal({"destroy_customers": [1], "candidate_routes": {"1": [0]},
                            "candidate_route_logits": {"1": {"0": bad}}})


def test_joint_training_data_contains_constraint_and_pbit_proposal(tmp_path, monkeypatch):
    instance = generate_synthetic_instance(6, 3, 30, seed=19, semantic_constraints=False)
    instance.description = "Minimize distance while respecting capacity."
    data = tmp_path / "data.jsonl"
    save_jsonl([instance], data)
    cpp = ConstraintProgramPackage.from_specs(
        instance.name, instance.description, instance.customer_ids, instance.constraints
    )
    constraint_records = tmp_path / "constraints.jsonl"
    constraint_records.write_text(
        json.dumps({"instance": instance.name, "status": "ok", "cpp": cpp.to_dict()}) + "\n",
        encoding="utf-8",
    )
    solver_results = tmp_path / "results.json"
    solver_results.write_text(
        json.dumps([{
            "instance": instance.name,
            "method": "llm",
            "status": "ok",
            "problem_prompt": instance.description,
            "constraints": [],
            "records": [
                {"accepted": True, "objective_improvement": 3.0, "candidate_route_recall": 1.0,
                 "candidate_compression": 0.5, "proposal_payload": {"destroy_customers": [1], "candidate_routes": {"1": [0, 1]}}},
                {"accepted": False, "objective_improvement": 0.0, "candidate_route_recall": 0.0,
                 "candidate_compression": 0.5, "proposal_payload": {"destroy_customers": [2], "candidate_routes": {"2": [2]}}},
            ],
            "solution": {"metadata": {"cold_start_records": [{
                "prompt": "cold numeric prompt",
                "teacher_payload": {"destroy_customers": [1], "candidate_routes": {"1": [0, 1]},
                                    "candidate_route_scores": {"1": {"0": 1.0, "1": -1.0}}},
                "rejected_payload": {"destroy_customers": [1], "candidate_routes": {"1": [0, 1]},
                                     "candidate_route_scores": {"1": {"0": -1.0, "1": 1.0}}},
                "objective_margin": 4.0,
            }]}},
        }]), encoding="utf-8"
    )
    output = tmp_path / "training"
    monkeypatch.setattr(sys, "argv", [
        "prepare_joint_training_data.py", "--data", str(data),
        "--constraint-records", str(constraint_records),
        "--solver-results", str(solver_results), "--output", str(output),
    ])
    assert main() == 0
    sft = [json.loads(line) for line in (output / "sft.jsonl").read_text(encoding="utf-8").splitlines()]
    dpo = [json.loads(line) for line in (output / "dpo.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {row["task"] for row in sft} == {"constraint", "proposal"}
    assert any(row["task"] == "proposal" for row in dpo)
    assert any(row.get("phase") == "cold_start" for row in sft)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["cold_start_sft"] == 1


def test_llm_audits_are_seed_aligned(tmp_path):
    audit = tmp_path / "llm_audit_rank0.jsonl"
    audit.write_text("\n".join([
        json.dumps({"instance": "case", "search_seed": 0, "iteration": 5, "ok": True}),
        json.dumps({"instance": "case", "search_seed": 1, "iteration": 5, "ok": True}),
        json.dumps({"instance": "case", "search_seed": 0, "iteration": 5, "phase": "cold_start", "ok": True}),
        json.dumps({"instance": "legacy", "iteration": 5, "ok": True}),
    ]) + "\n", encoding="utf-8")
    loaded = load_llm_audits(tmp_path)
    assert set(loaded) == {("case", 0, 5), ("case", 1, 5)}
